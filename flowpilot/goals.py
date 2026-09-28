"""Goals for the FlowPilot navigator: a point, a GPS waypoint or a route.

The graph reads two goal inputs per frame of its window: the point goal ``[distance, cos, sin]``
and the route patch, a raster of the route around the robot. Every goal here ends as those two:

- ``PointGoal``: an offset in the robot's ego frame (x forward, y left, metres).
- ``GpsGoal``: a latitude / longitude waypoint, converted to that offset.
- ``RouteGoal``: a route polyline. It renders the route patch and takes the point goal from the
  route, a fixed distance ahead of the robot.

World frames are metric with x east and y north; yaw is in radians, 0 facing east, positive
counter-clockwise. ``compass_to_yaw`` converts a compass heading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FRAME_DT = 0.05  # the observation window is sampled at 20 Hz
PATCH_PX = 80  # route patch: 80 x 80 cells
PATCH_MPP = 0.5  # metres per cell, so 40 x 40 m with the robot at the centre
GOAL_AHEAD_M = 12.0  # the route goal lies this far along the route
CROSSWALK_MAX_RUN_M = 40.0  # a longer run inside a crosswalk polygon stays sidewalk, as in training
EGO_WINDOW_STEPS = 4  # [v, w] from poses: finite differences over 0.2 s
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def compass_to_yaw(heading_deg: float) -> float:
    """Compass heading (degrees, 0 = north, clockwise) -> yaw (radians, 0 = east, counter-clockwise)."""
    yaw = math.pi / 2.0 - math.radians(heading_deg)
    return math.atan2(math.sin(yaw), math.cos(yaw))


def gps_to_local(latlon, origin) -> np.ndarray:
    """``(..., 2)`` [latitude, longitude] in degrees -> [east, north] metres from ``origin``.

    A tangent plane on the WGS84 ellipsoid at ``origin``. Against the geodesic the error is
    under 1 mm at 100 m, 2 cm at 500 m and 6 cm at 1 km, so keep the origin near the robot.
    """
    latlon = np.asarray(latlon, dtype=np.float64)
    lat0, lon0 = np.radians(np.asarray(origin, dtype=np.float64).reshape(2))
    curvature = 1.0 - WGS84_E2 * math.sin(lat0) ** 2
    meridian = WGS84_A * (1.0 - WGS84_E2) / curvature**1.5
    prime_vertical = WGS84_A / math.sqrt(curvature)
    east = prime_vertical * math.cos(lat0) * (np.radians(latlon[..., 1]) - lon0)
    north = meridian * (np.radians(latlon[..., 0]) - lat0)
    return np.stack([east, north], axis=-1)


def to_ego(points_xy, pose) -> np.ndarray:
    """World ``(..., 2)`` points -> [forward, left] of the pose ``(x, y, yaw)``."""
    points = np.asarray(points_xy, dtype=np.float64)
    x, y, yaw = (float(v) for v in pose)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = points[..., 0] - x, points[..., 1] - y
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)


def point_goal(goal_xy) -> np.ndarray:
    """``(n, 2)`` [forward, left] offsets -> ``(n, 3)`` [distance, cos, sin] of the bearing."""
    g = np.asarray(goal_xy, dtype=np.float64).reshape(-1, 2)
    d = np.hypot(g[:, 0], g[:, 1])
    far = d > 1e-6
    bearing = np.where(far[:, None], g / np.where(far, d, 1.0)[:, None], np.array([[1.0, 0.0]]))
    return np.column_stack([np.where(far, d, 0.0), bearing]).astype(np.float32)


def pose_history(ego_vw, dt: float = FRAME_DT) -> np.ndarray:
    """Odometry ``(n, 2)`` [v, w] per frame, oldest first -> ``(n, 3)`` poses in the newest frame's ego frame.

    The last row is the robot itself, ``(0, 0, 0)``.
    """
    ego = np.asarray(ego_vw, dtype=np.float64).reshape(-1, 2)
    pose = np.zeros((len(ego), 3))
    for i in range(1, len(ego)):
        v, w = 0.5 * (ego[i - 1] + ego[i])
        yaw = pose[i - 1, 2] + 0.5 * w * dt
        pose[i] = pose[i - 1] + np.array([v * math.cos(yaw) * dt, v * math.sin(yaw) * dt, w * dt])
    out = np.empty_like(pose)
    out[:, :2] = to_ego(pose[:, :2], pose[-1])
    out[:, 2] = pose[:, 2] - pose[-1, 2]
    return out


def world_poses(pose, relative) -> np.ndarray:
    """Poses ``(n, 3)`` in the ego frame of the world pose ``(x, y, yaw)`` -> world poses."""
    x, y, yaw = (float(v) for v in pose)
    relative = np.asarray(relative, dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack(
        [x + c * relative[:, 0] - s * relative[:, 1], y + s * relative[:, 0] + c * relative[:, 1], yaw + relative[:, 2]],
        axis=-1,
    )


def goal_history(goal_xy, ego_vw, dt: float = FRAME_DT) -> np.ndarray:
    """The goal ``(2,)`` of the newest frame, seen from every frame of the window: ``(n, 2)``."""
    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    return np.stack([to_ego(goal, pose) for pose in pose_history(ego_vw, dt)])


def ego_from_poses(poses, dt: float = FRAME_DT, window: int = EGO_WINDOW_STEPS) -> np.ndarray:
    """World poses ``(n, 3)`` at ``dt`` spacing -> ``(n, 2)`` [v, w] by finite differences over ``window`` steps.

    The first pose has no past and reads as a robot at rest.
    """
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    ego = np.zeros((len(poses), 2), dtype=np.float32)
    for i in range(1, len(poses)):
        k = min(window, i)
        (x0, y0, a0), (x1, y1, a1) = poses[i - k], poses[i]
        ego[i, 0] = ((x1 - x0) * math.cos(a1) + (y1 - y0) * math.sin(a1)) / (k * dt)
        ego[i, 1] = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0)) / (k * dt)
    return ego


@dataclass
class PointGoal:
    """A goal in the newest frame's ego frame: ``xy`` = [forward, left] in metres."""

    xy: tuple[float, float]

    def ego_xy(self) -> np.ndarray:
        return np.asarray(self.xy, dtype=np.float64).reshape(2)


@dataclass
class GpsGoal:
    """A GPS waypoint. ``goal`` and ``robot`` are [latitude, longitude] in degrees, ``yaw`` the robot's heading."""

    goal: tuple[float, float]
    robot: tuple[float, float]
    yaw: float  # radians, 0 = east, counter-clockwise; see compass_to_yaw

    def ego_xy(self) -> np.ndarray:
        return to_ego(gps_to_local(self.goal, self.robot), (0.0, 0.0, self.yaw))


class Route:
    """A route polyline with its crosswalks; tracks the robot's progress along it.

    Args:
        points_xy: ``(N, 2)`` route in world metres.
        crosswalks: polygons ``(K, 2)`` in the same frame. The route inside one is class 2.
        step_m: spacing the route is resampled to.
    """

    def __init__(self, points_xy, crosswalks=None, step_m: float = 0.25) -> None:
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        if len(points) < 2:
            raise ValueError("a route needs at least two points")
        dense = [points[:1]]
        for a, b in zip(points[:-1], points[1:]):
            n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step_m)))
            dense.append(a + (np.arange(1, n + 1, dtype=np.float64) / n)[:, None] * (b - a))
        self.points = np.concatenate(dense, axis=0)
        self.arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))])
        self.runs = self._crosswalk_runs(crosswalks)
        self.origin: np.ndarray | None = None
        self.progress: float | None = None

    @classmethod
    def from_gps(cls, latlon, crosswalks=None, origin=None, step_m: float = 0.25) -> "Route":
        """A route and its crosswalk polygons in [latitude, longitude]; the first point is the default origin."""
        latlon = np.asarray(latlon, dtype=np.float64).reshape(-1, 2)
        origin = latlon[0] if origin is None else np.asarray(origin, dtype=np.float64).reshape(2)
        polygons = None if crosswalks is None else [gps_to_local(p, origin) for p in crosswalks]
        route = cls(gps_to_local(latlon, origin), polygons, step_m)
        route.origin = origin
        return route

    def pose_from_gps(self, latlon, yaw: float) -> np.ndarray:
        """The robot's GPS fix and heading -> its pose ``(x, y, yaw)`` in the route's frame."""
        if self.origin is None:
            raise ValueError("this route has no GPS origin; build it with Route.from_gps")
        x, y = gps_to_local(latlon, self.origin)
        return np.array([x, y, yaw])

    def _crosswalk_runs(self, polygons) -> list[tuple[int, int]]:
        """Index runs (inclusive) of the route inside a crosswalk polygon."""
        if not polygons:
            return []
        import cv2

        inside = np.zeros(len(self.points), dtype=bool)
        for polygon in polygons:
            contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(contour) < 3:
                continue
            lo, hi = contour.min(axis=0), contour.max(axis=0)
            near = np.flatnonzero(np.all((self.points >= lo) & (self.points <= hi), axis=1))
            for k in near:
                point = (float(self.points[k, 0]), float(self.points[k, 1]))
                inside[k] |= cv2.pointPolygonTest(contour.reshape(-1, 1, 2), point, False) >= 0
        runs, k = [], 0
        while k < len(inside):
            if not inside[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(inside) and inside[j + 1]:
                j += 1
            if j > k and self.arc[j] - self.arc[k] <= CROSSWALK_MAX_RUN_M:
                runs.append((k, j))
            k = j + 1
        return runs

    def reset(self) -> None:
        self.progress = None

    def update_progress(self, xy) -> float:
        """Arc length of the route point nearest to ``xy``, searched around the last progress."""
        xy = np.asarray(xy, dtype=np.float64).reshape(2)
        idx = np.arange(len(self.points))
        if self.progress is not None:
            near = np.flatnonzero((self.arc >= self.progress - 3.0) & (self.arc <= self.progress + 15.0))
            idx = near if len(near) else idx
        k = idx[int(np.argmin(np.linalg.norm(self.points[idx] - xy, axis=1)))]
        self.progress = float(self.arc[k])
        return self.progress

    def point_at(self, arc_m: float) -> np.ndarray:
        """The route point at arc length ``arc_m``, held at the route's end."""
        k = min(int(np.searchsorted(self.arc, min(arc_m, self.arc[-1]))), len(self.points) - 1)
        return self.points[k]

    @property
    def remaining_m(self) -> float:
        """Route left after the last progress update."""
        return float(self.arc[-1] - (self.progress or 0.0))

    def render(self, pose, size: int = PATCH_PX, mpp: float = PATCH_MPP) -> np.ndarray:
        """The route around ``pose`` as ``(size, size)`` class ids: 0 background, 1 sidewalk, 2 crosswalk.

        The robot sits at the centre facing up: row 0 is ``size * mpp / 2`` metres ahead and
        column 0 that far to the left.
        """
        import cv2

        ego = to_ego(self.points, pose)
        centre = size // 2
        cells = np.stack([centre - ego[:, 1] / mpp, centre - ego[:, 0] / mpp], axis=-1)  # (column, row)
        cells = np.rint(np.clip(cells, -1.0e4, 1.0e4)).astype(np.int32)
        patch = np.zeros((size, size), dtype=np.uint8)
        cv2.polylines(patch, [cells.reshape(-1, 1, 2)], False, color=1, thickness=2)
        for a, b in self.runs:
            cv2.polylines(patch, [cells[a : b + 1].reshape(-1, 1, 2)], False, color=2, thickness=2)
        return patch


@dataclass
class RouteGoal:
    """Follow ``route`` from ``pose``.

    ``pose`` is the robot's world pose ``(x, y, yaw)`` at the newest frame, or ``(T, 3)`` with one
    pose per frame, oldest first. With a single pose the earlier ones come from the odometry.
    """

    route: Route
    pose: np.ndarray
    goal_ahead_m: float = GOAL_AHEAD_M

    def poses(self) -> np.ndarray:
        return np.asarray(self.pose, dtype=np.float64).reshape(-1, 3)

    def condition(self, poses, size: int = PATCH_PX, mpp: float = PATCH_MPP) -> tuple[np.ndarray, np.ndarray]:
        """World poses ``(n, 3)`` of the window -> (goal ``(n, 2)`` in each frame's ego frame, patches ``(n, size, size)``)."""
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        progress = self.route.update_progress(poses[-1, :2])
        goal = self.route.point_at(progress + self.goal_ahead_m)
        goals = np.stack([to_ego(goal, pose) for pose in poses])
        patches = np.stack([self.route.render(pose, size, mpp) for pose in poses])
        return goals, patches


Goal = PointGoal | GpsGoal | RouteGoal


def as_goals(goal, batch: int) -> list:
    """One goal per environment from a goal, a list of goals, or point goals as ``(2,)`` / ``(B, 2)``."""
    if goal is None:
        raise ValueError("the exports have no goal-free input: pass a point, GPS or route goal")
    if isinstance(goal, (PointGoal, GpsGoal, RouteGoal)):
        return [goal] * batch
    if isinstance(goal, (list, tuple)) and goal and isinstance(goal[0], (PointGoal, GpsGoal, RouteGoal)):
        if len(goal) != batch:
            raise ValueError(f"got {len(goal)} goals for a batch of {batch}")
        return list(goal)
    xy = np.asarray(goal, dtype=np.float64)
    if xy.shape[-1] != 2 or xy.ndim > 2 or (xy.ndim == 2 and len(xy) not in (1, batch)):
        raise ValueError(f"a point goal must be (2,) or ({batch}, 2), got {xy.shape}")
    return [PointGoal(tuple(row)) for row in np.broadcast_to(xy.reshape(-1, 2), (batch, 2))]
