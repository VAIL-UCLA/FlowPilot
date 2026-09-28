"""Trajectory-tracking controllers: an ego-frame plan in, a (v, w) command out.

Frame: x forward, y left, metres; w is positive when turning left. A plan is ``(W, C)`` or
``(B, W, C)`` with ``C = 2`` (``[x, y]``) or ``C = 5`` (``[x, y, yaw, v, w]``, the FlowPilot layout).
Waypoint ``i`` lies ``(i + 1) * dt`` seconds ahead. An xy-only plan gets its speed from the spacing
of its waypoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

PLAN_DT = 0.05  # FlowPilot plans: 80 steps x 0.05 s = 4 s
WHEEL_BASE = 0.456  # Coco robot, metres
MAX_STEER_RAD = 0.35


def split_plan(plan, dt: float = PLAN_DT) -> tuple[np.ndarray, np.ndarray, bool]:
    """``plan`` -> (xy ``(B, W, 2)``, speed ``(B, W)``, whether the input had a batch axis)."""
    p = np.asarray(plan, dtype=np.float64)
    batched = p.ndim == 3
    if p.ndim == 2:
        p = p[None]
    if p.ndim != 3 or p.shape[1] == 0 or p.shape[2] not in (2, 5):
        raise ValueError(f"plan must be (W, C) or (B, W, C) with C = 2 or 5, got {np.shape(plan)}")
    xy = p[..., :2]
    if p.shape[2] == 5:
        return xy, p[..., 3], batched
    step = np.diff(xy, axis=1, prepend=np.zeros_like(xy[:, :1]))
    length = np.linalg.norm(step, axis=-1)
    backwards = step[..., 0] < -0.5 * length
    return xy, np.where(backwards, -length, length) / dt, batched


def time_index(time_s: float, dt: float, num_waypoints: int) -> int:
    """Index of the waypoint ``time_s`` ahead, clipped to the plan."""
    return int(min(num_waypoints - 1, max(0, round(time_s / dt) - 1)))


def lookahead_point(path_xy: np.ndarray, lookahead: float) -> np.ndarray:
    """Where ``path_xy`` ``(W, 2)`` first leaves the circle of radius ``lookahead`` around the robot.

    The robot's own position starts the path, so the first segment is interpolated too. A path
    that stays inside the circle returns its last point.
    """
    path = np.vstack([np.zeros((1, 2)), np.asarray(path_xy, dtype=np.float64)])
    outside = np.nonzero(np.linalg.norm(path, axis=-1) >= lookahead)[0]
    if outside.size == 0:
        return path[-1].copy()
    j = int(outside[0])
    if j == 0:
        return path[0].copy()
    p1, p2 = path[j - 1], path[j]
    d = p2 - p1
    a = float(d @ d)
    if a < 1e-12:
        return p2.copy()
    b = 2.0 * float(p1 @ d)
    c = float(p1 @ p1) - lookahead * lookahead
    t = (-b + math.sqrt(max(b * b - 4.0 * a * c, 0.0))) / (2.0 * a)
    return p1 + min(1.0, max(0.0, t)) * d


def _per_env(value, batch: int, fallback: np.ndarray) -> np.ndarray:
    if value is None:
        return fallback
    return np.broadcast_to(np.asarray(value, dtype=np.float64).reshape(-1), (batch,)).copy()


def _output(v: np.ndarray, w: np.ndarray, batched: bool):
    v, w = v.astype(np.float32), w.astype(np.float32)
    return (v, w) if batched else (v[0], w[0])


@dataclass
class PurePursuitConfig:
    lookahead_base: float = 1.0  # metres
    lookahead_gain: float = 0.1  # metres of lookahead per m/s of ego speed
    max_lookahead: float = 2.0
    speed_time_s: float = 1.2  # the command speed is the plan's speed this far ahead
    dt: float = PLAN_DT
    wheelbase: float = WHEEL_BASE
    max_steering_angle: float | None = MAX_STEER_RAD  # None: no steering limit (differential drive)
    min_v: float = 0.0  # negative to allow reversing
    max_v: float = 2.5
    max_w: float = 2.0


class PurePursuitController:
    """Regulated pure pursuit. The lookahead grows with the ego speed and the speed follows the plan.

    ``w = v * tan(steer) / wheelbase`` with ``steer = atan2(2 * wheelbase * sin(alpha), distance)``,
    which is ``v * 2 sin(alpha) / distance`` until the steering limit binds. ``last_target`` and
    ``last_steer`` hold the lookahead point and steering angle of the latest call.
    """

    name = "pure_pursuit"

    def __init__(self, config: PurePursuitConfig | None = None, **overrides) -> None:
        self.config = replace(config or PurePursuitConfig(), **overrides)
        self.reset()

    def reset(self) -> None:
        self._last_v: np.ndarray | None = None
        self.last_target: np.ndarray | None = None
        self.last_steer: np.ndarray | None = None

    def step(self, plan, ego_speed=None):
        """``plan`` and the measured ego speed (m/s) -> ``(v, w)``.

        Without ``ego_speed`` the lookahead uses the previous command's speed.
        """
        cfg = self.config
        xy, speed, batched = split_plan(plan, cfg.dt)
        batch, num = speed.shape
        previous = self._last_v if self._last_v is not None and len(self._last_v) == batch else np.zeros(batch)
        ego = np.abs(_per_env(ego_speed, batch, previous))

        v = np.clip(speed[:, time_index(cfg.speed_time_s, cfg.dt, num)], cfg.min_v, cfg.max_v)
        target, steer, w = np.zeros((batch, 2)), np.zeros(batch), np.zeros(batch)
        for e in range(batch):
            lookahead = min(cfg.lookahead_base + cfg.lookahead_gain * ego[e], cfg.max_lookahead)
            target[e] = lookahead_point(xy[e], lookahead)
            distance = float(np.hypot(*target[e]))
            if distance < 1e-6:
                continue
            sin_alpha = target[e, 1] / distance
            if cfg.max_steering_angle is None:
                steer[e] = math.atan2(2.0 * cfg.wheelbase * sin_alpha, distance)
                w[e] = v[e] * 2.0 * sin_alpha / distance
            else:
                raw = math.atan2(2.0 * cfg.wheelbase * sin_alpha, distance)
                steer[e] = min(cfg.max_steering_angle, max(-cfg.max_steering_angle, raw))
                w[e] = v[e] * math.tan(steer[e]) / cfg.wheelbase
        w = np.clip(w, -cfg.max_w, cfg.max_w)

        self._last_v, self.last_target, self.last_steer = v, target, steer
        return _output(v, w, batched)

    __call__ = step


@dataclass
class PDConfig:
    target_time_s: float = 1.2  # the waypoint this far ahead is tracked
    kp: float = 0.8  # rad/s per rad of heading error
    kd: float = 0.05  # rad/s per rad/s of heading error rate
    control_dt: float = 0.05  # seconds between two step() calls
    dt: float = PLAN_DT
    max_accel: float = 1.0  # m/s^2, speeding up
    max_decel: float = 2.0  # m/s^2, slowing down
    standstill_m: float = 0.05  # a target closer than this has no heading
    min_v: float = 0.0
    max_v: float = 2.5
    max_w: float = 2.0


class PDController:
    """PD on the heading error to one waypoint; the speed follows the plan under acceleration limits."""

    name = "pd"

    def __init__(self, config: PDConfig | None = None, **overrides) -> None:
        self.config = replace(config or PDConfig(), **overrides)
        self.reset()

    def reset(self) -> None:
        self._last_v: np.ndarray | None = None
        self._last_error: np.ndarray | None = None
        self.last_target: np.ndarray | None = None

    def step(self, plan, ego_speed=None):
        """``plan`` -> ``(v, w)``. ``ego_speed`` seeds the acceleration limit on the first call."""
        cfg = self.config
        xy, speed, batched = split_plan(plan, cfg.dt)
        batch, num = speed.shape
        k = time_index(cfg.target_time_s, cfg.dt, num)
        target = xy[:, k]

        moving = np.linalg.norm(target, axis=-1) >= cfg.standstill_m
        error = np.where(moving, np.arctan2(target[:, 1], target[:, 0]), 0.0)
        if self._last_error is not None and len(self._last_error) == batch:
            rate = (error - self._last_error) / cfg.control_dt
        else:
            rate = np.zeros(batch)
        w = np.clip(cfg.kp * error + cfg.kd * rate, -cfg.max_w, cfg.max_w)

        v = np.clip(speed[:, k], cfg.min_v, cfg.max_v)
        previous = self._last_v if self._last_v is not None and len(self._last_v) == batch else None
        previous = _per_env(ego_speed, batch, previous) if previous is None else previous
        if previous is not None:
            v = np.clip(v, previous - cfg.max_decel * cfg.control_dt, previous + cfg.max_accel * cfg.control_dt)
            v = np.clip(v, cfg.min_v, cfg.max_v)

        self._last_v, self._last_error, self.last_target = v, error, target.copy()
        return _output(v, w, batched)

    __call__ = step


CONTROLLERS = {PurePursuitController.name: PurePursuitController, PDController.name: PDController}


def make_controller(name: str, **overrides):
    """``"pure_pursuit"`` or ``"pd"``, with config fields as keyword overrides."""
    if name not in CONTROLLERS:
        raise ValueError(f"controller must be one of {sorted(CONTROLLERS)}, got {name!r}")
    return CONTROLLERS[name](**overrides)
