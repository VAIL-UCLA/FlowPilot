"""FlowPilot ONNX navigator: camera frames and a goal in, a plan and a (v, w) command out.

    nav = FlowPilotNavigator(variant="flowpilot-dst-small", device="cuda")
    vw, plan = nav.inference_vw(obs, [8.0, 0.5], ego_vw=[1.2, 0.0])                     # point goal
    vw, plan = nav.inference_vw(obs, GpsGoal(goal, robot, yaw), ego_vw=[1.2, 0.0])      # GPS waypoint
    vw, plan = nav.inference_vw(obs, RouteGoal(route, pose), ego_vw=[1.2, 0.0])         # route
    vw, plan = nav.inference_vw(obs, [8.0, 0.5], ego_vw=[1.2, 0.0], controller="pd")    # PD, not pure pursuit

Frame: x forward, y left, metres. ``obs`` is ``(B, T, 3, H, W)`` RGB in [0, 1] at 20 Hz, oldest
first, ``T`` up to ``context_size``; shorter histories are padded with their oldest frame.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from .controllers import PLAN_DT, make_controller
from .goals import (
    FRAME_DT,
    PATCH_MPP,
    RouteGoal,
    as_goals,
    ego_from_poses,
    goal_history,
    point_goal,
    pose_history,
    world_poses,
)

HF_REPO = "UCLA-VAIL/Visual-Navigation-Model-Checkpoints"
VARIANTS = {
    "flowpilot-dst-small": "flowpilot-dst-small/flowpilot_dst_fastvit_t12",
    "flowpilot-dst-dune": "flowpilot-dst-dune/flowpilot_dst_dune_vitb14",
}
GRAPH_INPUTS = ("vision", "route_patch", "goal", "ego", "action_bounds")
# Per-step [dx, dy, dyaw, v, w] bounds of the clips1k corpus, [lo; hi]: the row both exports trained with.
CLIPS1K_ACTION_BOUNDS = ((-0.0, -0.09, -0.05, 0.0, -0.85), (0.14, 0.09, 0.05, 2.76, 0.85))


def download(variant: str) -> Path:
    """Fetch a variant's ONNX graph and its sidecars from the model zoo; returns the ONNX path."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    from huggingface_hub import hf_hub_download

    for suffix in (".metadata.json", ".inputs.npz"):
        hf_hub_download(HF_REPO, VARIANTS[variant] + suffix)
    return Path(hf_hub_download(HF_REPO, VARIANTS[variant] + ".onnx"))


def front_pad(a: np.ndarray, n: int) -> np.ndarray:
    """Keep the newest ``n`` entries; a shorter array repeats its oldest entry at the front."""
    a = np.asarray(a)
    if len(a) >= n:
        return a[-n:]
    return np.concatenate([np.repeat(a[:1], n - len(a), axis=0), a], axis=0)


def resize_frames(frames, height: int, width: int) -> np.ndarray:
    """``(n, 3, H, W)`` RGB -> ``(n, 3, height, width)`` float32; area-averaged when shrinking."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.shape[-2:] == (height, width):
        return frames
    import cv2

    shrink = frames.shape[-2] >= height and frames.shape[-1] >= width
    mode = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    out = [cv2.resize(f.transpose(1, 2, 0), (width, height), interpolation=mode) for f in frames]
    return np.stack(out).transpose(0, 3, 1, 2)


class FlowPilotNavigator:
    """FlowPilot-DST navigator: point, GPS or route goals, tracked by pure pursuit or PD.

    Args:
        onnx_path: an exported graph. Omitted, ``variant`` is downloaded from the model zoo.
        variant: ``"flowpilot-dst-small"`` or ``"flowpilot-dst-dune"``.
        device: ``"cuda"`` or ``"cpu"``; falls back to the CPU without a CUDA execution provider.
        controller: the default controller, ``"pure_pursuit"`` or ``"pd"``.
        max_v, max_w: command limits of both controllers.
        action_bounds: ``(2, 5)`` per-step bounds. Omitted, they come from the export's
            ``.inputs.npz`` sidecar, else from the clips1k corpus.
    """

    multimodal = True

    def __init__(
        self,
        onnx_path: str | Path | None = None,
        variant: str = "flowpilot-dst-small",
        device: str = "cuda",
        controller: str = "pure_pursuit",
        max_v: float = 2.5,
        max_w: float = 2.0,
        action_bounds=None,
    ) -> None:
        import onnxruntime as ort

        path = Path(onnx_path) if onnx_path is not None else download(variant)
        ort.set_default_logger_severity(3)
        providers = ["CPUExecutionProvider"]
        if str(device).startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
        self._session = ort.InferenceSession(str(path), providers=providers)
        self.device = "cuda" if len(providers) > 1 else "cpu"

        shapes = {i.name: i.shape for i in self._session.get_inputs()}
        outputs = [o.name for o in self._session.get_outputs()]
        missing = sorted(set(GRAPH_INPUTS) - set(shapes))
        if missing or "modes" not in outputs or "probs" not in outputs:
            raise ValueError(f"{path.name} is not a FlowPilot-DST export: inputs {sorted(shapes)}, outputs {outputs}")
        self._outputs = ["modes", "probs"] + (["speed"] if "speed" in outputs else [])
        _, self.context_size, _, self.image_h, self.image_w = shapes["vision"]
        self.patch_px = int(shapes["route_patch"][-1])

        metadata = path.with_name(path.stem + ".metadata.json")
        times = json.loads(metadata.read_text()).get("target_times_s") if metadata.exists() else None
        self.plan_dt = float(times[0]) if times else PLAN_DT
        self.action_bounds = self._load_action_bounds(path, action_bounds)

        limits = dict(dt=self.plan_dt, max_v=max_v, max_w=max_w)
        self.controllers = {name: make_controller(name, **limits) for name in ("pure_pursuit", "pd")}
        if controller not in self.controllers:
            raise ValueError(f"controller must be one of {sorted(self.controllers)}, got {controller!r}")
        self.controller = controller
        self.reset()

    @staticmethod
    def _load_action_bounds(path: Path, action_bounds) -> np.ndarray:
        if action_bounds is None:
            sidecar = path.with_name(path.stem + ".inputs.npz")
            if sidecar.exists():
                with np.load(sidecar) as inputs:
                    action_bounds = inputs["action_bounds"]
            else:
                action_bounds = CLIPS1K_ACTION_BOUNDS
        return np.asarray(action_bounds, dtype=np.float32).reshape(1, 2, 5)

    def reset(self) -> None:
        """Clear the controller state, the last command and the streaming buffers."""
        for controller in self.controllers.values():
            controller.reset()
        self._last_vw: np.ndarray | None = None
        self._ego_speed: np.ndarray | None = None
        self._frames: deque = deque(maxlen=self.context_size)
        self._ego: deque = deque(maxlen=self.context_size)
        self._poses: deque = deque(maxlen=self.context_size)
        self.last_speed_head: np.ndarray | None = None
        self.last_feeds: dict | None = None

    def _ego_window(self, ego_vw, batch: int, frames: int) -> np.ndarray | None:
        """``ego_vw`` as ``(B, frames, 2)``: ``(2,)`` and ``(B, 2)`` hold over the window."""
        if ego_vw is None:
            return None
        ego = np.asarray(ego_vw, dtype=np.float32)
        if ego.shape[-1] != 2 or ego.ndim > 3:
            raise ValueError(f"ego_vw must be (2,), (B, 2) or (B, T, 2), got {ego.shape}")
        if ego.ndim == 1:
            ego = ego[None, None]
        elif ego.ndim == 2:
            ego = ego[:, None]
        if ego.shape[1] != 1:
            ego = np.stack([front_pad(e, frames) for e in ego])
        return np.broadcast_to(ego, (batch, frames, 2)).copy()

    def _last_command(self, env: int, batch: int, frames: int) -> np.ndarray:
        known = self._last_vw is not None and len(self._last_vw) == batch
        return np.tile(self._last_vw[env] if known else np.zeros(2, dtype=np.float32), (frames, 1))

    def _condition(self, goal, ego, env: int, batch: int, frames: int):
        """One environment's goal -> (goal ``(n, 2)`` per frame, route patches or None, ego ``(n, 2)``)."""
        if not isinstance(goal, RouteGoal):
            ego = self._last_command(env, batch, frames) if ego is None else ego
            return goal_history(goal.ego_xy(), ego, FRAME_DT), None, ego
        poses = goal.poses()
        if len(poses) > 1:  # a pose per frame: the odometry can come from them
            poses = front_pad(poses, frames)
            ego = ego_from_poses(poses, FRAME_DT) if ego is None else ego
        else:
            ego = self._last_command(env, batch, frames) if ego is None else ego
            poses = world_poses(poses[0], pose_history(ego, FRAME_DT))
        goals, patches = goal.condition(poses, self.patch_px, PATCH_MPP)
        return goals, patches, ego

    def _feeds(self, frames, goal_slots, ego, route_patch) -> dict:
        n, size = len(frames), self.context_size
        if route_patch is None:
            patch = np.zeros((size, self.patch_px, self.patch_px), dtype=np.float32)
        else:
            patch = front_pad(np.asarray(route_patch, dtype=np.float32)[-n:], size)
        return dict(
            vision=front_pad(resize_frames(frames, self.image_h, self.image_w), size)[None],
            route_patch=patch[None],
            goal=front_pad(point_goal(goal_slots), size)[None],
            ego=front_pad(np.asarray(ego, dtype=np.float32), size)[None],
            action_bounds=self.action_bounds,
        )

    def inference_trajectory(self, obs, goal=None, ego_vw=None, route_patch=None, *, goal_xy=None):
        """Run the model.

        Args:
            obs: ``(B, T, 3, H, W)`` RGB in [0, 1] at 20 Hz, oldest first, ``T <= context_size``.
            goal: a ``PointGoal``, ``GpsGoal`` or ``RouteGoal``, a list of one per environment, or a
                point goal as ``(2,)`` / ``(B, 2)`` [forward, left] metres in the newest frame's
                ego frame. ``goal_xy`` is the same point goal under the model zoo's name.
            ego_vw: measured [v m/s, w rad/s] as ``(2,)``, ``(B, 2)`` or per frame ``(B, T, 2)``.
                Omitted, a route goal with a pose per frame derives it from the poses; otherwise
                the previous command stands in, which is zero before the first call.
            route_patch: ``(B, T, P, P)`` route raster that replaces the one the goal renders.

        Returns:
            trajectory ``(B, M, W, 5)`` of [x, y, yaw, v, w], modes ranked best first, and
            scores ``(B, M)``.
        """
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 5 or obs.shape[2] != 3 or obs.shape[1] == 0:
            raise ValueError(f"obs must be (B, T, 3, H, W), got {obs.shape}")
        obs = obs[:, -self.context_size :]
        batch, frames = obs.shape[:2]
        goals = as_goals(goal_xy if goal is None else goal, batch)
        ego = self._ego_window(ego_vw, batch, frames)
        patches = None if route_patch is None else np.asarray(route_patch, dtype=np.float32)
        if patches is not None and patches.ndim == 3:
            patches = np.broadcast_to(patches, (batch, *patches.shape))

        modes, probs, speeds, ego_speed = [], [], [], []
        for b in range(batch):
            slots, rendered, ego_b = self._condition(goals[b], None if ego is None else ego[b], b, batch, frames)
            self.last_feeds = self._feeds(obs[b], slots, ego_b, rendered if patches is None else patches[b])
            out = self._session.run(self._outputs, self.last_feeds)
            order = np.argsort(-out[1][0], kind="stable")
            modes.append(out[0][0][order])
            probs.append(out[1][0][order])
            speeds.append(float(out[2].reshape(-1)[0]) if len(out) > 2 else float("nan"))
            ego_speed.append(float(ego_b[-1][0]))
        self.last_speed_head = np.asarray(speeds, dtype=np.float32)
        self._ego_speed = np.asarray(ego_speed, dtype=np.float32)
        return np.stack(modes), np.stack(probs)

    def inference_vw(self, obs, goal=None, ego_vw=None, route_patch=None, controller: str | None = None, *, goal_xy=None):
        """Run the model and track its best plan.

        ``controller`` overrides the default for this call. Returns ``vw`` ``(B, 2)`` of
        [v m/s, w rad/s] and the tracked plan ``(B, W, 5)``.
        """
        name = controller or self.controller
        if name not in self.controllers:
            raise ValueError(f"controller must be one of {sorted(self.controllers)}, got {name!r}")
        trajectory, _ = self.inference_trajectory(obs, goal, ego_vw, route_patch, goal_xy=goal_xy)
        best = trajectory[:, 0]
        v, w = self.controllers[name].step(best, ego_speed=self._ego_speed)
        self._last_vw = np.stack([v, w], axis=1)
        return self._last_vw.copy(), best

    def step(self, frame, goal, ego_vw=None, controller: str | None = None):
        """Streaming use: one new frame per call at 20 Hz; the navigator keeps the window.

        ``frame`` is ``(3, H, W)`` RGB in [0, 1]. A ``RouteGoal`` carries the robot's current pose;
        the navigator keeps the pose of every frame, and derives the odometry from them when
        ``ego_vw`` is omitted. Returns ``vw`` ``(2,)`` and the plan ``(W, 5)``.
        """
        (goal,) = as_goals(goal, 1)
        if isinstance(goal, RouteGoal):
            self._poses.append(goal.poses()[-1])
            goal = RouteGoal(goal.route, np.stack(self._poses), goal.goal_ahead_m)
            if ego_vw is None:
                ego_vw = ego_from_poses(goal.pose, FRAME_DT)[-1]
        if ego_vw is None:
            ego_vw = self._last_vw[0] if self._last_vw is not None else np.zeros(2)
        frame = np.asarray(frame, dtype=np.float32)
        self._frames.append(resize_frames(frame[None], self.image_h, self.image_w)[0])
        self._ego.append(np.asarray(ego_vw, dtype=np.float32).reshape(2))
        vw, plan = self.inference_vw(np.stack(self._frames)[None], goal, np.stack(self._ego)[None], controller=controller)
        return vw[0], plan[0]
