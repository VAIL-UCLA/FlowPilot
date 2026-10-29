"""Improved FlowPilot: a streaming route follower with no ego status and no point goal.

    nav = ImprovedFlowPilotNavigator(device="cuda")            # downloads the graph
    vw, plan = nav.step(frame, RouteGoal(route, pose))          # every 20 Hz frame, in order
    nav.reset()                                                 # between episodes

Each call takes the [previous, current] frame, the route patch at the current pose and the features of the
4 calls before it, which the navigator keeps. The route patch is the only guidance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .controllers import make_controller
from .goals import RouteGoal
from .navigator import FlowPilotNavigator, download, make_session, resize_frames


class ImprovedFlowPilotNavigator:
    """Args: ``onnx_path`` (omitted, the ``improved_flowpilot`` graph is downloaded), ``device``, the default
    ``controller`` (``"pure_pursuit"`` or ``"pd"``), command limits ``max_v`` and ``max_w``, and ``noise_scale``,
    the spread of the flow's start noise (0, the default, decodes deterministically)."""

    multimodal = True

    def __init__(self, onnx_path=None, device="cuda", controller="pure_pursuit", max_v=2.5, max_w=2.0, noise_scale=0.0):
        path = Path(onnx_path) if onnx_path is not None else download("improved_flowpilot")
        self._session, self.device = make_session(path, device)
        shapes = {i.name: i.shape for i in self._session.get_inputs()}
        self.image_h, self.image_w = shapes["vision"][-2:]
        self.patch_px = int(shapes["route_patch"][-1])
        self._buffer_shape, self._noise_shape = tuple(shapes["feature_buffer"]), tuple(shapes["noise"])
        self.action_bounds = FlowPilotNavigator._load_action_bounds(path, None)
        self.noise_scale, self._rng = noise_scale, np.random.default_rng(0)
        self.controllers = {name: make_controller(name, max_v=max_v, max_w=max_w) for name in ("pure_pursuit", "pd")}
        self.controller = controller
        self.reset()

    def reset(self) -> None:
        for controller in self.controllers.values():
            controller.reset()
        self._prev = None  # zeros on an episode's first call, as at the start of a training window
        self._features = np.zeros(self._buffer_shape, np.float32)
        self._valid = np.zeros(self._buffer_shape[:2], np.float32)
        self.last_feeds: dict | None = None

    def step(self, frame, goal, controller: str | None = None):
        """``frame`` ``(3, H, W)`` RGB in [0, 1]; ``goal`` a ``RouteGoal`` at the current pose, or its ``(P, P)``
        route patch. Returns ``vw`` ``(2,)`` and the best plan ``(80, 5)`` of [x, y, yaw, v, w] every 0.05 s."""
        patch = goal.route.render(goal.poses()[-1], self.patch_px) if isinstance(goal, RouteGoal) else goal
        frame = resize_frames(np.asarray(frame)[None], self.image_h, self.image_w)[0]
        prev = np.zeros_like(frame) if self._prev is None else self._prev
        self._prev = frame
        self.last_feeds = dict(
            vision=np.stack([prev, frame])[None],
            feature_buffer=self._features,
            frame_valid=self._valid,
            route_patch=np.asarray(patch, np.float32)[None],
            action_bounds=self.action_bounds,
            noise=(self.noise_scale * self._rng.standard_normal(self._noise_shape)).astype(np.float32),
        )
        modes, probs, feature = self._session.run(["modes", "probs", "feat_out"], self.last_feeds)
        self._features = np.concatenate([self._features[:, 1:], feature[:, None]], axis=1)
        self._valid = np.concatenate([self._valid[:, 1:], np.ones((1, 1), np.float32)], axis=1)
        plan = modes[0, int(np.argmax(probs[0]))]
        v, w = self.controllers[controller or self.controller].step(plan[None])
        return np.array([v[0], w[0]]), plan
