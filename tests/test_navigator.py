"""Runs against a real export: set FLOWPILOT_ONNX to a FlowPilot-DST ``.onnx`` with its sidecars."""

import os
from pathlib import Path

import numpy as np
import pytest

from flowpilot import GpsGoal, PointGoal, Route, RouteGoal
from flowpilot.goals import goal_history, pose_history, world_poses

ONNX = os.environ.get("FLOWPILOT_ONNX")
pytestmark = pytest.mark.skipif(not ONNX, reason="FLOWPILOT_ONNX is not set")


@pytest.fixture(scope="module")
def navigator():
    from flowpilot import FlowPilotNavigator

    return FlowPilotNavigator(onnx_path=ONNX, device="cpu")


@pytest.fixture(scope="module")
def sample():
    path = Path(ONNX)
    with np.load(path.with_name(path.stem + ".inputs.npz")) as inputs:
        return {k: inputs[k] for k in inputs.files}


def test_feeds_reproduce_the_export_sample(navigator, sample):
    goal_xy = sample["goal"][0, -1, 0] * sample["goal"][0, -1, 1:]
    slots = goal_history(goal_xy, sample["ego"][0])
    feeds = navigator._feeds(sample["vision"][0], slots, sample["ego"][0], sample["route_patch"][0])
    assert feeds["vision"] == pytest.approx(sample["vision"])
    assert feeds["ego"] == pytest.approx(sample["ego"])
    assert feeds["action_bounds"] == pytest.approx(sample["action_bounds"])
    assert feeds["goal"][0, -1] == pytest.approx(sample["goal"][0, -1])


@pytest.mark.parametrize("controller", ["pure_pursuit", "pd"])
def test_inference_vw_shapes_and_limits(navigator, sample, controller):
    navigator.reset()
    trajectory, scores = navigator.inference_trajectory(sample["vision"], [8.0, 0.0], ego_vw=sample["ego"])
    assert trajectory.shape[0] == 1 and trajectory.shape[-1] == 5
    assert scores.shape == trajectory.shape[:2]
    assert np.all(np.diff(scores[0]) <= 0.0)

    vw, plan = navigator.inference_vw(sample["vision"], [8.0, 0.0], ego_vw=sample["ego"], controller=controller)
    assert vw.shape == (1, 2) and plan.shape == (1, *trajectory.shape[2:])
    assert plan == pytest.approx(trajectory[:, 0])
    assert 0.0 <= vw[0, 0] <= 2.5 and abs(vw[0, 1]) <= 2.0


def test_short_history_and_other_frame_sizes(navigator, sample):
    navigator.reset()
    frames = np.repeat(np.repeat(sample["vision"][:, -5:], 2, axis=-1), 2, axis=-2)
    vw, plan = navigator.inference_vw(frames, [8.0, 0.0], ego_vw=[1.0, 0.0])
    assert vw.shape == (1, 2) and np.all(np.isfinite(plan))


def test_streaming_matches_the_window_call(navigator, sample):
    navigator.reset()
    for frame, ego in zip(sample["vision"][0], sample["ego"][0]):
        vw, plan = navigator.step(frame, [8.0, 0.0], ego_vw=ego)
    navigator.reset()
    expected, _ = navigator.inference_vw(sample["vision"], [8.0, 0.0], ego_vw=sample["ego"])
    assert vw == pytest.approx(expected[0], abs=1e-4)


def test_goal_types_agree_on_the_same_goal(navigator, sample):
    obs, ego = sample["vision"], sample["ego"]
    expected, _ = navigator.inference_trajectory(obs, [8.0, 0.0], ego_vw=ego)
    for goal in (PointGoal((8.0, 0.0)), GpsGoal((34.0, -118.0 + 8.0 / 92384.786), (34.0, -118.0), yaw=0.0)):
        trajectory, _ = navigator.inference_trajectory(obs, goal, ego_vw=ego)
        assert trajectory == pytest.approx(expected, abs=1e-3)
    trajectory, _ = navigator.inference_trajectory(obs, goal_xy=np.array([8.0, 0.0]), ego_vw=ego)
    assert trajectory == pytest.approx(expected)
    with pytest.raises(ValueError):
        navigator.inference_trajectory(obs, ego_vw=ego)


def test_route_goal_feeds_the_patch_and_the_goal(navigator, sample):
    obs, ego = sample["vision"], sample["ego"]
    route = Route([[0.0, 0.0], [10.0, 0.0], [10.0, 30.0]])
    trajectory, _ = navigator.inference_trajectory(obs, RouteGoal(route, (4.0, 0.0, 0.0)), ego_vw=ego)
    feeds = navigator.last_feeds
    assert np.all(np.isfinite(trajectory))
    assert set(np.unique(feeds["route_patch"])) == {0.0, 1.0}
    assert feeds["route_patch"][0, -1, 40, 40] == 1.0  # the robot stands on the route
    assert feeds["goal"][0, -1] == pytest.approx([np.hypot(6.0, 6.0), np.sqrt(0.5), np.sqrt(0.5)], abs=1e-5)

    poses = world_poses((4.0, 0.0, 0.0), pose_history(ego[0]))  # the same drive as a pose per frame
    navigator.inference_trajectory(obs, RouteGoal(route, poses))
    assert navigator.last_feeds["goal"] == pytest.approx(feeds["goal"], abs=1e-4)
    assert navigator.last_feeds["route_patch"] == pytest.approx(feeds["route_patch"])
    assert navigator.last_feeds["ego"][0, 4:] == pytest.approx(ego[0, 4:], abs=1e-3)


def test_streaming_a_route(navigator, sample):
    navigator.reset()
    route = Route([[0.0, 0.0], [40.0, 0.0]])
    for i, frame in enumerate(sample["vision"][0][:6]):
        vw, plan = navigator.step(frame, RouteGoal(route, (0.05 * i, 0.0, 0.0)))
    assert vw.shape == (2,) and plan.shape[-1] == 5
    assert navigator.last_feeds["ego"][0, -1] == pytest.approx([1.0, 0.0], abs=1e-4)  # 5 cm per 50 ms
    assert route.progress == pytest.approx(0.25)
