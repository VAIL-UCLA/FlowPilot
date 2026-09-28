"""Runs against a real export: set FLOWPILOT_ONNX to a FlowPilot-DST ``.onnx`` with its sidecars."""

import os
from pathlib import Path

import numpy as np
import pytest

from flowpilot.navigator import goal_history, point_goal

ONNX = os.environ.get("FLOWPILOT_ONNX")
needs_model = pytest.mark.skipif(not ONNX, reason="FLOWPILOT_ONNX is not set")


def test_point_goal_encodes_distance_and_bearing():
    goal = point_goal([[3.0, 4.0], [0.0, 0.0], [0.0, -2.0]])
    assert goal == pytest.approx(np.array([[5.0, 0.6, 0.8], [0.0, 1.0, 0.0], [2.0, 0.0, -1.0]]))


def test_goal_history_of_a_straight_drive():
    goals = goal_history([8.0, 1.0], np.tile([1.0, 0.0], (20, 1)))
    assert goals[-1] == pytest.approx([8.0, 1.0])
    assert goals[0] == pytest.approx([8.0 + 19 * 0.05, 1.0])


def test_goal_history_of_a_turn_keeps_the_distance_to_a_goal_on_the_axis():
    goals = goal_history([0.0, 0.0], np.tile([0.0, 0.5], (20, 1)))
    assert goals == pytest.approx(np.zeros((20, 2)), abs=1e-9)
    goals = goal_history([5.0, 0.0], np.tile([0.0, 0.5], (20, 1)))
    assert np.linalg.norm(goals, axis=-1) == pytest.approx(np.full(20, 5.0))
    assert goals[0, 1] > 0.0  # the robot turned left, so the goal used to be further left


@pytest.fixture(scope="module")
def navigator():
    from flowpilot import FlowPilotNavigator

    return FlowPilotNavigator(onnx_path=ONNX, device="cpu")


@pytest.fixture(scope="module")
def sample():
    path = Path(ONNX)
    with np.load(path.with_name(path.stem + ".inputs.npz")) as inputs:
        return {k: inputs[k] for k in inputs.files}


@needs_model
def test_feeds_reproduce_the_export_sample(navigator, sample):
    goal_xy = sample["goal"][0, -1, 0] * sample["goal"][0, -1, 1:]
    feeds = navigator._feeds(sample["vision"][0], goal_xy, sample["ego"][0], sample["route_patch"][0])
    assert feeds["vision"] == pytest.approx(sample["vision"])
    assert feeds["ego"] == pytest.approx(sample["ego"])
    assert feeds["action_bounds"] == pytest.approx(sample["action_bounds"])
    assert feeds["goal"][0, -1] == pytest.approx(sample["goal"][0, -1])


@needs_model
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


@needs_model
def test_short_history_and_other_frame_sizes(navigator, sample):
    navigator.reset()
    frames = np.repeat(np.repeat(sample["vision"][:, -5:], 2, axis=-1), 2, axis=-2)
    vw, plan = navigator.inference_vw(frames, [8.0, 0.0], ego_vw=[1.0, 0.0])
    assert vw.shape == (1, 2) and np.all(np.isfinite(plan))


@needs_model
def test_streaming_matches_the_window_call(navigator, sample):
    navigator.reset()
    for frame, ego in zip(sample["vision"][0], sample["ego"][0]):
        vw, plan = navigator.step(frame, [8.0, 0.0], ego_vw=ego)
    navigator.reset()
    expected, _ = navigator.inference_vw(sample["vision"], [8.0, 0.0], ego_vw=sample["ego"])
    assert vw == pytest.approx(expected[0], abs=1e-4)
