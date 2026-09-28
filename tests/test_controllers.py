import math

import numpy as np
import pytest

from flowpilot.controllers import (
    PDController,
    PurePursuitController,
    lookahead_point,
    make_controller,
    split_plan,
)

DT = 0.05


def arc_plan(speed: float, yaw_rate: float, steps: int = 80) -> np.ndarray:
    """A constant (v, w) rollout as a FlowPilot plan [x, y, yaw, v, w]."""
    t = DT * np.arange(1, steps + 1)
    yaw = yaw_rate * t
    if abs(yaw_rate) < 1e-9:
        x, y = speed * t, np.zeros_like(t)
    else:
        x, y = speed / yaw_rate * np.sin(yaw), speed / yaw_rate * (1.0 - np.cos(yaw))
    return np.stack([x, y, yaw, np.full_like(t, speed), np.full_like(t, yaw_rate)], axis=-1)


def test_lookahead_point_lies_on_the_circle_and_the_path():
    plan = arc_plan(1.5, 0.3)
    target = lookahead_point(plan[:, :2], 1.2)
    assert math.hypot(*target) == pytest.approx(1.2, abs=1e-9)
    radius = 1.5 / 0.3
    assert math.hypot(target[0], target[1] - radius) == pytest.approx(radius, abs=1e-3)


def test_lookahead_point_interpolates_the_first_segment():
    target = lookahead_point(np.array([[4.0, 0.0], [8.0, 0.0]]), 1.0)
    assert target == pytest.approx([1.0, 0.0])


def test_lookahead_point_returns_the_end_of_a_short_path():
    target = lookahead_point(np.array([[0.2, 0.0], [0.4, 0.1]]), 1.0)
    assert target == pytest.approx([0.4, 0.1])


def test_split_plan_reads_speed_from_xy_spacing():
    plan = arc_plan(1.2, 0.0)
    _, speed, batched = split_plan(plan[:, :2], DT)
    assert not batched
    assert speed[0] == pytest.approx(np.full(80, 1.2))


def test_split_plan_rejects_other_layouts():
    with pytest.raises(ValueError):
        split_plan(np.zeros((80, 3)))


@pytest.mark.parametrize("name", ["pure_pursuit", "pd"])
def test_straight_plan_drives_straight(name):
    v, w = make_controller(name).step(arc_plan(1.4, 0.0), ego_speed=1.4)
    assert v == pytest.approx(1.4, abs=1e-6)
    assert w == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("name", ["pure_pursuit", "pd"])
@pytest.mark.parametrize("yaw_rate", [0.3, -0.3])
def test_turn_direction_follows_the_plan(name, yaw_rate):
    _, w = make_controller(name).step(arc_plan(1.2, yaw_rate), ego_speed=1.2)
    assert np.sign(w) == np.sign(yaw_rate)


def test_pure_pursuit_recovers_the_yaw_rate_of_an_arc():
    controller = PurePursuitController(max_steering_angle=None)
    v, w = controller.step(arc_plan(1.2, 0.25), ego_speed=1.2)
    assert v == pytest.approx(1.2, abs=1e-6)
    assert w == pytest.approx(0.25, abs=1e-3)


def test_pure_pursuit_respects_the_steering_limit():
    controller = PurePursuitController()
    v, w = controller.step(arc_plan(1.0, 1.5), ego_speed=1.0)
    cfg = controller.config
    assert abs(controller.last_steer[0]) == pytest.approx(cfg.max_steering_angle)
    assert w == pytest.approx(v * math.tan(cfg.max_steering_angle) / cfg.wheelbase, rel=1e-6)


def test_stopped_plan_commands_a_stop():
    plan = np.zeros((80, 5))
    for name in ("pure_pursuit", "pd"):
        v, w = make_controller(name).step(plan, ego_speed=0.0)
        assert v == 0.0 and w == 0.0


def test_reversing_is_off_by_default():
    plan = arc_plan(-1.0, 0.0)
    assert PurePursuitController().step(plan)[0] == 0.0
    assert PurePursuitController(min_v=-1.0).step(plan)[0] == pytest.approx(-1.0)


def test_limits_clip_the_command():
    v, w = PurePursuitController(max_v=1.0, max_w=0.2).step(arc_plan(2.0, 1.0), ego_speed=2.0)
    assert v == pytest.approx(1.0)
    assert w == pytest.approx(0.2)


def test_pd_limits_the_acceleration_between_calls():
    controller = PDController(max_accel=1.0, control_dt=0.05)
    assert controller.step(arc_plan(0.0, 0.0), ego_speed=0.0)[0] == 0.0
    assert controller.step(arc_plan(2.0, 0.0))[0] == pytest.approx(0.05)
    assert controller.step(arc_plan(2.0, 0.0))[0] == pytest.approx(0.10)
    controller.reset()
    assert controller.step(arc_plan(2.0, 0.0))[0] == pytest.approx(2.0)


def test_pd_derivative_term_acts_on_a_changing_error():
    with_d, without_d = PDController(kd=0.1), PDController(kd=0.0)
    for controller in (with_d, without_d):
        controller.step(arc_plan(1.0, 0.0))
    assert with_d.step(arc_plan(1.0, 0.4))[1] > without_d.step(arc_plan(1.0, 0.4))[1]


def test_batched_plans_match_single_plans():
    plans = np.stack([arc_plan(1.0, 0.2), arc_plan(1.5, -0.3), arc_plan(0.5, 0.0)])
    for name in ("pure_pursuit", "pd"):
        v, w = make_controller(name).step(plans, ego_speed=[1.0, 1.5, 0.5])
        assert v.shape == w.shape == (3,)
        for i, plan in enumerate(plans):
            vi, wi = make_controller(name).step(plan, ego_speed=float([1.0, 1.5, 0.5][i]))
            assert (v[i], w[i]) == pytest.approx((vi, wi))


def test_unknown_controller_is_rejected():
    with pytest.raises(ValueError):
        make_controller("mpc")
