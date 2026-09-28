import math

import numpy as np
import pytest

from flowpilot.goals import (
    GpsGoal,
    PointGoal,
    Route,
    RouteGoal,
    as_goals,
    compass_to_yaw,
    ego_from_poses,
    goal_history,
    gps_to_local,
    point_goal,
    pose_history,
    world_poses,
)

UCLA = (34.0689, -118.4452)


def test_point_goal_encodes_distance_and_bearing():
    goal = point_goal([[3.0, 4.0], [0.0, 0.0], [0.0, -2.0]])
    assert goal == pytest.approx(np.array([[5.0, 0.6, 0.8], [0.0, 1.0, 0.0], [2.0, 0.0, -1.0]]))


def test_compass_to_yaw():
    assert compass_to_yaw(90.0) == pytest.approx(0.0)  # east
    assert compass_to_yaw(0.0) == pytest.approx(math.pi / 2)  # north
    assert abs(compass_to_yaw(270.0)) == pytest.approx(math.pi)  # west
    assert compass_to_yaw(180.0) == pytest.approx(-math.pi / 2)  # south


def test_gps_to_local_at_the_equator():
    east, north = gps_to_local([0.0, 1e-4], [0.0, 0.0])
    assert east == pytest.approx(11.1319, abs=1e-3)  # one arc of the equatorial radius
    assert north == pytest.approx(0.0)
    assert gps_to_local([1e-4, 0.0], [0.0, 0.0])[1] == pytest.approx(11.0574, abs=1e-3)


def test_gps_to_local_matches_the_geodesic_at_mid_latitude():
    # geodesic lengths from geographiclib for 1e-3 degrees at 34.0689 N: 110.9236 m and 92.3101 m
    east, north = gps_to_local([UCLA[0] + 1e-3, UCLA[1] + 1e-3], UCLA)
    assert north == pytest.approx(110.9236, abs=1e-3)
    assert east == pytest.approx(92.3101, abs=1e-3)


def test_gps_goal_is_an_ego_offset():
    north_of_robot = (UCLA[0] + 1e-4, UCLA[1])
    ahead = GpsGoal(north_of_robot, UCLA, yaw=compass_to_yaw(0.0)).ego_xy()  # facing north
    assert ahead == pytest.approx([11.092, 0.0], abs=1e-3)
    left = GpsGoal(north_of_robot, UCLA, yaw=compass_to_yaw(90.0)).ego_xy()  # facing east
    assert left == pytest.approx([0.0, 11.092], abs=1e-3)
    behind = GpsGoal(north_of_robot, UCLA, yaw=compass_to_yaw(180.0)).ego_xy()  # facing south
    assert behind == pytest.approx([-11.092, 0.0], abs=1e-3)


def test_pose_history_of_a_straight_drive():
    poses = pose_history(np.tile([1.0, 0.0], (20, 1)))
    assert poses[-1] == pytest.approx([0.0, 0.0, 0.0])
    assert poses[0] == pytest.approx([-19 * 0.05, 0.0, 0.0])


def test_goal_history_of_a_straight_drive():
    goals = goal_history([8.0, 1.0], np.tile([1.0, 0.0], (20, 1)))
    assert goals[-1] == pytest.approx([8.0, 1.0])
    assert goals[0] == pytest.approx([8.0 + 19 * 0.05, 1.0])


def test_goal_history_of_a_turn_on_the_spot():
    goals = goal_history([5.0, 0.0], np.tile([0.0, 0.5], (20, 1)))
    assert np.linalg.norm(goals, axis=-1) == pytest.approx(np.full(20, 5.0))
    assert goals[0, 1] > 0.0  # the robot turned left, so the goal used to be further left


def test_odometry_and_poses_round_trip():
    ego = np.tile([1.2, 0.3], (20, 1))
    poses = world_poses((3.0, -2.0, 0.7), pose_history(ego))
    assert poses[-1] == pytest.approx([3.0, -2.0, 0.7])
    recovered = ego_from_poses(poses)
    assert recovered[0] == pytest.approx([0.0, 0.0])
    assert recovered[1:] == pytest.approx(ego[1:], abs=2e-3)


def test_route_is_resampled_and_measured():
    route = Route([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]])
    assert route.arc[-1] == pytest.approx(15.0)
    assert np.max(np.diff(route.arc)) <= 0.25 + 1e-9


def test_route_patch_of_a_route_ahead():
    patch = Route([[0.0, 0.0], [30.0, 0.0]]).render((0.0, 0.0, 0.0))
    assert patch.shape == (80, 80) and patch.dtype == np.uint8
    assert np.all(patch[:40, 40] == 1)  # from the robot up to 20 m ahead
    assert not patch[42:].any()  # nothing behind
    assert not patch[:, :38].any() and not patch[:, 43:].any()


def test_route_patch_turns_with_the_robot():
    route = Route([[0.0, 0.0], [0.0, 30.0]])  # heads north
    facing_east = route.render((0.0, 0.0, 0.0))
    assert np.all(facing_east[40, :40] == 1)  # the route runs to the left
    assert not facing_east[:, 43:].any()
    facing_north = route.render((0.0, 0.0, math.pi / 2))
    assert np.all(facing_north[:40, 40] == 1)


def test_route_patch_marks_crosswalks():
    crosswalk = [[5.0, -2.0], [10.0, -2.0], [10.0, 2.0], [5.0, 2.0]]
    route = Route([[0.0, 0.0], [30.0, 0.0]], crosswalks=[crosswalk])
    patch = route.render((0.0, 0.0, 0.0))
    assert set(np.unique(patch)) == {0, 1, 2}
    assert np.all(patch[21:30, 40] == 2)  # 5 .. 10 m ahead = rows 30 .. 20
    assert patch[35, 40] == 1 and patch[10, 40] == 1
    assert Route([[0.0, 0.0], [30.0, 0.0]]).runs == []


def test_long_crosswalk_runs_stay_sidewalk():
    long_polygon = [[1.0, -2.0], [60.0, -2.0], [60.0, 2.0], [1.0, 2.0]]
    assert Route([[0.0, 0.0], [80.0, 0.0]], crosswalks=[long_polygon]).runs == []


def test_route_goal_lies_ahead_on_the_route():
    route = Route([[0.0, 0.0], [10.0, 0.0], [10.0, 30.0]])
    poses = world_poses((4.0, 0.3, 0.0), pose_history(np.tile([1.0, 0.0], (20, 1))))
    goals, patches = RouteGoal(route, poses[-1]).condition(poses)
    assert route.progress == pytest.approx(4.0)
    assert goals.shape == (20, 2) and patches.shape == (20, 80, 80)
    assert goals[-1] == pytest.approx([6.0, 6.0 - 0.3])  # 12 m along: 6 m east, then 6 m north
    assert goals[0] == pytest.approx([6.0 + 19 * 0.05, 5.7])


def test_route_goal_holds_at_the_end_of_the_route():
    route = Route([[0.0, 0.0], [10.0, 0.0]])
    goals, _ = RouteGoal(route, (7.0, 0.0, 0.0)).condition(np.array([[7.0, 0.0, 0.0]]))
    assert goals[-1] == pytest.approx([3.0, 0.0])
    assert route.remaining_m == pytest.approx(3.0)


def test_route_progress_does_not_jump_to_a_later_pass():
    # an out-and-back route: the return leg passes 1 m from the way out
    route = Route([[0.0, 0.0], [40.0, 0.0], [40.0, 1.0], [0.0, 1.0]])
    assert route.update_progress([5.0, 0.0]) == pytest.approx(5.0)
    assert route.update_progress([6.0, 0.8]) == pytest.approx(6.0)  # nearer to the return leg


def test_route_from_gps():
    latlon = [UCLA, (UCLA[0] + 1e-3, UCLA[1])]  # 110.92 m north
    route = Route.from_gps(latlon)
    assert route.arc[-1] == pytest.approx(110.92, abs=0.02)
    pose = route.pose_from_gps((UCLA[0] + 1e-4, UCLA[1]), compass_to_yaw(0.0))
    assert pose == pytest.approx([0.0, 11.092, math.pi / 2], abs=1e-3)
    assert np.all(route.render(pose)[:40, 40] == 1)
    with pytest.raises(ValueError):
        Route([[0.0, 0.0], [1.0, 0.0]]).pose_from_gps(UCLA, 0.0)


def test_as_goals():
    assert [g.xy for g in as_goals([8.0, 0.5], 2)] == [(8.0, 0.5), (8.0, 0.5)]
    assert [g.xy for g in as_goals(np.array([[1.0, 0.0], [2.0, 0.0]]), 2)] == [(1.0, 0.0), (2.0, 0.0)]
    goal = PointGoal((1.0, 2.0))
    assert as_goals(goal, 3) == [goal] * 3
    assert as_goals([goal, goal], 2) == [goal, goal]
    for bad in (None, [goal], np.zeros((3, 2)), [1.0, 2.0, 3.0]):
        with pytest.raises(ValueError):
            as_goals(bad, 2)
