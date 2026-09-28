"""FlowPilot inference: an ONNX navigator and the controllers that track its plans."""

from .controllers import PDConfig, PDController, PurePursuitConfig, PurePursuitController, make_controller
from .goals import GpsGoal, PointGoal, Route, RouteGoal, compass_to_yaw, gps_to_local

__all__ = [
    "FlowPilotNavigator",
    "GpsGoal",
    "PDConfig",
    "PDController",
    "PointGoal",
    "PurePursuitConfig",
    "PurePursuitController",
    "Route",
    "RouteGoal",
    "compass_to_yaw",
    "gps_to_local",
    "make_controller",
]


def __getattr__(name: str):
    if name == "FlowPilotNavigator":  # imported on first use, so the controllers need only NumPy
        from .navigator import FlowPilotNavigator

        return FlowPilotNavigator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
