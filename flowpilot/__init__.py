"""FlowPilot inference: an ONNX navigator and the controllers that track its plans."""

from .controllers import PDConfig, PDController, PurePursuitConfig, PurePursuitController, make_controller

__all__ = [
    "FlowPilotNavigator",
    "PDConfig",
    "PDController",
    "PurePursuitConfig",
    "PurePursuitController",
    "make_controller",
]


def __getattr__(name: str):
    if name == "FlowPilotNavigator":  # imported on first use, so the controllers need only NumPy
        from .navigator import FlowPilotNavigator

        return FlowPilotNavigator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
