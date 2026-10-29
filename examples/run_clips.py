"""Run FlowPilot on recorded clips, following each clip's route; writes <clip>/<variant>.npz.

    python examples/run_clips.py [--variant flowpilot-dst-dune] mini_data/*/
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from flowpilot import FlowPilotNavigator, Route, RouteGoal

parser = argparse.ArgumentParser()
parser.add_argument("clips", nargs="+", type=Path)
parser.add_argument("--variant", default="flowpilot-dst-small")
args = parser.parse_args()
variant = args.variant
nav = FlowPilotNavigator(variant=variant, device="cuda")
T, H = nav.context_size, 80  # window frames, plan steps (4 s)

for clip in args.clips:
    load = lambda name: np.load(clip / f"{name}.npy")
    cap, frames = cv2.VideoCapture(str(clip / "front_fe_384x216.mp4")), []
    while (ok := cap.read())[0]:
        frames.append(cv2.cvtColor(ok[1], cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / np.float32(255))

    # Route: route_full.json is [lon, lat] (plain or GeoJSON); poses come from GPS + ENU yaw.
    route_json = json.loads((clip / "route_full.json").read_text())
    route = Route.from_gps(np.array(route_json.get("geometry", route_json)["coordinates"])[:, ::-1])
    poses = np.array([route.pose_from_gps(g, np.radians(y)) for g, y in zip(load("frame_gps"), load("frame_yaw_deg"))])
    ego = np.stack([load("frame_speeds"), load("frame_angular_velocities")[:, 2]], axis=1)

    # Ground truth for scoring: the odometry track, in each frame's ego frame.
    xy, q = load("frame_positions")[:, :2], load("frame_orientations")  # quaternion wxyz
    yaw = 2 * np.arctan2(q[:, 3], q[:, 0])

    out = {k: [] for k in ("t", "vw", "plan", "goal", "patch", "gt")}
    for t in range(T - 1, len(frames)):
        w = slice(t - T + 1, t + 1)
        vw, plan = nav.inference_vw(np.stack(frames[w])[None], RouteGoal(route, poses[w]), ego[w][None])
        c, s = np.cos(yaw[t]), np.sin(yaw[t])
        d = xy[t + 1 : t + 1 + H] - xy[t]
        out["t"].append(t)
        out["vw"].append(vw[0])
        out["plan"].append(plan[0])
        out["goal"].append(nav.last_feeds["goal"][0, -1])  # [distance, cos, sin] at the newest frame
        out["patch"].append(nav.last_feeds["route_patch"][0, -1].astype(np.uint8))  # the route raster it saw
        out["gt"].append(np.pad(d @ [[c, -s], [s, c]], ((0, H - len(d)), (0, 0)), constant_values=np.nan))
    out = {k: np.stack(v) for k, v in out.items()}
    np.savez(clip / f"{variant}.npz", **out)

    err = np.linalg.norm(out["plan"][..., :2] - out["gt"], axis=-1)  # (windows, 80)
    full = ~np.isnan(err).any(axis=1)
    print(f"{clip.name[:8]}  windows {len(err)}  ADE {err[full].mean():.2f} m  FDE {err[full, -1].mean():.2f} m  "
          f"v_cmd {out['vw'][:, 0].mean():.2f} vs v_gt {ego[T - 1 :, 0].mean():.2f} m/s  "
          f"goal {out['goal'][:, 0].min():.0f}-{out['goal'][:, 0].max():.0f} m")
