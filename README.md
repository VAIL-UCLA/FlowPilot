# FlowPilot

**From Imitation to Alignment: Human-Preference Flow Policies for Long-Horizon Sidewalk Navigation**

Conference on Robot Learning (CoRL) 2026

[![arXiv](https://img.shields.io/badge/arXiv-2606.12603-blue)](https://arxiv.org/abs/2606.12603)
[![CoRL 2026](https://img.shields.io/badge/CoRL-2026-orange)](https://www.corl.org/)
[![VisNavKit](https://img.shields.io/badge/VisNavKit-training%20recipe-green)](https://github.com/VAIL-UCLA/visnavkit)

<p align="center">
  <img src="_assets/teaser.gif" alt="FlowPilot teaser" width="800">
</p>
<p align="center"><sub>GPS-guided long-horizon navigation · sidewalk lane keeping · obstacle avoidance · pedestrian awareness · night driving</sub></p>

<p align="center">
  <img src="_assets/sidewalkbench_gs_cars_crossing.gif" alt="FlowPilot on SidewalkBench-GS: crossing with parked cars" width="398">
  <img src="_assets/sidewalkbench_gs_yellow_car_crossing.gif" alt="FlowPilot on SidewalkBench-GS: crossing past a yellow car" width="398">
</p>
<p align="center"><sub>Closed-loop rollouts in SidewalkBench-GS · Gaussian-splat reconstructions of real sidewalks</sub></p>

## Setup

The model code lives in **[VisNavKit](https://github.com/VAIL-UCLA/visnavkit)**, included here as a submodule on its `dev` branch.

```bash
git clone --recurse-submodules https://github.com/VAIL-UCLA/FlowPilot.git
cd FlowPilot/visnavkit
uv sync --extra export
```

In an existing clone, run `git submodule update --init` instead.

## Checkpoints

Weights are hosted in the [VAIL model zoo](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints) on Hugging Face.

| Weights | Experiment | Model | Params | Checkpoint | ONNX |
| --- | --- | --- | --- | --- | --- |
| `flowpilot-dst-small` | `flowpilot_dst_clips1k` | FastViT-T12 frame pairs, anchored flow DiT | 21.4M | [ckpt](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints/resolve/main/flowpilot-dst-small/flowpilot_dst_fastvit_t12.ckpt) | [onnx](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints/resolve/main/flowpilot-dst-small/flowpilot_dst_fastvit_t12.onnx) |
| `flowpilot-dst-dune` | `flowpilot_dune_dst_clips1k` | frozen DUNE ViT-B/14, anchored flow DiT | 208.2M | [ckpt](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints/resolve/main/flowpilot-dst-dune/flowpilot_dst_dune_vitb14.ckpt) | [onnx](https://huggingface.co/UCLA-VAIL/Visual-Navigation-Model-Checkpoints/resolve/main/flowpilot-dst-dune/flowpilot_dst_dune_vitb14.onnx) |

Inputs, outputs and a minimal ONNX Runtime example are in the [ONNX guide](https://github.com/VAIL-UCLA/visnavkit/blob/dev/docs/flowpilot_dst_onnx.md).

**Goal-free FlowPilot.** [`FlowPilot_Official/flowpilot_base_goal_less.onnx`](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/main/FlowPilot_Official/flowpilot_base_goal_less.onnx) in the Navigation Model Zoo is a different model, with its own `inference.py` and `model_info_goal_free.yaml` there; `flowpilot` does not wrap it.

- **Camera:** pinhole at 512 x 288, fx = fy = 210.667, cx = 256, cy = 144, 0.456 m above the ground. Not the fisheye below.
- **Frames:** 21 RGB frames in [0, 1] at **5 Hz** (t-20 ... t, 4 s of history, every 4th frame of a 20 Hz stream), oldest first, `(B, 21, 3, 288, 512)`.
- **Inputs:** no goal; `embodiment_id` 0 and `metric_spacing` `[0.0, 0.51, -0.16, 0.16]`.
- **Outputs:** 64 modes of 20 `[x, y]` waypoints at 5 Hz (4 s) and their scores. The raw graph returns 65 modes: drop index 0, a register token.

## Inference

`flowpilot` wraps the ONNX exports in the interface of the [Navigation Model Zoo](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public): frames and a goal in, a plan and a `(v, w)` command out.

```bash
pip install -e .                 # from the FlowPilot root
pip install onnxruntime-gpu      # optional, replaces onnxruntime for CUDA (1.23.2 runs on CUDA 12 drivers; 1.31 needs CUDA 13)
```

```python
import numpy as np
from flowpilot import FlowPilotNavigator, GpsGoal, Route, RouteGoal, compass_to_yaw

nav = FlowPilotNavigator(variant="flowpilot-dst-small", device="cuda")   # downloads the graph

obs = np.random.rand(1, nav.context_size, 3, 216, 384).astype(np.float32)   # (B, T, 3, H, W) in [0, 1]
ego_vw = [1.2, 0.0]     # measured speed (m/s) and yaw rate (rad/s)

# Point goal: metres in the ego frame, x forward, y left
vw, plan = nav.inference_vw(obs, [8.0, 0.5], ego_vw)                    # pure pursuit, vw: (B, 2)
vw, plan = nav.inference_vw(obs, [8.0, 0.5], ego_vw, controller="pd")   # PD
traj, scores = nav.inference_trajectory(obs, [8.0, 0.5], ego_vw)        # (B, 6, 80, 5), (B, 6)

# GPS goal: a waypoint and the robot's fix, converted to a point goal
goal = GpsGoal(goal=(34.06901, -118.44512), robot=(34.06893, -118.44520), yaw=compass_to_yaw(45.0))
vw, plan = nav.inference_vw(obs, goal, ego_vw)

# Route goal: the route patch around the robot, plus the route point 20 m ahead as the point goal
route = Route([[0.0, 0.0], [25.0, 0.0], [25.0, 40.0]], crosswalks=None)   # world metres; Route.from_gps for lat / lon
vw, plan = nav.inference_vw(obs, RouteGoal(route, pose=(3.0, 0.2, 0.0)), ego_vw)

vw, plan = nav.step(obs[0, -1], [8.0, 0.5], ego_vw)   # streaming: one frame per call at 20 Hz
nav.reset()                                           # between episodes
```

- **Observations** are 20 frames at 20 Hz, oldest first, at any size. Shorter histories are padded with their oldest frame.
- **Plans** hold `[x, y, yaw, v, w]` every 0.05 s up to 4 s, with modes ranked best first.
- **Goals** are required, as the exports have no goal-free input. World poses are `(x, y, yaw)` with x east, y north and yaw counter-clockwise from east.
- **Ego status** should be measured odometry. Without it the previous command stands in, and a window of zeros reads as a robot at rest. A route goal with a pose per frame derives it from the poses.
- **Controllers** also run on their own: `make_controller("pure_pursuit").step(plan, ego_speed=1.2)` takes any `[x, y]` or `[x, y, yaw, v, w]` path. Limits, lookahead and gains are config fields, for example `max_v`, `max_steering_angle=None` for differential drive, `kp` and `kd`.

### Camera

> **Use the training camera.** Both DST exports were trained on clips1k from a single camera model; other optics, mounts or crops shift the plans. We highly recommend using this standard configuration.

| | |
| --- | --- |
| Model | Kannala-Brandt fisheye (OpenCV `fisheye`), forward-facing |
| Intrinsics at 1920 x 1080 | fx = fy = 803.0, cx = 960.0, cy = 540.0 |
| Intrinsics at 384 x 216 (model input) | fx = fy = 160.6, cx = 192.0, cy = 108.0 |
| Distortion k1..k4 | -0.027864114, 0.002959364, 0, 0 |
| Height above ground | 0.456 m |
| Frames | full 16:9 view (no crop), resized to 384 x 216, 20 Hz, RGB |

clips1k robots carry a small per-robot principal-point offset (`principal_point_delta` in each clip's `camera.json`), which training removes by shifting the frames; a camera centred at (cx, cy) above needs no shift.

### Recorded clips

```bash
python examples/run_clips.py mini_data/*/ [--variant flowpilot-dst-dune]   # every 20-frame window -> <clip>/<variant>.npz
python examples/render_clips.py mini_data/*/                                # one panel per model -> <clip>/flowpilot_viz.mp4
```

A clip holds `front_fe_384x216.mp4`, the `frame_*.npy` arrays at 20 Hz and `route_full.json`. The `.npz` holds the commands, plans, goals, route patches and ground truth. The video draws each model's 4 s plan as a cyan ribbon on the ground, projected through the camera above, with the route patch the model saw as an inset.

- **Goal distance.** The point-goal encoder clips distance at 20 m, so a goal 20 m or farther reads as "keep driving"; a nearer goal tells the model it is arriving, and it slows down. The route goal therefore sits 20 m ahead along the route (`goal_ahead_m`); its straight-line distance dips a little on curves (17 m on these clips) and drops only near the route's end.
- **Route.** `route_full.json` is a polyline of `[lon, lat]` points (plain `coordinates` or a GeoJSON `geometry`). `Route.from_gps` takes it as `[lat, lon]` in a local east-north frame, and the robot's pose per frame is `frame_gps` (`[lat, lon]`) with `frame_yaw_deg` (counter-clockwise from east). From the poses the route renders the 40 x 40 m patch per frame and gives the point goal.
- **Ego status** is `frame_speeds` and the yaw rate `frame_angular_velocities[:, 2]`.
- **Ground truth** for scoring is the future of `frame_positions` / `frame_orientations` (wxyz) in the current ego frame.
- The principal-point shift is skipped here, since these clips carry no `camera.json`.

## Train and export

Run these from the `visnavkit` directory.

```bash
uv run visnavkit-train dataset=torch model=flowpilot                  # FlowPilot: FastViT-MA36, 64 anchors
uv run visnavkit-train experiment=flowpilot_dst_clips1k               # flowpilot-dst-small
uv run visnavkit-train experiment=flowpilot_dune_dst_clips1k \
  model.frame_encoder.weights=<DUNE ViT-B/14 ckpt>                    # flowpilot-dst-dune
uv run visnavkit-export-dst checkpoint=<ckpt> output=flowpilot_dst.onnx
```

## Release

The human-preference alignment stage will be released here.
