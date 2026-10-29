"""Run the goal-free FlowPilot (old recipe) on clips and draw its plan; writes <clip>/goal_free.npz and goal_free_viz.mp4.

    python examples/run_goal_free.py mini_data/*/

The model takes a PINHOLE camera at 512x288 (fx = fy = 210.667, cx = 256, cy = 144), so each fisheye frame is
undistorted to that camera first, and the plan is drawn through it. Input: 21 frames at 5 Hz (every 4th frame of
the 20 Hz video, 4 s of history), no goal. Output: 64 modes of 20 [x, y] waypoints at 5 Hz.
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from render_clips import CAM_FWD, CAM_H, CX, CY, FX, FY, K1, K2, draw_ribbon, label

W, H, F = 512, 288, 210.667  # the model's pinhole camera
STRIDE, CONTEXT, SCALE = 4, 21, 1.5  # 20 Hz -> 5 Hz; frames per window; drawn at 768x432
FISHEYE = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
PINHOLE = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
MAPS = cv2.fisheye.initUndistortRectifyMap(FISHEYE, np.array([K1, K2, 0, 0]), np.eye(3), PINHOLE, (W, H), cv2.CV_32FC1)


def project(pts):
    """Ground points [forward, left] (m) -> pixels of the upscaled pinhole view, and which are in front."""
    X, Y, Z = -pts[:, 1], np.full(len(pts), CAM_H), pts[:, 0] - CAM_FWD
    Zs = np.maximum(Z, 1e-6)
    return np.stack([F * X / Zs + W / 2, F * Y / Zs + H / 2], axis=-1) * SCALE, Z > 0.25


parser = argparse.ArgumentParser()
parser.add_argument("clips", nargs="+", type=Path)
args = parser.parse_args()
onnx = hf_hub_download("UCLA-VAIL/Navigation-Model-Zoo-Public", "FlowPilot_Official/flowpilot_base_goal_less.onnx")
session = ort.InferenceSession(onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

for clip in args.clips:
    cap, frames = cv2.VideoCapture(str(clip / "front_fe_384x216.mp4")), []
    while (ok := cap.read())[0]:
        frames.append(cv2.remap(ok[1], *MAPS, cv2.INTER_LINEAR))  # fisheye -> pinhole 512x288, BGR
    rgb = np.stack(frames)[..., ::-1].transpose(0, 3, 1, 2) / np.float32(255)

    first = STRIDE * (CONTEXT - 1)
    out = {"t": [], "plan": []}
    for t in range(first, len(frames)):
        modes, scores = session.run(None, {
            "observation": rgb[t - first : t + 1 : STRIDE][None],
            "embodiment_id": np.zeros((1, 1), np.int64),
            "metric_spacing": np.array([[0.0, 0.51, -0.16, 0.16]], np.float32),
        })
        modes, scores = modes[0, 1:], scores[0, 1:]  # mode 0 is a register token
        order = np.argsort(-scores)
        forward = order[modes[order, -1, 0] >= modes[order, 0, 0]]  # best-scored mode that moves forward
        out["t"].append(t)
        out["plan"].append(modes[forward[0] if len(forward) else order[0]])
    np.savez(clip / "goal_free.npz", **{k: np.stack(v) for k, v in out.items()})

    size = (int(W * SCALE), int(H * SCALE))
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", "20",
         "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(clip / "goal_free_viz.mp4")],
        stdin=subprocess.PIPE)
    for t, frame in enumerate(frames):
        img = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
        if t >= first:
            draw_ribbon(img, out["plan"][t - first], project)
        label(img, "goal-free, pinhole 512x288, 21 frames @ 5 Hz")
        ffmpeg.stdin.write(img.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(clip / "goal_free_viz.mp4")
