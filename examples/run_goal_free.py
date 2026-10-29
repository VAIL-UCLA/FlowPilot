"""Run the goal-free FlowPilot (old recipe) on clips and draw its plan; writes <clip>/goal_free.npz and goal_free_viz.mp4.

    python examples/run_goal_free.py mini_data/*/   # clips with front_fe_384x216.mp4 (or front.mp4, fleet fisheye at any size)

As the Navigation Model Zoo's open-loop test does: each fisheye frame is undistorted to the model's PINHOLE camera
(512x288, fx = fy = 210.667, cx = 256, cy = 144; the fisheye is the same focal length at the video's size, with the
fleet k1, k2), 21 of those frames at 5 Hz (every 4th of the 20 Hz video, 4 s of history) go in, no goal. Out: 64
modes of 20 [x, y] waypoints at 5 Hz; the top-scored mode is drawn through the pinhole camera.
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from render_clips import CAM_H, K1, K2, draw_ribbon, label

W, H, F = 512, 288, 210.667  # the model's pinhole camera
STRIDE, CONTEXT, SCALE = 4, 21, 1.5  # 20 Hz -> 5 Hz; frames per window; drawn at 768x432


def undistort_maps(src_wh, dst_wh):
    """Fleet fisheye at ``src_wh`` -> the model's pinhole camera at ``dst_wh`` (both scaled from 512x288)."""
    K = lambda w, h: np.array([[F * w / W, 0, w / 2], [0, F * h / H, h / 2], [0, 0, 1]])
    return cv2.fisheye.initUndistortRectifyMap(K(*src_wh), np.array([K1, K2, 0, 0]), np.eye(3), K(*dst_wh), dst_wh, cv2.CV_32FC1)


def project(pts):
    """Ground points [forward, left] (m) -> pixels of the upscaled pinhole view, and which are in front."""
    X, Y, Z = -pts[:, 1], np.full(len(pts), CAM_H), pts[:, 0]
    Zs = np.maximum(Z, 1e-6)
    return np.stack([F * X / Zs + W / 2, F * Y / Zs + H / 2], axis=-1) * SCALE, Z > 0.25


parser = argparse.ArgumentParser()
parser.add_argument("clips", nargs="+", type=Path)
args = parser.parse_args()
onnx = hf_hub_download("UCLA-VAIL/Navigation-Model-Zoo-Public", "FlowPilot_Official/flowpilot_base_goal_less.onnx")
session = ort.InferenceSession(onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
size = (int(W * SCALE), int(H * SCALE))

for clip in args.clips:
    video = clip / "front.mp4" if (clip / "front.mp4").exists() else clip / "front_fe_384x216.mp4"
    cap = cv2.VideoCapture(str(video))
    src = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    to_model, to_view = undistort_maps(src, (W, H)), undistort_maps(src, size)
    model_in, views = [], []
    while (ok := cap.read())[0]:
        model_in.append(cv2.remap(ok[1], *to_model, cv2.INTER_LINEAR))
        views.append(cv2.remap(ok[1], *to_view, cv2.INTER_LINEAR))
    rgb = np.stack(model_in)[..., ::-1].transpose(0, 3, 1, 2) / np.float32(255)

    first = STRIDE * (CONTEXT - 1)
    out = {"t": [], "plan": []}
    for t in range(first, len(rgb)):
        modes, scores = session.run(None, {
            "observation": rgb[t - first : t + 1 : STRIDE][None],
            "embodiment_id": np.zeros((1, 1), np.int64),
            "metric_spacing": np.array([[0.0, 0.51, -0.16, 0.16]], np.float32),
        })
        modes, scores = modes[0, 1:], scores[0, 1:]  # mode 0 is a register token
        out["t"].append(t)
        out["plan"].append(modes[scores.argmax()])
    np.savez(clip / "goal_free.npz", **{k: np.stack(v) for k, v in out.items()})

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", "20",
         "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(clip / "goal_free_viz.mp4")],
        stdin=subprocess.PIPE)
    for t, img in enumerate(views):
        if t >= first:
            draw_ribbon(img, out["plan"][t - first], project)
        label(img, "goal-free, pinhole 512x288, 21 frames @ 5 Hz")
        ffmpeg.stdin.write(img.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(clip / "goal_free_viz.mp4")
