"""Draw each model's 4 s plan on a clip's video, one panel per model; writes <clip>/flowpilot_viz.mp4.

    python examples/render_clips.py mini_data/*/ [--variants flowpilot-dst-small flowpilot-dst-dune]

The look follows cocobench-gs scripts/compose_r2g_views.py: a cyan ribbon of the robot's width on the ground,
solid by the robot and fading out at the end of the plan, and the route patch the model saw as an inset.
The camera is the training camera (README, "Camera"); the panels are upscaled 2x.
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

FX, FY, CX, CY, K1, K2 = 160.6, 160.6, 192.0, 108.0, -0.027864114, 0.002959364  # fleet fisheye @ 384x216
CAM_H, CAM_FWD, HALF_WIDTH = 0.456, 0.38, 0.28  # camera height and forward offset, robot half width (m)
NEAR, FAR, RIM = (np.array(c, np.float32) for c in ((255, 240, 0), (255, 190, 60), (255, 255, 215)))  # BGR
SCALE, FADE_POWER, ALPHA = 2, 2.0, 0.6


def project(pts):
    """Ground points [forward, left] (m) -> fisheye pixels of the upscaled panel, and which are in front."""
    X, Y, Z = -pts[:, 1], np.full(len(pts), CAM_H), pts[:, 0] - CAM_FWD
    r = np.hypot(X, Y)
    th = np.arctan2(r, np.maximum(Z, 1e-6))
    s = th * (1 + K1 * th**2 + K2 * th**4) / np.maximum(r, 1e-9)
    return np.stack([FX * s * X + CX, FY * s * Y + CY], axis=-1) * SCALE, Z > 0.25


def draw_ribbon(img, plan_xy, glow=0.18):
    pts = np.concatenate([[[0.0, 0.0]], plan_xy], axis=0).astype(np.float64)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if arc[-1] < 0.35:  # standing still: no path
        return
    keep = np.concatenate([[True], np.diff(arc) > 1e-4])
    s = np.arange(0.0, arc[-1], 0.08)
    path = np.stack([np.interp(s, arc[keep], pts[keep, i]) for i in range(2)], axis=-1)
    if len(path) < 3:
        return
    k = 7  # 0.6 m moving average: no kinks from per-step jitter
    pad = np.pad(path, ((k // 2, k // 2), (0, 0)), mode="edge")
    path = np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="valid") for i in range(2)], axis=-1)
    d = np.gradient(path, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=-1) / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    (luv, lok), (ruv, rok) = project(path + HALF_WIDTH * nrm), project(path - HALF_WIDTH * nrm)
    ok = lok & rok & np.all(np.abs(luv) < 2e4, axis=1) & np.all(np.abs(ruv) < 2e4, axis=1)
    h = img.shape[0]
    a_map, rim = np.zeros(img.shape[:2], np.float32), np.zeros(img.shape[:2], np.float32)
    c_map = np.zeros(img.shape, np.float32)
    frac = s / arc[-1]
    for j in range(len(path) - 1):
        if not (ok[j] and ok[j + 1]):
            continue
        quad = np.round([luv[j], luv[j + 1], ruv[j + 1], ruv[j]]).astype(np.int32)
        fade = float(1.0 - frac[j] ** FADE_POWER)
        cv2.fillConvexPoly(a_map, quad, fade)
        cv2.fillConvexPoly(c_map, quad, tuple(map(float, NEAR * (1 - frac[j]) + FAR * frac[j])))
        for a, b in ((quad[0], quad[1]), (quad[3], quad[2])):
            cv2.line(rim, tuple(map(int, a)), tuple(map(int, b)), fade, max(1, round(h / 300)), cv2.LINE_AA)
    if not a_map.any():
        return
    soft = max(0.8, h / 300)
    cover = cv2.GaussianBlur((a_map > 0).astype(np.float32), (0, 0), soft)
    color = cv2.GaussianBlur(c_map, (0, 0), soft) / np.maximum(cover, 1e-3)[..., None]
    a = ALPHA * cv2.GaussianBlur(a_map, (0, 0), soft)[..., None]
    rim = 0.55 * cv2.GaussianBlur(rim, (0, 0), 0.7 * max(h / 432, 1.0))[..., None]
    out = img.astype(np.float32) + glow * cv2.GaussianBlur(a_map, (0, 0), h / 45)[..., None] * NEAR
    out = out * (1 - a) + np.clip(color, 0, 255) * a
    img[:] = np.clip(out * (1 - rim) + RIM * rim, 0, 255).astype(np.uint8)


def draw_inset(img, patch, size=120):
    """The route patch as fed to the model: sidewalk route blue, crosswalk green, robot at the centre, up = forward."""
    rgb = np.zeros((80, 80, 3), np.uint8)
    rgb[patch == 1], rgb[patch == 2] = (255, 64, 64), (64, 255, 64)
    x0, y0 = img.shape[1] - size - 12, 12
    img[y0 : y0 + size, x0 : x0 + size] = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + size, y0 + size), (230, 230, 230), 1)


def label(img, text):
    for color, width in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, width, cv2.LINE_AA)


parser = argparse.ArgumentParser()
parser.add_argument("clips", nargs="+", type=Path)
parser.add_argument("--variants", nargs="+", default=["flowpilot-dst-small", "flowpilot-dst-dune"])
args = parser.parse_args()

for clip in args.clips:
    runs = [np.load(clip / f"{v}.npz") for v in args.variants]
    cap = cv2.VideoCapture(str(clip / "front_fe_384x216.mp4"))
    size = (384 * SCALE * len(runs), 216 * SCALE)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", "20",
         "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(clip / "flowpilot_viz.mp4")],
        stdin=subprocess.PIPE)
    t = 0
    while (ok := cap.read())[0]:
        panels = []
        for variant, run in zip(args.variants, runs):
            img = cv2.resize(ok[1], (size[0] // len(runs), size[1]), interpolation=cv2.INTER_CUBIC)
            i = np.searchsorted(run["t"], t)
            if i < len(run["t"]) and run["t"][i] == t:
                draw_ribbon(img, run["plan"][i, :, :2])
                draw_inset(img, run["patch"][i])
            label(img, variant.removeprefix("flowpilot-dst-"))
            panels.append(img)
        ffmpeg.stdin.write(np.hstack(panels).tobytes())
        t += 1
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(clip / "flowpilot_viz.mp4")
