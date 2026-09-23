# FlowPilot
From Imitation to Alignment: Human-Preference Flow Policies for Long-Horizon Sidewalk Navigation

[![arXiv](https://img.shields.io/badge/arXiv-2606.12603-blue)](https://arxiv.org/abs/2606.12603)
[![CoRL](https://img.shields.io/badge/CoRL-2026-orange)](https://arxiv.org/abs/2606.12603)
[![VisNavKit](https://img.shields.io/badge/VisNavKit-training%20recipe-green)](https://github.com/DhlinV/visnavkit)

<p align="center">
  <img src="_assets/teaser.gif" alt="FlowPilot teaser" width="800">
</p>
<p align="center"><sub>GPS-guided long-horizon navigation · sidewalk lane keeping · obstacle avoidance · pedestrian awareness · night driving</sub></p>

<p align="center">
  <img src="_assets/sidewalkbench_gs_cars_crossing.gif" alt="FlowPilot on SidewalkBench-GS: crossing with parked cars" width="398">
  <img src="_assets/sidewalkbench_gs_yellow_car_crossing.gif" alt="FlowPilot on SidewalkBench-GS: crossing past a yellow car" width="398">
</p>
<p align="center"><sub>Closed-loop rollouts in SidewalkBench-GS · Gaussian-splat reconstructions of real sidewalks</sub></p>

The training architecture is available as the `model=flowpilot` recipe in **[VisNavKit](https://github.com/DhlinV/visnavkit)** — FastViT-MA36, a 4-layer scene transformer, 64 k-means anchors and the anchored rectified-flow decoder with Beta(1.5, 1) time sampling, composed from swappable Hydra groups, with ONNX export and an open-loop benchmark:

```bash
uv run visnavkit-train dataset=torch model=flowpilot
```

Code, checkpoints and the human-preference alignment stage will be released here.
