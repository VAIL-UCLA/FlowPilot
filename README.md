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
