# AUTOLAB Simulation Notebooks

Google Colab notebooks for running autonomous robot arm simulations with **Robosuite** and **LIBERO**.

## Notebooks

| Notebook | Framework | What it does |
|---|---|---|
| [`robosuite_sim.ipynb`](robosuite_sim.ipynb) | Robosuite v1.5 | Lift, Stack, and PickPlace tasks with Panda arm |
| [`libero_sim.ipynb`](libero_sim.ipynb) | LIBERO (robosuite v1.4) | 130 benchmark tasks across 4 suites |

## Quick Start

1. Open a notebook in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
3. Run cells top-to-bottom — system setup, install, then experiments

## What's Inside

### Robosuite notebook
- Headless EGL rendering setup (no display needed)
- Lift, Stack, PickPlaceSingle environments
- Scripted reach-and-grasp policy example
- Multi-robot comparison (Panda, Sawyer, IIWA, Jaco)
- Multi-camera view capture
- Video recording and inline playback
- Observation space explorer

### LIBERO notebook
- Full LIBERO installation with compatible dependencies
- All 3 core suites: Spatial, Object, Goal
- Task listing with natural language descriptions
- Batch evaluation across an entire suite
- Observation space explorer
- Google Drive integration for saving results
- Training command reference for BC-RNN, Transformer, and ViLT policies

## Important Notes

- **Robosuite vs LIBERO versions:** LIBERO requires `robosuite==1.4.x`. The robosuite notebook uses v1.5. Do not run both in the same Colab session — use separate runtimes.
- **Free tier limits:** Colab free tier provides ~12 hours per session and limited GPU time. Save checkpoints to Google Drive frequently.
- **Rendering:** Both notebooks use EGL (GPU-accelerated headless rendering). If you use a CPU-only runtime, change `MUJOCO_GL` to `osmesa`.

## Environments Reference

### Robosuite Tasks
- `Lift` — Pick up a cube
- `Stack` — Stack one cube on another
- `PickPlaceSingle` — Pick and place one object
- `PickPlace` — Multi-object pick and place
- `NutAssembly` — Nut-and-peg assembly
- `Door` — Open a door
- `Wipe` — Wipe a surface

### LIBERO Suites
- `libero_spatial` — 10 tasks testing spatial transfer
- `libero_object` — 10 tasks testing object transfer
- `libero_goal` — 10 tasks testing goal transfer
- `libero_100` — 100 tasks (split: 90 pretrain + 10 eval)
