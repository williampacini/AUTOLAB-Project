# SmolVLA × LIBERO / NutAssembly — Vision-Language-Action Robotics Project

## Project Goal

Train SmolVLA (450M params) on LIBERO human demonstrations and evaluate on both
standard LIBERO (task success) **and** LIBERO-PRO (perturbation robustness) to test
whether the model genuinely understands tasks vs. memorizing action sequences.

Additionally, post-train SmolVLA on **RobotSuite NutAssembly** (full: round + square
nuts on pegs) and evaluate with per-episode MP4 video output.

## Why This Matters

Current state-of-the-art VLAs score >90% on standard LIBERO but **collapse to 0%**
under LIBERO-PRO perturbations. Even random text instructions don't change model
outputs — proving models memorize trajectories, not understand tasks. Any model that
achieves meaningfully non-zero LIBERO-PRO scores demonstrates genuine progress.

## Architecture

```
robosuite-vla/
├── config/          # YAML configs (libero_spatial, nut_assembly, etc.)
├── data/            # Download, inspect, convert, collect demos
│   ├── collect_nut_assembly.py    # Scripted demo collection for NutAssembly
│   ├── download_nut_assembly.py   # Download robomimic HDF5 from HuggingFace
│   ├── build_nut_assembly_dataset.py  # Full data pipeline orchestrator
│   └── convert_to_lerobot.py      # HDF5 → LeRobot format (LIBERO + robomimic)
├── train/           # Fine-tune SmolVLA via LeRobot
├── eval/            # LIBERO + LIBERO-PRO + NutAssembly evaluation
│   └── eval_nut_assembly.py       # NutAssembly eval with MP4 video output
├── viz/             # Rollout recording, plots, attention maps
├── notebooks/       # Colab notebooks (01-03: LIBERO, 04: NutAssembly)
└── outputs/         # Checkpoints, videos, result CSVs
```

## Quick Commands

### LIBERO Pipeline
```bash
python data/download_libero.py --suite libero_spatial
python data/inspect_demos.py --suite libero_spatial --task 0 --demo 0
python data/convert_to_lerobot.py --suite libero_spatial --repo-id USER/libero_spatial
python train/train_smolvla.py --config config/libero_spatial.yaml
python eval/eval_libero.py --checkpoint outputs/checkpoints/smolvla_spatial --suite libero_spatial
python eval/eval_libero_pro.py --checkpoint outputs/checkpoints/smolvla_spatial --suite libero_spatial
python eval/eval_compare.py --results-dir outputs/results/
```

### NutAssembly Pipeline
```bash
# Full data pipeline: download HF + collect scripted + convert
python data/build_nut_assembly_dataset.py

# Or step by step:
python data/download_nut_assembly.py --output-dir data/robomimic_nut_assembly
python data/collect_nut_assembly.py --num-demos 50 --visualize
python data/build_nut_assembly_dataset.py --skip-download --skip-collect

# Train
python train/train_smolvla.py --config config/nut_assembly.yaml

# Evaluate with video output for every episode
python eval/eval_nut_assembly.py --checkpoint outputs/checkpoints/smolvla_nut_assembly --episodes 20 --speedup 3
```

## Environment Setup — Critical Notes

### Headless Rendering (MUST be first)
```python
import os
os.environ["MUJOCO_GL"] = "osmesa"   # or "egl" if GPU available
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
```

### Known Bugs
- **Camera images are flipped**: Always `np.flip(obs["agentview_image"], axis=0)`
- **RoboSuite 1.5 `env.name`**: Does not exist — store manually
- **numpy version**: Colab needs `numpy>=2.0,<2.1` (numba compat)
- **macros_private**: Create empty file in robosuite package dir to silence warning
- **PyTorch 2.7+ GPU memory**: Use `torch.cuda.get_device_properties(0).total_memory` (not `total_mem`)
- **LeRobot v0.4+ CLI**: Use `lerobot-train` command (not `python -m lerobot.scripts.train`)
- **LeRobot v0.4+ requires policy.repo_id**: Add `--policy.repo_id=username/model-name`
- **Dataset format**: Use `HuggingFaceVLA/libero` (public, v3.0). The `yifengzhu-hf/*` datasets are gated and v2.0 (incompatible)
- **F-string backslash in Colab**: Use string concatenation instead of `\` line continuation in f-strings
- **SmolVLA 3-camera mismatch**: `--policy.path=lerobot/smolvla_base` expects 3 cameras but LIBERO has 2. Use `--policy.type=smolvla --policy.load_vlm_weights=true` instead (auto-infers features from dataset)
- **SmolVLA batch size**: Official examples use batch_size=4 (not 64). SmolVLA is memory-heavy on A100
- **SmolVLA image keys**: Must use `observation.images.image` and `observation.images.image2` (not `observation.image`). The `.images.` prefix is required.
- **SmolVLA wrist camera**: Provide `robot0_eye_in_hand_image` as `observation.images.image2` during inference
- **SmolVLA policy.reset()**: Must call `policy.reset()` at the start of each episode to clear internal action queue
- **PyTorch 2.6+ torch.load**: LIBERO's `get_task_init_states()` uses `torch.load` which defaults to `weights_only=True` in PyTorch 2.6+. Monkey-patch with `weights_only=False` before importing LIBERO

### Action Space (Panda + OSC_POSE)
7-dim: `[dx, dy, dz, dax, day, daz, gripper]`
- Position deltas: [-1, 1], typical 0.05–0.3
- Gripper: -1 = open, +1 = close
- Control freq: 20Hz
- Always `np.clip(action, low, high)`

### LIBERO Demo Data Format (HDF5)
Each demo contains:
- `obs/agentview_image` — (T, 128, 128, 3)
- `obs/robot0_eye_in_hand_image` — (T, 128, 128, 3) (some configs)
- `obs/robot0_eef_pos` — (T, 3)
- `obs/robot0_eef_quat` — (T, 4)
- `obs/robot0_gripper_qpos` — (T, 2)
- `obs/robot0_joint_pos` — (T, 7)
- `actions` — (T, 7)

### SmolVLA Post-Training Fix
After training, change `n_action_steps` from 1 to 50 in model config.json
before inference. Without this, inference is 50× slower.

## NutAssembly-Specific Notes

### Observation Keys
- `SquareNut_pos` — (3,) world position of square nut
- `RoundNut_pos` — (3,) world position of round nut
- `SquareNut_to_robot0_eef_pos` — (3,) relative position
- `RoundNut_to_robot0_eef_pos` — (3,) relative position
- Peg positions: access via `env.sim.data.body_xpos[env.peg1_body_id]` (square target) and `env.peg2_body_id` (round target)
- Nut handle positions: `env.sim.data.site_xpos[env.sim.model.site_name2id(nut.important_sites["handle"])]`

### NutAssembly vs LIBERO
- **Horizon**: NutAssembly uses 1000 steps (LIBERO uses 600)
- **Success**: `env._check_success()` — both nuts on correct pegs (XY dist < 0.03)
- **Environment**: Direct `robosuite.make()` (no BDDL files)
- **Data**: 3/4 from robomimic HF (NutAssemblySquare + NutAssemblyRound), 1/4 scripted
- **Cameras**: agentview + robot0_eye_in_hand, both at 256x256

### Video Output
- `eval/eval_nut_assembly.py` records MP4 for EVERY test episode
- Speedup factor (default 3x): increases output FPS so 50s real-time plays in ~17s
- Filenames: `scenario_{idx:03d}_{success|fail}.mp4`
- HTML report generated at `outputs/results/nut_assembly_eval.html`

## Data Sources
- **LIBERO**: Human teleoperation demos (50/task, SpaceMouse, 20Hz)
- **NutAssembly**: robomimic HF (`amandlek/robomimic`) + scripted policy demos
- **Pretrained base**: SmolVLA pretrained on 10M frames from 487 LeRobot datasets
- **NOT using**: Claude-generated demos (failed — cannot do spatial reasoning from images)

## Key Papers
- LIBERO: https://arxiv.org/abs/2306.03310
- LIBERO-PRO: https://arxiv.org/abs/2510.03827
- SmolVLA: https://arxiv.org/abs/2506.01844
- OpenVLA: https://arxiv.org/abs/2406.09246

## Current Status
- [ ] Phase 1: LIBERO setup + data download
- [ ] Phase 2: SmolVLA fine-tuning
- [ ] Phase 3: Standard + PRO evaluation
- [ ] Phase 4: Baseline comparison + report
- [ ] NutAssembly: Post-train + evaluate with video
