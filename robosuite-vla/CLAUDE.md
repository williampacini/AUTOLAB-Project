# SmolVLA × LIBERO — Vision-Language-Action Robotics Project

## Project Goal

Train SmolVLA (450M params) on LIBERO human demonstrations and evaluate on both
standard LIBERO (task success) **and** LIBERO-PRO (perturbation robustness) to test
whether the model genuinely understands tasks vs. memorizing action sequences.

## Why This Matters

Current state-of-the-art VLAs score >90% on standard LIBERO but **collapse to 0%**
under LIBERO-PRO perturbations. Even random text instructions don't change model
outputs — proving models memorize trajectories, not understand tasks. Any model that
achieves meaningfully non-zero LIBERO-PRO scores demonstrates genuine progress.

## Architecture

```
robosuite-vla/
├── config/          # Hydra-style YAML configs per suite
├── data/            # Download, inspect, convert LIBERO demos
├── train/           # Fine-tune SmolVLA via LeRobot
├── eval/            # Standard LIBERO + LIBERO-PRO evaluation
├── viz/             # Rollout recording, plots, attention maps
├── notebooks/       # Colab notebooks for each phase
└── outputs/         # Checkpoints, videos, result CSVs
```

## Quick Commands

```bash
# Phase 1: Download LIBERO data
python data/download_libero.py --suite libero_spatial

# Inspect demos (generates GIF previews)
python data/inspect_demos.py --suite libero_spatial --task 0 --demo 0

# Convert to LeRobot format
python data/convert_to_lerobot.py --suite libero_spatial --repo-id USER/libero_spatial

# Phase 2: Train SmolVLA
python train/train_smolvla.py --config config/libero_spatial.yaml

# Phase 3: Evaluate standard LIBERO
python eval/eval_libero.py --checkpoint outputs/checkpoints/smolvla_spatial --suite libero_spatial

# Phase 3b: Evaluate LIBERO-PRO
python eval/eval_libero_pro.py --checkpoint outputs/checkpoints/smolvla_spatial --suite libero_spatial

# Phase 4: Compare models
python eval/eval_compare.py --results-dir outputs/results/
python eval/generate_report.py --results-dir outputs/results/
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
- **convert_to_lerobot.py info.json format**: LeRobot v0.4+ requires `features` dict in `info.json`. Old format used flat keys (`task_name`, `action_dim`, etc.) which causes `KeyError: 'features'`. Parquet must use array columns (`action`, `observation.state`) not scalar columns (`action_dx`, `action_dy`). Fixed in convert_to_lerobot.py.
- **LeRobot v3.0 metadata format**: `tasks.parquet` and `episodes.parquet` (NOT `.jsonl`). The installed LeRobot calls `pd.read_parquet()` for these files. Using `.jsonl` causes `FileNotFoundError`. Set `codebase_version: "v3.0"` in info.json.
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

## Data Sources
- **Primary**: LIBERO human teleoperation demos (50/task, SpaceMouse, 20Hz)
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
