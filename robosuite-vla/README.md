# SmolVLA × LIBERO: Robustness-Aware Robot Manipulation

Fine-tune [SmolVLA](https://arxiv.org/abs/2506.01844) (450M params) on [LIBERO](https://arxiv.org/abs/2306.03310) human demonstrations and evaluate generalization using [LIBERO-PRO](https://arxiv.org/abs/2510.03827) perturbation benchmarks.

## Motivation

State-of-the-art VLAs achieve >90% on standard LIBERO — but score **0%** when objects are moved, recolored, or instructions are paraphrased (LIBERO-PRO). They memorize trajectories, not understand tasks. This project aims to train models that show genuine understanding.

## Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/YOUR_USER/robosuite-vla.git
cd robosuite-vla

# Install
pip install -e ".[dev]"

# Or for Colab — see notebooks/01_setup_and_data.ipynb
```

## Phases

### Phase 1: Data
```bash
python data/download_libero.py --suite all
python data/inspect_demos.py --suite libero_spatial --task 0
python data/convert_to_lerobot.py --suite libero_spatial --repo-id USER/libero_spatial
```

### Phase 2: Train
```bash
python train/train_smolvla.py --config config/libero_spatial.yaml
```

### Phase 3: Evaluate
```bash
python eval/eval_libero.py --checkpoint outputs/checkpoints/best --suite libero_spatial
python eval/eval_libero_pro.py --checkpoint outputs/checkpoints/best --suite libero_spatial
```

### Phase 4: Report
```bash
python eval/generate_report.py --results-dir outputs/results/
```

## Project Structure

```
robosuite-vla/
├── config/          # Training configs per LIBERO suite
├── data/            # Download, inspect, convert demo data
├── train/           # SmolVLA fine-tuning scripts
├── eval/            # Standard + LIBERO-PRO evaluation
├── viz/             # Rollout recording, plots, attention viz
├── notebooks/       # Google Colab notebooks
└── outputs/         # Checkpoints, videos, results
```

## LIBERO Task Suites

| Suite | Tasks | Tests |
|---|---|---|
| LIBERO-Spatial | 10 | Spatial reasoning (object arrangements) |
| LIBERO-Object | 10 | Object recognition (different objects) |
| LIBERO-Goal | 10 | Procedural understanding (different goals) |
| LIBERO-10 | 10 | Mixed complexity, longer horizons |
| LIBERO-90 | 90 | Large-scale multi-task |

## LIBERO-PRO Perturbation Dimensions

| Perturbation | What Changes |
|---|---|
| Object | Appearance, color, scale of target objects |
| Position | Object locations within the scene |
| Instruction | Paraphrased task descriptions |
| Environment | Backgrounds, lighting, scene layout |

## Expected Results

| Model | Standard LIBERO | LIBERO-PRO (position) | LIBERO-PRO (combined) |
|---|---|---|---|
| OpenVLA (7B) | >90% | ~0% | 0% |
| SmolVLA (450M, ours) | TBD | TBD | TBD |

Any non-zero LIBERO-PRO score is meaningful progress.

## References

- [LIBERO](https://arxiv.org/abs/2306.03310) — Benchmark
- [LIBERO-PRO](https://arxiv.org/abs/2510.03827) — Perturbation evaluation
- [SmolVLA](https://arxiv.org/abs/2506.01844) — Our primary model
- [OpenVLA](https://arxiv.org/abs/2406.09246) — Comparison baseline
- [LeRobot](https://github.com/huggingface/lerobot) — Training framework
