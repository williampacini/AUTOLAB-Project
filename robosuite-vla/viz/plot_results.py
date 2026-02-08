#!/usr/bin/env python3
"""Plot evaluation results — success rates, comparisons, training curves.

Usage:
    python viz/plot_results.py --results-dir outputs/results/ --output-dir outputs/results/plots/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_suite_comparison(results_dir, output_dir):
    """Plot success rates across LIBERO suites."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suites = {}
    for f in results_dir.glob("*_seed*.json"):
        if "_pro_" in f.name:
            continue
        with open(f) as fh:
            data = json.load(fh)
        suite = data["suite"]
        if suite not in suites:
            suites[suite] = []
        suites[suite].append(data["overall_success_rate"])

    if not suites:
        print("No standard results found.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    suite_names = sorted(suites.keys())
    means = [np.mean(suites[s]) for s in suite_names]
    stds = [np.std(suites[s]) for s in suite_names]

    bars = ax.bar(suite_names, means, yerr=stds, capsize=5,
                  color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"][:len(suite_names)],
                  edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Success Rate")
    ax.set_title("SmolVLA — Standard LIBERO Evaluation")
    ax.set_ylim(0, 1.05)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
               f"{m:.1%}", ha="center", fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "suite_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'suite_comparison.png'}")


def plot_training_curve(log_file, output_dir):
    """Plot training loss curve from W&B or log file."""
    # Placeholder — would parse wandb logs or training output
    print(f"Training curve plotting not yet implemented for {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--output-dir", type=str, default="outputs/results/plots")
    args = parser.parse_args()

    plot_suite_comparison(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
