#!/usr/bin/env python3
"""Generate evaluation report with charts and tables.

Usage:
    python eval/generate_report.py --results-dir outputs/results/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(results_dir):
    """Load all result JSON files."""
    results_dir = Path(results_dir)
    standard = []
    pro = []

    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        if "perturbations" in data:
            pro.append(data)
        else:
            standard.append(data)

    return standard, pro


def plot_standard_results(standard_results, output_dir):
    """Plot per-task success rates for standard LIBERO evaluation."""
    if not standard_results:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in standard_results:
        suite = result["suite"]
        tasks = result["tasks"]

        fig, ax = plt.subplots(figsize=(12, 5))

        task_ids = [t["task_id"] for t in tasks]
        success_rates = [t["success_rate"] for t in tasks]
        languages = [t["language"][:30] for t in tasks]

        colors = ["#2ecc71" if sr > 0.7 else "#f39c12" if sr > 0.3 else "#e74c3c"
                  for sr in success_rates]

        bars = ax.bar(task_ids, success_rates, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Task ID")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Standard LIBERO — {suite}\nOverall: {result['overall_success_rate']:.1%}")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(task_ids)
        ax.axhline(y=result["overall_success_rate"], color="navy", linestyle="--",
                   alpha=0.7, label=f"Mean: {result['overall_success_rate']:.1%}")
        ax.legend()

        # Add task labels
        for bar, lang in zip(bars, languages):
            ax.text(bar.get_x() + bar.get_width() / 2, -0.08, lang,
                   ha="center", va="top", fontsize=6, rotation=45)

        plt.tight_layout()
        fig.savefig(output_dir / f"{suite}_standard.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir / f'{suite}_standard.png'}")


def plot_pro_results(pro_results, output_dir):
    """Plot LIBERO-PRO perturbation breakdown."""
    if not pro_results:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in pro_results:
        suite = result["suite"]
        perturbations = result["perturbations"]

        fig, ax = plt.subplots(figsize=(8, 5))

        ptypes = list(perturbations.keys())
        rates = [perturbations[p]["overall_success_rate"] for p in ptypes]

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
        bars = ax.bar(ptypes, rates, color=colors[:len(ptypes)], edgecolor="white")

        ax.set_ylabel("Success Rate")
        ax.set_title(f"LIBERO-PRO — {suite}\nPerturbation Breakdown")
        ax.set_ylim(0, max(max(rates) * 1.3, 0.1) if rates else 0.1)

        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                   f"{rate:.1%}", ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        fig.savefig(output_dir / f"{suite}_pro.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir / f'{suite}_pro.png'}")


def plot_comparison(standard_results, pro_results, output_dir):
    """Plot standard vs PRO comparison."""
    if not standard_results or not pro_results:
        return

    output_dir = Path(output_dir)

    # Match by suite
    std_by_suite = {r["suite"]: r["overall_success_rate"] for r in standard_results}
    pro_by_suite = {}
    for r in pro_results:
        suite = r["suite"]
        if "combined" in r["perturbations"]:
            pro_by_suite[suite] = r["perturbations"]["combined"]["overall_success_rate"]
        else:
            # Average across perturbation types
            rates = [p["overall_success_rate"] for p in r["perturbations"].values()]
            pro_by_suite[suite] = np.mean(rates) if rates else 0

    suites = sorted(set(std_by_suite) & set(pro_by_suite))
    if not suites:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(suites))
    width = 0.35

    std_rates = [std_by_suite[s] for s in suites]
    pro_rates = [pro_by_suite[s] for s in suites]

    ax.bar(x - width/2, std_rates, width, label="Standard LIBERO", color="#3498db")
    ax.bar(x + width/2, pro_rates, width, label="LIBERO-PRO", color="#e74c3c")

    ax.set_xticks(x)
    ax.set_xticklabels(suites, rotation=15)
    ax.set_ylabel("Success Rate")
    ax.set_title("Standard LIBERO vs LIBERO-PRO")
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "standard_vs_pro.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'standard_vs_pro.png'}")


def generate_text_report(standard_results, pro_results, output_dir):
    """Generate a text summary report."""
    output_dir = Path(output_dir)
    report_path = output_dir / "evaluation_report.md"

    lines = ["# Evaluation Report\n"]

    if standard_results:
        lines.append("## Standard LIBERO Results\n")
        lines.append("| Suite | Success Rate | Trials |")
        lines.append("|---|---|---|")
        for r in standard_results:
            lines.append(f"| {r['suite']} | {r['overall_success_rate']:.1%} | {r['total_trials']} |")
        lines.append("")

    if pro_results:
        lines.append("## LIBERO-PRO Results\n")
        for r in pro_results:
            lines.append(f"### {r['suite']}\n")
            lines.append("| Perturbation | Success Rate | Trials |")
            lines.append("|---|---|---|")
            for ptype, pres in r["perturbations"].items():
                lines.append(f"| {ptype} | {pres['overall_success_rate']:.1%} | {pres['total_trials']} |")
            lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--output-dir", type=str, default="outputs/results")
    args = parser.parse_args()

    standard, pro = load_results(args.results_dir)
    print(f"Found {len(standard)} standard results, {len(pro)} PRO results")

    plot_standard_results(standard, args.output_dir)
    plot_pro_results(pro, args.output_dir)
    plot_comparison(standard, pro, args.output_dir)
    generate_text_report(standard, pro, args.output_dir)


if __name__ == "__main__":
    main()
