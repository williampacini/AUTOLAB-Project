#!/usr/bin/env python3
"""Compare evaluation results across multiple models.

Usage:
    python eval/eval_compare.py --results-dir outputs/results/
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_all_results(results_dir):
    """Load all JSON result files from a directory."""
    results_dir = Path(results_dir)
    results = []

    for json_file in sorted(results_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
            data["_file"] = json_file.name
            results.append(data)

    return results


def build_comparison_table(results):
    """Build a comparison table from results."""
    rows = []

    for r in results:
        model_name = Path(r.get("checkpoint", "unknown")).name
        suite = r.get("suite", "unknown")
        seed = r.get("seed", "?")

        if "perturbations" in r:
            # LIBERO-PRO results
            for ptype, pres in r["perturbations"].items():
                rows.append({
                    "model": model_name,
                    "suite": suite,
                    "eval_type": f"PRO-{ptype}",
                    "success_rate": pres["overall_success_rate"],
                    "trials": pres.get("total_trials", 0),
                    "seed": seed,
                })
        else:
            # Standard LIBERO results
            rows.append({
                "model": model_name,
                "suite": suite,
                "eval_type": "standard",
                "success_rate": r.get("overall_success_rate", 0),
                "trials": r.get("total_trials", 0),
                "seed": seed,
            })

    return pd.DataFrame(rows)


def print_comparison(df):
    """Print a formatted comparison table."""
    if df.empty:
        print("No results found.")
        return

    # Pivot: models as rows, eval types as columns
    pivot = df.pivot_table(
        index=["model", "suite"],
        columns="eval_type",
        values="success_rate",
        aggfunc="mean",
    )

    print("\n" + "=" * 80)
    print("  Model Comparison — Success Rates")
    print("=" * 80)

    # Format as percentages
    formatted = pivot.map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    print(formatted.to_string())
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare model results")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--output", type=str, default=None, help="Save comparison CSV")
    args = parser.parse_args()

    results = load_all_results(args.results_dir)
    if not results:
        print(f"No result files found in {args.results_dir}")
        return

    df = build_comparison_table(results)
    print_comparison(df)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
