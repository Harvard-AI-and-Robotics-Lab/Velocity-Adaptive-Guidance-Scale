"""
Plot cumulative curves of avg_cos_sim from two CSV files on the same axes,
with shaded bands showing the [min_cos_sim, max_cos_sim] error bounds.

X-axis: cumulative count of prompts (sorted by avg_cos_sim ascending)
Y-axis: avg_cos_sim, with min/max as the shaded band

Usage:
    python plot_cumulative_cos_sim.py file1.csv file2.csv [--labels A B] [--out plot.png]
    python plot_cumulative_cos_sim.py file1.csv file2.csv --labels "Baseline" "Proposed (k=1)" --out data4paper/cumulative_cos_sim_prompt.svg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLS = ["min_cos_sim", "max_cos_sim", "avg_cos_sim"]


def load_and_sort(path: str) -> pd.DataFrame:
    """Load CSV, validate columns, and sort by avg_cos_sim ascending.

    The min/max columns are reordered alongside avg so each row's band still
    corresponds to the same prompt after sorting.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.sort_values("avg_cos_sim", ascending=True).reset_index(drop=True)
    # Cumulative count: 1..N (so the first point represents "1 prompt has sim <= x")
    df["cum_count"] = np.arange(1, len(df) + 1)
    return df


def plot_curves(dfs: list[pd.DataFrame], labels: list[str], out_path: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Distinct colors that work well together
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for df, label, color in zip(dfs, labels, colors):
        x = df["cum_count"].values
        y = df["avg_cos_sim"].values
        y_lo = df["min_cos_sim"].values
        y_hi = df["max_cos_sim"].values

        ax.fill_between(x, y_lo, y_hi, alpha=0.20, color=color,
                        label=f"{label} (min–max range)")
        ax.plot(x, y, color=color, linewidth=1.8, label=f"{label} (avg)")

    ax.set_xlabel("Number of prompts (cumulative, sorted by avg cosine similarity)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cumulative distribution of prompt cosine similarity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", nargs=2, help="Two CSV files to plot")
    parser.add_argument("--labels", nargs=2, default=None,
                        help="Labels for the two curves (defaults to filenames)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output image path. If omitted, the plot is shown interactively.")
    args = parser.parse_args()

    labels = args.labels or [Path(p).stem for p in args.csv_files]
    dfs = [load_and_sort(p) for p in args.csv_files]

    for path, df in zip(args.csv_files, dfs):
        print(f"{path}: {len(df)} prompts, "
              f"avg range [{df['avg_cos_sim'].min():.4f}, {df['avg_cos_sim'].max():.4f}]")

    plot_curves(dfs, labels, args.out)


if __name__ == "__main__":
    main()