"""
Plot cumulative curves of avg_cos_sim from two CSV files in side-by-side subplots,
with shaded bands showing the [min_cos_sim, max_cos_sim] error bounds.

Left subplot:  Baseline (blue)
Right subplot: Proposed (orange)

X-axis: cumulative count of prompts (sorted by avg_cos_sim ascending)
Y-axis: avg_cos_sim, with min/max as the shaded band

Usage:
    python analysis/plot_cos_sim_prompt.py baseline.csv proposed.csv --labels 'Baseline' 'Proposed ($\alpha=1$)' [--out plot.png]
    python analysis/plot_cos_sim_prompt.py data4paper/prompt_cos_sim_k0.csv data4paper/prompt_cos_sim_k1.csv --labels 'Baseline' 'Proposed ($\alpha=1$)' --out data4paper/cos_sim_prompt.svg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLS = ["min_cos_sim", "max_cos_sim", "avg_cos_sim", "std_cos_sim"]

# Larger fonts across the board for clarity.
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
})


def load_and_sort(path: str) -> pd.DataFrame:
    """Load CSV, validate columns, and sort by avg_cos_sim ascending."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.sort_values("avg_cos_sim", ascending=True).reset_index(drop=True)
    df["cum_count"] = np.arange(1, len(df) + 1)
    return df


def plot_one(ax, df: pd.DataFrame, label: str, color: str) -> None:
    x = df["cum_count"].values
    y = df["avg_cos_sim"].values
    y_lo = df["min_cos_sim"].values
    y_hi = df["max_cos_sim"].values

    ax.fill_between(x, y_lo, y_hi, alpha=0.25, color=color,
                    label="min–max range")
    ax.plot(x, y, color=color, linewidth=2.4, label="avg")

    ax.set_title(label)
    ax.set_xlabel("Number of prompts (cumulative)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)


def plot_curves(dfs: list[pd.DataFrame], labels: list[str], out_path: Path | None) -> None:
    # Baseline = cool blue, Proposed = warm orange — colorblind-friendlier than red/blue.
    colors = ["#1f77b4", "#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)

    for ax, df, label, color in zip(axes, dfs, labels, colors):
        plot_one(ax, df, label, color)

    axes[0].set_ylabel("Cosine similarity")
    # fig.suptitle("Cumulative distribution of prompt cosine similarity", y=1.02)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", nargs=2,
                        help="Two CSV files: baseline first, proposed second")
    parser.add_argument("--labels", nargs=2, default=["Baseline", "Proposed"],
                        help="Labels for the two subplots (default: Baseline, Proposed)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output image path. If omitted, the plot is shown interactively.")
    args = parser.parse_args()

    dfs = [load_and_sort(p) for p in args.csv_files]

    for path, df, label in zip(args.csv_files, dfs, args.labels):
        print(f"{label} ({path}): {len(df)} prompts, "
              f"avg range [{df['avg_cos_sim'].min():.4f}, {df['avg_cos_sim'].max():.4f}]")

    plot_curves(dfs, args.labels, args.out)


if __name__ == "__main__":
    main()