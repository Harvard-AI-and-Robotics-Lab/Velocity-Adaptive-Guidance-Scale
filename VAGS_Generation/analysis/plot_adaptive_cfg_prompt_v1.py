"""
Plot a cumulative curve of avg_lambda_i from a CSV file,
with a shaded band showing the [min_lambda_i, max_lambda_i] error bounds.

X-axis: cumulative count of prompts (sorted by avg_lambda_i ascending)
Y-axis: avg_lambda_i, with min/max as the shaded band

Usage:
    python analysis/plot_adaptive_cfg_prompt.py proposed.csv --label 'Proposed ($\kappa=1$)' [--out plot.png]
    python analysis/plot_adaptive_cfg_prompt.py data4paper/prompt_lambda_i_k1.csv --label 'Proposed ($\kappa=1$)' --out data4paper/adaptive_cfg_prompt.svg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLS = ["avg_lambda_i"]

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
    """Load CSV, validate columns, and sort by avg_lambda_i ascending."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.sort_values("avg_lambda_i", ascending=True).reset_index(drop=True)
    df["cum_count"] = np.arange(1, len(df) + 1)
    return df


def plot_one(ax, df: pd.DataFrame, label: str, color: str,
             ylim: tuple[float, float] | None = None) -> None:
    x = df["cum_count"].values
    y = df["avg_lambda_i"].values

    ax.plot(x, y, color=color, linewidth=2.4, label="avg")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(label)
    ax.set_xlabel("Number of prompts (cumulative)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)


def plot_curves(df: pd.DataFrame, label: str, out_path: Path | None,
                ylim: tuple[float, float] | None = None) -> None:
    # Proposed = warm orange/red.
    color = "#d62728"

    fig, ax = plt.subplots(1, 1, figsize=(8, 9))

    plot_one(ax, df, label, color, ylim=ylim)

    ax.set_ylabel("Adaptive CFG")
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file",
                        help="CSV file for the proposed method")
    parser.add_argument("--label", default="Proposed",
                        help="Label for the subplot (default: Proposed)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output image path. If omitted, the plot is shown interactively.")
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Explicit y-axis limits (default: matplotlib auto).")
    args = parser.parse_args()

    df = load_and_sort(args.csv_file)

    print(f"{args.label} ({args.csv_file}): {len(df)} prompts, "
          f"avg range [{df['avg_lambda_i'].min():.4f}, {df['avg_lambda_i'].max():.4f}]")

    plot_curves(df, args.label, args.out, ylim=args.ylim)


if __name__ == "__main__":
    main()