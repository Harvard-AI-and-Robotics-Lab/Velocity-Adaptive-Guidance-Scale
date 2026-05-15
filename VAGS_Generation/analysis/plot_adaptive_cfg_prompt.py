"""
Plot a violin distribution of avg_lambda_i from a CSV file.

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


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV and validate that avg_lambda_i is present."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def plot_violin(ax, df: pd.DataFrame, label: str, color: str,
                ylim: tuple[float, float] | None = None) -> None:
    values = df["avg_lambda_i"].values

    parts = ax.violinplot(
        values,
        positions=[0],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )

    # Style the violin body to match the requested color scheme.
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.45)
        body.set_linewidth(1.5)

    q1, median, q3 = np.percentile(values, [25, 50, 75])
    vmin, vmax = float(np.min(values)), float(np.max(values))
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    # IQR bar and min/max whisker.
    ax.vlines(0, q1, q3, color=color, linewidth=8, alpha=0.9, zorder=3)
    ax.vlines(0, vmin, vmax, color=color, linewidth=1.8, alpha=0.9, zorder=2)

    # Median tick and mean diamond.
    ax.scatter([0], [median], marker="_", color="white", s=260, linewidth=3,
               zorder=4, label=f"median = {median:.4f}")
    ax.scatter([0], [mean], marker="D", color="white", edgecolor=color,
               s=70, linewidth=1.5, zorder=4, label=f"mean = {mean:.4f}")
    # Invisible handle so std appears as a legend entry without an extra glyph.
    ax.plot([], [], ' ', label=f"std = {std:.4f}")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(label)
    ax.set_xticks([0])
    ax.set_xticklabels([f"n = {len(values)}"])
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylabel("Adaptive CFG")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)


def plot_curves(df: pd.DataFrame, label: str, out_path: Path | None,
                ylim: tuple[float, float] | None = None) -> None:
    # Proposed = warm red, same palette as the original script.
    color = "#d62728"

    fig, ax = plt.subplots(1, 1, figsize=(8, 9))

    plot_violin(ax, df, label, color, ylim=ylim)

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
                        help="Explicit y-axis limits for the violin (default: matplotlib auto).")
    args = parser.parse_args()

    df = load_csv(args.csv_file)

    print(f"{args.label} ({args.csv_file}): {len(df)} prompts, "
          f"avg range [{df['avg_lambda_i'].min():.4f}, {df['avg_lambda_i'].max():.4f}]")

    plot_curves(df, args.label, args.out, ylim=args.ylim)


if __name__ == "__main__":
    main()