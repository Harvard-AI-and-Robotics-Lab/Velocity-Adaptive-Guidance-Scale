r"""
Plot avg_lambda_i vs step_index from a CSV file.

X-axis: step index
Y-axis: avg_lambda_i

Usage:
    python analysis/plot_adaptive_cfg_step.py proposed.csv --label 'Proposed ($\kappa=1$)' [--out plot.png]
    python analysis/plot_adaptive_cfg_step.py data4paper/step_lambda_i_k1.csv --label 'Proposed ($\kappa=1$)' --out data4paper/adaptive_cfg_step.svg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLS = ["step_index", "avg_lambda_i"]

# Static CFG value used by the baseline.
STATIC_CFG_BASELINE = 7.0

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


def load_and_validate(path: str) -> pd.DataFrame:
    """Load CSV and validate required columns."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def plot_one(ax, df: pd.DataFrame, label: str, color: str,
             ylim: tuple[float, float] | None = None) -> None:
    x = df["step_index"].values+1
    y = df["avg_lambda_i"].values

    ax.plot(x, y, color=color, linewidth=2.4, label="Avgerage Adaptive CFG across Samples")

    # Horizontal line at the mean of the adaptive CFG values.
    mean_val = float(y.mean())
    ax.axhline(
        mean_val,
        color=color,
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
        label=f"Mean Average Adaptive CFG ({mean_val:.4f})",
    )

    # Horizontal line at the static CFG used by the baseline.
    ax.axhline(
        STATIC_CFG_BASELINE,
        color="#1f77b4",
        linestyle="--",
        linewidth=2.0,
        alpha=0.9,
        label=f"Static CFG for Baseline ({STATIC_CFG_BASELINE:g})",
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Adaptive CFG")
    ax.grid(True, alpha=0.3)
    # ax.legend(loc="lower right", framealpha=0.9)
    ax.legend(loc="upper left", framealpha=0.9)


def plot_curves(df: pd.DataFrame, label: str, out_path: Path | None,
                ylim: tuple[float, float] | None = None) -> None:
    # Proposed = warm orange/red.
    color = "#d62728"

    fig, ax = plt.subplots(1, 1, figsize=(8, 9))

    plot_one(ax, df, label, color, ylim=ylim)

    # Match x ticks to actual step indices.
    n_steps = len(df)
    tick_step = max(1, n_steps // 12)
    ax.set_xticks(range(0, n_steps, tick_step))

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

    df = load_and_validate(args.csv_file)

    print(f"{args.label} ({args.csv_file}): {len(df)} steps, "
          f"avg range [{df['avg_lambda_i'].min():.4f}, {df['avg_lambda_i'].max():.4f}], "
          f"mean {df['avg_lambda_i'].mean():.4f}")

    plot_curves(df, args.label, args.out, ylim=args.ylim)


if __name__ == "__main__":
    main()