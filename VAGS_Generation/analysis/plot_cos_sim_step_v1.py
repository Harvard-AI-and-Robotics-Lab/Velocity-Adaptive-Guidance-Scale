r"""
Plot avg_cos_sim vs step_index from two CSV files in three side-by-side subplots,
with shaded bands showing the [min_cos_sim, max_cos_sim] error bounds.

Left subplot:   Baseline (blue)
Middle subplot: Proposed (orange/red)
Right subplot:  std_cos_sim difference (proposed - baseline) vs step index

X-axis (left/middle): step index
Y-axis (left/middle): avg_cos_sim, with min/max as the shaded band
X-axis (right):       step index
Y-axis (right):       std_cos_sim(proposed) - std_cos_sim(baseline)

Usage:
    python analysis/plot_cos_sim_step.py baseline.csv proposed.csv --labels 'Baseline' 'Proposed ($\kappa=1$)' [--out plot.png]
    python analysis/plot_cos_sim_step.py data4paper/step_cos_sim_k0.csv data4paper/step_cos_sim_k1.csv --labels 'Baseline' 'Proposed ($\kappa=1$)' --out data4paper/cos_sim_step.svg
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLS = ["step_index", "min_cos_sim", "max_cos_sim", "avg_cos_sim", "std_cos_sim"]

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


def plot_one(ax, df: pd.DataFrame, label: str, color: str) -> None:
    x = df["step_index"].values
    y = df["avg_cos_sim"].values
    y_lo = df["min_cos_sim"].values
    y_hi = df["max_cos_sim"].values

    ax.fill_between(x, y_lo, y_hi, alpha=0.25, color=color,
                    label="min–max range")
    ax.plot(x, y, color=color, linewidth=2.4, label="avg")

    ax.set_title(label)
    ax.set_xlabel("Step index")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)


def plot_std_diff(ax, baseline_df: pd.DataFrame, proposed_df: pd.DataFrame,
                  baseline_label: str, proposed_label: str) -> None:
    """Plot std_cos_sim(proposed) - std_cos_sim(baseline) against step index."""
    # Align on step_index in case the two CSVs aren't already aligned.
    merged = pd.merge(
        baseline_df[["step_index", "std_cos_sim"]].rename(columns={"std_cos_sim": "std_baseline"}),
        proposed_df[["step_index", "std_cos_sim"]].rename(columns={"std_cos_sim": "std_proposed"}),
        on="step_index",
        how="inner",
    ).sort_values("step_index")

    x = merged["step_index"].values
    diff = merged["std_baseline"].values - merged["std_proposed"].values

    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.fill_between(x, 0.0, diff, where=(diff >= 0), alpha=0.25, color="#2ca02c",
                    interpolate=True, label="proposed < baseline")
    ax.fill_between(x, 0.0, diff, where=(diff < 0), alpha=0.25, color="#9467bd",
                    interpolate=True, label="proposed > baseline")
    ax.plot(x, diff, color="#333333", linewidth=2.4, label=r"$\Delta$ std")

    # Symmetric y-axis around 0, with a small margin so the curve doesn't touch the frame.
    max_abs = float(np.max(np.abs(diff))) if len(diff) else 1.0
    if max_abs == 0.0:
        max_abs = 1.0
    y_lim = max_abs * 1.1
    ax.set_ylim(-y_lim, y_lim)

    ax.set_title(f"Std difference ({proposed_label} − {baseline_label})")
    ax.set_xlabel("Step index")
    ax.set_ylabel(r"$\Delta$ std of cosine similarity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)


def plot_curves(dfs: list[pd.DataFrame], labels: list[str], out_path: Path | None) -> None:
    # Baseline = cool blue, Proposed = warm orange — colorblind-friendlier than red/blue.
    colors = ["#1f77b4", "#d62728"]

    # Three subplots: baseline avg, proposed avg, std difference.
    # Only the first two share the y-axis (cosine similarity); the third has its own scale.
    fig, axes = plt.subplots(1, 3, figsize=(22, 9))

    # Share y between the first two for direct visual comparison of avg cos sim.
    axes[0].sharey(axes[1])

    for ax, df, label, color in zip(axes[:2], dfs, labels, colors):
        plot_one(ax, df, label, color)

    # Third subplot: std(proposed) - std(baseline).
    plot_std_diff(axes[2], dfs[0], dfs[1], labels[0], labels[1])

    # Match x ticks to actual step indices on all subplots.
    n_steps = len(dfs[0])
    tick_step = max(1, n_steps // 12)
    for ax in axes:
        ax.set_xticks(range(0, n_steps, tick_step))

    axes[0].set_ylabel("Cosine similarity")
    # fig.suptitle("Cosine similarity per step (avg with min/max bounds)", y=1.02)
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

    dfs = [load_and_validate(p) for p in args.csv_files]

    for path, df, label in zip(args.csv_files, dfs, args.labels):
        print(f"{label} ({path}): {len(df)} steps, "
              f"avg range [{df['avg_cos_sim'].min():.4f}, {df['avg_cos_sim'].max():.4f}], "
              f"std range [{df['std_cos_sim'].min():.4f}, {df['std_cos_sim'].max():.4f}]")

    plot_curves(dfs, args.labels, args.out)


if __name__ == "__main__":
    main()