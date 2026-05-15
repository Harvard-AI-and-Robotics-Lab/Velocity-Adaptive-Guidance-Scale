"""
Plot cumulative curves of avg_cos_sim from two CSV files in three side-by-side subplots,
with shaded bands showing the [min_cos_sim, max_cos_sim] error bounds.

Left subplot:   Baseline (blue)
Middle subplot: Proposed (orange/red)
Right subplot:  std_cos_sim difference (baseline - proposed), sorted ascending
                — effectively a signed CDF of the std difference across prompts.

X-axis (left/middle): cumulative count of prompts (sorted by avg_cos_sim ascending)
Y-axis (left/middle): avg_cos_sim, with min/max as the shaded band
X-axis (right):       cumulative count of prompts (sorted by the diff itself)
Y-axis (right):       std_cos_sim(baseline) - std_cos_sim(proposed), symmetric around 0

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


def plot_std_diff(ax, baseline_df: pd.DataFrame, proposed_df: pd.DataFrame,
                  baseline_label: str, proposed_label: str) -> None:
    """Plot std_cos_sim(baseline) - std_cos_sim(proposed) as a sorted (CDF-like) curve.

    Why sort the diff itself instead of using the avg-sorted index? With many prompts
    the raw per-rank diff is visually noise (signs alternate every few points and
    the green/purple fills cancel), making it impossible to read whether one method
    has lower std overall. Sorting the diff ascending turns the curve into an
    empirical CDF of the std difference: the x-position where it crosses zero
    is exactly the fraction of prompts where proposed has higher std than baseline,
    and the area above/below zero shows the magnitude of each side's advantage.

    Note: each side's std array is sorted independently by its own avg_cos_sim
    upstream, so the rank-vs-rank pairing matches the other two subplots before
    we re-sort the diff here. If the two CSVs have different lengths we truncate
    to the shorter one.
    """
    n = min(len(baseline_df), len(proposed_df))
    std_baseline = baseline_df["std_cos_sim"].values[:n]
    std_proposed = proposed_df["std_cos_sim"].values[:n]
    diff_unsorted = std_baseline - std_proposed

    # Sort ascending so the curve is monotone — readable as a CDF of the diff.
    diff = np.sort(diff_unsorted)
    x = np.arange(1, n + 1)

    # Summary stats for the annotation.
    frac_proposed_lower = float(np.mean(diff_unsorted > 0))  # baseline - proposed > 0  ⇒ proposed has lower std
    frac_proposed_higher = float(np.mean(diff_unsorted < 0))
    mean_diff = float(np.mean(diff_unsorted))
    median_diff = float(np.median(diff_unsorted))

    # Fills: green where proposed has lower std, purple where it has higher.
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.fill_between(x, 0.0, diff, where=(diff >= 0), alpha=0.35, color="#2ca02c",
                    interpolate=True, label="proposed < baseline")
    ax.fill_between(x, 0.0, diff, where=(diff < 0), alpha=0.35, color="#9467bd",
                    interpolate=True, label="proposed > baseline")
    # Thinner line so the fills carry the visual weight.
    ax.plot(x, diff, color="#222222", linewidth=1.2, label=r"$\Delta$ std (sorted)")

    # Mark the zero-crossing — the fraction of prompts where proposed has lower std.
    zero_cross_idx = int(np.searchsorted(diff, 0.0))
    if 0 < zero_cross_idx < n:
        ax.axvline(zero_cross_idx, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)

    # Symmetric y-axis around 0, with a small margin so the curve doesn't touch the frame.
    max_abs = float(np.max(np.abs(diff))) if len(diff) else 1.0
    if max_abs == 0.0:
        max_abs = 1.0
    y_lim = max_abs * 1.15
    ax.set_ylim(-y_lim, y_lim)

    # Annotation box with the headline numbers.
    summary = (
        f"proposed lower std: {frac_proposed_lower:.1%}\n"
        f"proposed higher std: {frac_proposed_higher:.1%}\n"
        f"mean $\\Delta$: {mean_diff:+.4f}\n"
        f"median $\\Delta$: {median_diff:+.4f}"
    )
    ax.text(
        0.03, 0.97, summary,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )

    ax.set_title(f"Std difference ({baseline_label} − {proposed_label})")
    ax.set_xlabel("Prompts (sorted by $\\Delta$ std)")
    ax.set_ylabel(r"$\Delta$ std of cosine similarity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)


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

    # Third subplot: std(baseline) - std(proposed), sorted.
    plot_std_diff(axes[2], dfs[0], dfs[1], labels[0], labels[1])

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
              f"avg range [{df['avg_cos_sim'].min():.4f}, {df['avg_cos_sim'].max():.4f}], "
              f"std range [{df['std_cos_sim'].min():.4f}, {df['std_cos_sim'].max():.4f}]")

    plot_curves(dfs, args.labels, args.out)


if __name__ == "__main__":
    main()