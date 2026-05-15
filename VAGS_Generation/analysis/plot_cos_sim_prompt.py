"""
Plot violin distributions of avg_cos_sim and the sorted std difference between two
methods, in two side-by-side subplots.

Left subplot:  Violins of avg_cos_sim for Baseline (blue) and Proposed (red),
               drawn side by side in the same axes for direct comparison.
Right subplot: std_cos_sim difference (baseline - proposed), sorted ascending —
               effectively a signed CDF of the std difference across prompts.
               This subplot is only drawn if both CSVs include a `std_cos_sim` column.

Usage:
    python analysis/plot_cos_sim_prompt.py baseline.csv proposed.csv --labels 'Baseline' 'Proposed ($\\kappa=1$)' [--out plot.png]
    python analysis/plot_cos_sim_prompt.py data4paper/prompt_cos_sim_k0.csv data4paper/prompt_cos_sim_k1.csv --labels 'Baseline' 'Proposed ($\kappa=1$)' --out data4paper/cos_sim_prompt.svg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLS = ["avg_cos_sim"]

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
    """Load CSV and validate that avg_cos_sim is present."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def plot_violin_at(ax, values: np.ndarray, position: float, color: str,
                   label: str) -> None:
    """Draw one violin at the given x position with mean/median/std annotations."""
    parts = ax.violinplot(
        values,
        positions=[position],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.7,
    )

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
    ax.vlines(position, q1, q3, color=color, linewidth=8, alpha=0.9, zorder=3)
    ax.vlines(position, vmin, vmax, color=color, linewidth=1.8, alpha=0.9, zorder=2)

    # Median tick and mean diamond.
    ax.scatter([position], [median], marker="_", color="white", s=260, linewidth=3,
               zorder=4)
    ax.scatter([position], [mean], marker="D", color="white", edgecolor=color,
               s=70, linewidth=1.5, zorder=4)

    # Legend entry: a coloured patch-style line plus the headline stats as text.
    ax.plot([], [], color=color, linewidth=8, alpha=0.7,
            label=f"{label}\nmedian={median:.4f}  mean={mean:.4f}  std={std:.4f}")


def plot_violins(ax, dfs: list[pd.DataFrame], labels: list[str],
                 colors: list[str]) -> None:
    """Draw both violins side by side in a single axes."""
    positions = [0, 1]
    for pos, df, label, color in zip(positions, dfs, labels, colors):
        plot_violin_at(ax, df["avg_cos_sim"].values, pos, color, label)

    ax.set_title("avg_cos_sim distribution")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{lbl}\n(n = {len(df)})" for lbl, df in zip(labels, dfs)])
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylabel("Cosine similarity (avg per prompt)")
    ax.grid(True, axis="y", alpha=0.3)
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

    If the two CSVs have different lengths we truncate to the shorter one.
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
    # Baseline = cool blue, Proposed = warm red — same palette as the original script.
    colors = ["#1f77b4", "#d62728"]

    has_std = all("std_cos_sim" in df.columns for df in dfs)
    if not has_std:
        print("Note: `std_cos_sim` not found in both CSVs — drawing only the violin subplot.")

    n_subplots = 2 if has_std else 1
    fig, axes = plt.subplots(1, n_subplots, figsize=(8 * n_subplots + 2, 9))
    axes = np.atleast_1d(axes)

    plot_violins(axes[0], dfs, labels, colors)

    if has_std:
        plot_std_diff(axes[1], dfs[0], dfs[1], labels[0], labels[1])

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

    dfs = [load_csv(p) for p in args.csv_files]

    for path, df, label in zip(args.csv_files, dfs, args.labels):
        msg = (f"{label} ({path}): {len(df)} prompts, "
               f"avg range [{df['avg_cos_sim'].min():.4f}, {df['avg_cos_sim'].max():.4f}]")
        if "std_cos_sim" in df.columns:
            msg += (f", std range "
                    f"[{df['std_cos_sim'].min():.4f}, {df['std_cos_sim'].max():.4f}]")
        print(msg)

    plot_curves(dfs, args.labels, args.out)


if __name__ == "__main__":
    main()