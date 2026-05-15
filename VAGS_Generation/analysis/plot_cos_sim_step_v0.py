"""
Plot avg_cos_sim vs step_index for baseline and proposed CSVs side by side,
with shaded bands showing [min_cos_sim, max_cos_sim].

Usage:
    python plot_cos_sim.py baseline.csv proposed.csv [--out out.png]
    python analysis/plot_cos_sim_step.py data4paper/step_cos_sim_k0.csv data4paper/step_cos_sim_k1.csv --out data4paper/cos_sim_step.svg
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_one(ax, df, label, color, title):
    x = df["step_index"]
    ax.plot(x, df["avg_cos_sim"], color=color, lw=2, label=f"{label} (avg)")
    ax.fill_between(
        x,
        df["min_cos_sim"],
        df["max_cos_sim"],
        color=color,
        alpha=0.2,
        label=f"{label} (min-max range)",
    )
    ax.set_xlabel("Step index")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_csv", help="Path to baseline CSV")
    parser.add_argument("proposed_csv", help="Path to proposed CSV")
    parser.add_argument("--out", default="cos_sim_comparison.png",
                        help="Output image path")
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_csv)
    proposed = pd.read_csv(args.proposed_csv)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)

    # Distinct colors: baseline vs proposed
    plot_one(axes[0], baseline, "Baseline", "#1f77b4", "Baseline")
    plot_one(axes[1], proposed, r"Proposed ($\kappa=1$)", "#d62728", "Proposed")

    # Match x ticks to actual step indices (0..24)
    for ax in axes:
        ax.set_xticks(range(0, len(baseline), max(1, len(baseline) // 12)))

    # fig.suptitle("Cosine similarity per step (avg with min/max bounds)",
    #              fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()