import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
data = {
    0.0001: {"FID": 20.93, "METEOR": 30.22},
    0.0005: {"FID": 21.53, "METEOR": 29.81},
    0.001:  {"FID": 20.57, "METEOR": 29.93},
    0.005:  {"FID": 21.01, "METEOR": 30.11},
}

output_file = 'data4paper/ablation_lambda.svg'

# Unpack and sort data
sorted_lambdas = sorted(data.keys())
fids = [data[l]["FID"] for l in sorted_lambdas]
meteors = [data[l]["METEOR"] for l in sorted_lambdas]

# --- Style Settings for Nature ---
plt.rcParams.update({
    "text.usetex": False,  # Use mathtext instead of LaTeX for simplicity
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "legend.fontsize": 14,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "figure.figsize": (6, 4),  # Nature standard single-column width
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- Create Figure ---
fig, ax1 = plt.subplots(figsize=(6, 4))

# --- Primary Y-Axis (FID) ---
color_fid = '#D35400'  # Dark Orange
ax1.set_xlabel(r'$\lambda$', fontsize=16, fontweight='bold')
ax1.set_ylabel('FID Score', color=color_fid, fontsize=16, fontweight='bold')
line1 = ax1.plot(
    sorted_lambdas, fids,
    color=color_fid,
    marker='o',
    linestyle='-',
    linewidth=2.5,
    markersize=7,
    label='FID Score'
)
ax1.tick_params(axis='y', labelcolor=color_fid)
ax1.tick_params(axis='x', which='both', direction='in')
ax1.set_xscale('log')

# Manually set x-tick labels for clean appearance
ax1.set_xticks(sorted_lambdas)
ax1.set_xticklabels([r'$10^{-4}$', r'$5\times10^{-4}$', r'$10^{-3}$', r'$5\times10^{-3}$'])

# --- Secondary Y-Axis (METEOR) ---
ax2 = ax1.twinx()
color_meteor = '#16A085'  # Teal
ax2.set_ylabel('METEOR Score', color=color_meteor, fontsize=16, fontweight='bold')
line2 = ax2.plot(
    sorted_lambdas, meteors,
    color=color_meteor,
    marker='s',
    linestyle='--',
    linewidth=2.5,
    markersize=7,
    label='METEOR Score'
)
ax2.tick_params(axis='y', labelcolor=color_meteor)

# --- Legend ---
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(
    lines, labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.2),
    ncol=2,
    frameon=False,
    fontsize=14
)

# --- Final Layout ---
fig.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for legend

# --- Save ---
plt.savefig(output_file, format='svg', bbox_inches='tight', dpi=300)
# Optional: also save as PDF for LaTeX inclusion
# plt.savefig('data4paper/ablation_lambda.pdf', format='pdf', bbox_inches='tight', dpi=300)

# Uncomment to display
# plt.show()