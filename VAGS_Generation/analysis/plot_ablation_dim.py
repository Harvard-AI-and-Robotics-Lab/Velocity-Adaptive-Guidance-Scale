import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.patches as mpatches

# Configure matplotlib for publication quality
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 14
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.major.size'] = 5
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.size'] = 5
rcParams['ytick.major.width'] = 1.2
rcParams['xtick.minor.size'] = 3
rcParams['xtick.minor.width'] = 0.8
rcParams['ytick.minor.size'] = 3
rcParams['ytick.minor.width'] = 0.8
rcParams['legend.frameon'] = False
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

# --- Data ---
data = {
    32: {"FID": 21.10, "METEOR": 30.35},
    64: {"FID": 20.57, "METEOR": 29.93},
    128: {"FID": 20.93, "METEOR": 29.78},
}
output_file = 'data4paper/ablation_dim.svg'

# Unpack the sorted data
sorted_lambdas = sorted(data.keys())
fids = [data[l]["FID"] for l in sorted_lambdas]
meteors = [data[l]["METEOR"] for l in sorted_lambdas]

# --- Create Figure ---
fig, ax1 = plt.subplots(figsize=(7, 5))  # Nature standard single column width

# --- Colors (Nature-style palette) ---
color_fid = '#E74C3C'      # Elegant red
color_meteor = '#3498DB'    # Professional blue

# --- Primary Y-Axis (FID) ---
ax1.set_xlabel('Dimension of Coefficient Network', fontsize=18, fontweight='normal')
ax1.set_ylabel('FID score', color=color_fid, fontsize=18, fontweight='normal')

# Plot FID with error-bar style caps for professional look
line1 = ax1.plot(sorted_lambdas, fids, 
                  color=color_fid, 
                  marker='o', 
                  markersize=8,
                  markerfacecolor=color_fid,
                  markeredgecolor='white',
                  markeredgewidth=1.5,
                  linestyle='-', 
                  linewidth=2.5,
                  label='FID',
                  zorder=3)

# Style the primary y-axis
ax1.tick_params(axis='y', labelcolor=color_fid, labelsize=14, direction='out', length=5)
ax1.tick_params(axis='x', labelsize=14, direction='out', length=5)
ax1.set_xscale('log', base=2)  # Use base 2 for powers of 2

# Set y-axis limits with some padding
ax1.set_ylim(20.0, 21.5)

# Add minor ticks for professional appearance
ax1.xaxis.set_minor_locator(plt.NullLocator())  # No minor ticks for discrete values
ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

# --- Secondary Y-Axis (METEOR) ---
ax2 = ax1.twinx()
ax2.spines['right'].set_visible(True)  # Show right spine for secondary axis
ax2.spines['top'].set_visible(False)

ax2.set_ylabel('METEOR score', color=color_meteor, fontsize=18, fontweight='normal')

# Plot METEOR with square markers
line2 = ax2.plot(sorted_lambdas, meteors, 
                  color=color_meteor, 
                  marker='s', 
                  markersize=7,
                  markerfacecolor=color_meteor,
                  markeredgecolor='white',
                  markeredgewidth=1.5,
                  linestyle='--', 
                  linewidth=2.5,
                  dashes=(6, 3),  # Custom dash pattern
                  label='METEOR',
                  zorder=3)

ax2.tick_params(axis='y', labelcolor=color_meteor, labelsize=14, direction='out', length=5)
ax2.set_ylim(29.5, 30.5)
ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

# --- Grid ---
ax1.grid(True, which='major', axis='both', alpha=0.3, linewidth=0.7, linestyle='-')
ax1.grid(True, which='minor', axis='both', alpha=0.15, linewidth=0.5, linestyle='-')
ax1.set_axisbelow(True)

# --- Legend ---
# Create custom legend with clean styling
legend_elements = [
    plt.Line2D([0], [0], color=color_fid, marker='o', markersize=8, 
               markerfacecolor=color_fid, markeredgecolor='white', 
               markeredgewidth=1.5, linestyle='-', linewidth=2.5, label='FID'),
    plt.Line2D([0], [0], color=color_meteor, marker='s', markersize=7, 
               markerfacecolor=color_meteor, markeredgecolor='white', 
               markeredgewidth=1.5, linestyle='--', linewidth=2.5, 
               dashes=(6, 3), label='METEOR')
]

legend = ax1.legend(handles=legend_elements, 
                    loc='best', 
                    fontsize=14,
                    frameon=False,
                    handlelength=2.5,
                    handletextpad=0.5,
                    columnspacing=1.0)

# --- Format x-axis labels ---
ax1.set_xticks(sorted_lambdas)
ax1.set_xticklabels([str(val) for val in sorted_lambdas])

# Rotate x-labels if needed for clarity
for tick in ax1.get_xticklabels():
    tick.set_rotation(0)

# --- Additional styling ---
# Make spines thinner and darker
for spine in ax1.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#333333')
    
for spine in ax2.spines.values():
    if spine.spine_type == 'right':
        spine.set_linewidth(1.2)
        spine.set_color('#333333')

# Ensure proper spacing
fig.tight_layout()

# --- Save the Figure ---
plt.savefig(output_file, 
            format='svg', 
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

# Optional: Also save as high-resolution PNG for immediate viewing
# plt.savefig(output_file.replace('.svg', '.png'), 
#             format='png', 
#             dpi=300,
#             bbox_inches='tight',
#             facecolor='white',
#             edgecolor='none')

# plt.show()