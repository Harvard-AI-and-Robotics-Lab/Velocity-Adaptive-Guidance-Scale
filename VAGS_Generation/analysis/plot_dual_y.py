import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import MultipleLocator

# Set up Nature style
plt.rcParams.update({
    'font.sans-serif': ['Arial'],
    'font.family': 'sans-serif',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'figure.dpi': 300,
})

# Data
x = [0, 0.001, 0.01, 0.1, 1]
y1 = [29.07542902583166, 29.084061987198055, 29.59289408795984, 30.515814939859236, 30.07265249656747]
y2 = [37.94647139428155, 37.89192734143206, 38.87953864991642, 40.17152399408919, 39.4908228215931]

# Create figure with specific Nature-style dimensions (89 mm width is standard for single column)
fig, ax1 = plt.subplots(figsize=(3.5, 2.625))  # 89 mm width (3.5 inches)

# Plot the first dataset (METEOR) with blue color
color1 = '#0072B2'  # Blue
ax1.plot(x, y1, color=color1, marker='o', label='METEOR')
ax1.set_xlabel(r'$\lambda$')
ax1.set_ylabel('METEOR', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.spines['top'].set_visible(False)

# Create the second y-axis
ax2 = ax1.twinx()
color2 = '#D55E00'  # Red/orange
ax2.plot(x, y2, color=color2, marker='s', label='ROUGE-1')
ax2.set_ylabel('ROUGE-1', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.spines['top'].set_visible(False)

# Set x-axis to log scale
ax1.set_xscale('log')
ax1.set_xlim([min(x) * 0.5, max(x) * 2])  # Add some padding

# Add grid lines (light gray)
ax1.grid(True, linestyle='--', alpha=0.3)

# Create a combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False)

# Adjust layout and save
plt.tight_layout()
plt.savefig('data4paper/dual_y_plot.pdf', format='pdf', bbox_inches='tight')
# plt.savefig('dual_y_plot.png', format='png', dpi=300, bbox_inches='tight')

plt.show()