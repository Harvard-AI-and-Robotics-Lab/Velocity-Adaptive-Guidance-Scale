import matplotlib.pyplot as plt
import numpy as np

# Data
labels = [r'$\tau_{\text{curv}}$=1', r'$\tau_{\text{curv}}$=15', r'$\tau_{\text{curv}}$=20', r'$\tau_{\text{curv}}$=25']
fids = [75.88, 76.79, 77.55, 77.81]
clipscores = [0.3388, 0.3383, 0.3385, 0.3387]

# Nature journal style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(4.0, 2.5), dpi=300)

# X positions
x = np.arange(len(labels))
width = 0.35

# Plot FID scores on primary y-axis
bars1 = ax1.bar(x - width/2, fids, width, 
                label='FID Score', 
                color='#2E86AB',  # Nature-style blue
                edgecolor='black',
                linewidth=0.5,
                alpha=0.9)

# ax1.set_xlabel(r'Curvature regularization strength ($\tau_{\text{curv}}$)', fontsize=8, fontweight='bold')
ax1.set_ylabel('FID Score', fontsize=8, fontweight='bold', color='#2E86AB')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.tick_params(axis='y', labelcolor='#2E86AB', labelsize=7)
ax1.tick_params(axis='x', labelsize=7)
ax1.set_ylim([75, 79])

# Add grid for better readability (Nature style - subtle)
ax1.grid(axis='y', alpha=0.3, linewidth=0.3, linestyle='--')
ax1.set_axisbelow(True)

# Remove top and right spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Create secondary y-axis for CLIP scores
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, clipscores, width,
                label='CLIP Score',
                color='#E63946',  # Nature-style red
                edgecolor='black',
                linewidth=0.5,
                alpha=0.9)

ax2.set_ylabel('CLIP Score', fontsize=8, fontweight='bold', color='#E63946')
ax2.tick_params(axis='y', labelcolor='#E63946', labelsize=7)
ax2.set_ylim([0.337, 0.340])

# Remove top and left spines from secondary axis
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=6, color='#2E86AB')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=6, color='#E63946')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
          loc='upper left', 
          frameon=False, 
          fontsize=7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save figure
# plt.savefig('/mnt/user-data/outputs/tau_curv_barchart.png', 
#             dpi=300, 
#             bbox_inches='tight',
#             facecolor='white')
plt.savefig('data4paper/ablation_barchart_tau.pdf', 
            bbox_inches='tight',
            facecolor='white')

print("Bar chart saved successfully!")
print("- PNG: tau_curv_barchart.png")
print("- PDF: tau_curv_barchart.pdf")

plt.show()