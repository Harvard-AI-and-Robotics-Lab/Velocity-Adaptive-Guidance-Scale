import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Style Settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12})

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
# fig.suptitle('Flow Sampling: Without vs With Stabilization', fontsize=16, fontweight='bold', y=0.95)

TARGET = np.array([0.5, 0.5])

# Ideal trajectory - FIXED BROADCASTING
t = np.linspace(1, 0, 200).reshape(-1, 1)  # Shape (200, 1) for proper broadcasting
z_ideal = (1-t)**3 * np.array([4.5, 4.0]) + 3*(1-t)**2*t * np.array([3.0, 3.5]) + \
          3*(1-t)*t**2 * np.array([1.5, 2.0]) + t**3 * TARGET

# Discrete timesteps
timesteps = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))

def plot_trajectory(ax, path, color, label, title_emoji, title_color):
    """Helper function to plot a trajectory panel"""
    ax.plot(z_ideal[:, 0], z_ideal[:, 1], '--', color='black', 
            linewidth=2.5, alpha=0.7, label='Ideal Continuous Flow', zorder=3)
    
    for i in range(len(path) - 1):
        arrow = FancyArrowPatch(
            path[i], path[i+1], arrowstyle='->', mutation_scale=20,
            linewidth=2.5, color=color, alpha=0.8, connectionstyle="arc3,rad=0.05"
        )
        ax.add_patch(arrow)
        ax.scatter(path[i, 0], path[i, 1], s=100, color=colors[i], 
                  edgecolor=color, linewidth=2, zorder=4)
    
    ax.scatter(path[-1, 0], path[-1, 1], s=150, color=colors[-1], 
              edgecolor=color, linewidth=2, marker='s', label=f'Final Sample ({label})', zorder=4)
    
    ax.scatter(TARGET[0], TARGET[1], s=200, marker='*', color='gold', 
              edgecolor='black', linewidth=1.5, label=r'Target ($p_0$)', zorder=5)
    
    # ax.set_title(f'{title_emoji} {title_color}', fontsize=14, color=title_color, pad=15)
    ax.set_title(f'{title_emoji}', fontsize=14, color=title_color, pad=15)
    # ax.set_xlabel('State Dimension 1', fontweight='bold')
    # ax.set_ylabel('State Dimension 2', fontweight='bold')
    ax.set_xlim(-0.5, 5.0)
    ax.set_ylim(-0.5, 4.5)
    ax.legend(loc='lower right', framealpha=0.95)  # ← CHANGED LOCATION HERE
    ax.set_aspect('equal')
    
    # SNR regions
    ax.add_patch(plt.Rectangle((2.5, 2.5), 2.5, 2.0, fill=True, color='blue', alpha=0.1))
    ax.text(4.7, 3.7, 'Low SNR\nRegion', ha='center', fontsize=9, color='blue', fontweight='bold', alpha=0.7)
    ax.add_patch(plt.Rectangle((-0.2, -0.2), 2.0, 2.0, fill=True, color='orange', alpha=0.1))
    ax.text(0.5, 0.2, 'High SNR\nRegion', ha='center', fontsize=9, color='darkorange', fontweight='bold', alpha=1.0)

# --- LEFT PANEL: WITHOUT STABILIZATION ---
unstable_path = np.array([
    [4.3, 3.8], [3.5, 4.2], [2.8, 3.9], 
    [2.1, 2.8], [1.2, 1.8], [0.8, 1.0]
])
plot_trajectory(ax1, unstable_path, 'red', 'Deviated', 'Flow Sampling without Stabilization', 'darkred')

ax1.annotate('Divergence & Overshoot', xy=(3.5, 4.2), xytext=(1.0, 4.2),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, color='red', fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", alpha=0.7))

# --- RIGHT PANEL: WITH STABILIZATION ---
stabilized_path = np.array([
    [4.3, 3.8], [3.6, 3.4], [2.8, 2.8], 
    [2.0, 2.1], [1.2, 1.3], TARGET
])
plot_trajectory(ax2, stabilized_path, 'green', 'Accurate', 'Flow Sampling with Stabilization', 'darkgreen')

ax2.annotate('Stable Progression', xy=(2.8, 2.8), xytext=(3.5, 2.2),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=10, color='darkgreen', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7))

plt.tight_layout()
plt.savefig('flow_stabilization.svg', dpi=300, bbox_inches='tight')
plt.show()