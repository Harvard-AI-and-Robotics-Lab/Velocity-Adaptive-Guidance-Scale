import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
# It's good practice to sort the data by the independent variable (lambda)
# to ensure the lines on the plot connect in the correct order.
data = {
    0.0001: {"FID": 20.93, "METEOR": 30.22},
    0.0005: {"FID": 21.53, "METEOR": 29.81},
    0.001:  {"FID": 20.57, "METEOR": 29.93},
    0.005:  {"FID": 21.01, "METEOR": 30.11},
}

output_file = 'data4paper/ablation_lambda.svg'

# Unpack the sorted data into separate lists for plotting
sorted_lambdas = sorted(data.keys())
fids = [data[l]["FID"] for l in sorted_lambdas]
meteors = [data[l]["METEOR"] for l in sorted_lambdas]

# --- Plotting ---
# Use a visually appealing style
plt.style.use('seaborn-v0_8-whitegrid')

# Create the figure and the primary y-axis
fig, ax1 = plt.subplots(figsize=(12, 8)) # Increased height slightly for better spacing

# --- Primary Y-Axis (FID) ---
# New color for FID: Dark Orange
color_fid = '#D35400' 
ax1.set_xlabel('$\lambda$ (Lambda)', fontsize=20, fontweight='bold')
ax1.set_ylabel('FID Score', color=color_fid, fontsize=18, fontweight='bold')
# Plot FID data with markers
line1 = ax1.plot(sorted_lambdas, fids, color=color_fid, marker='o', linestyle='-', linewidth=2.5, markersize=9, label='FID Score')
ax1.tick_params(axis='y', labelcolor=color_fid, labelsize=16)
ax1.tick_params(axis='x', labelsize=16)

# Set the x-axis to a logarithmic scale, which is appropriate for these lambda values
ax1.set_xscale('log')

# --- Secondary Y-Axis (METEOR) ---
# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
# New color for METEOR: Teal
color_meteor = '#16A085'
ax2.set_ylabel('METEOR Score', color=color_meteor, fontsize=18, fontweight='bold')
# Plot METEOR data with a different marker style
line2 = ax2.plot(sorted_lambdas, meteors, color=color_meteor, marker='s', linestyle='--', linewidth=2.5, markersize=9, label='METEOR Score')
ax2.tick_params(axis='y', labelcolor=color_meteor, labelsize=16)


# --- Final Touches ---
# Title for the entire plot with larger font
# plt.title('Ablation Study: Effect of $\lambda$ on FID and METEOR Scores', fontsize=22, fontweight='bold', pad=20)

# Create a combined legend for both lines with larger font
# We get the handles and labels from both plots and combine them
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), fancybox=True, shadow=True, ncol=2, fontsize=16)

# Ensure the plot layout is tight and all elements are visible
fig.tight_layout()

# --- Save the Figure ---
# Instead of displaying the plot, save it to a file.
# Using .svg format for a high-quality vector graphic.
# bbox_inches='tight' ensures the legend isn't cut off.
plt.savefig(output_file, format='svg', bbox_inches='tight')

# If you still want to see the plot after saving, you can add plt.show() after savefig.
# plt.show()