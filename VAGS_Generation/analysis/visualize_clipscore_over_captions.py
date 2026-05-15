import pandas as pd
import matplotlib.pyplot as plt

input_csv = "/path/to/datasets/coco17/validation/clip_scores.csv"  # change this to your actual filename
output = "data4paper/coco17_average_clip_score_cliplarge.pdf"

# input_csv = "/path/to/datasets/flickr30k/validation/clip_scores.csv"  # change this to your actual filename
# output = "data4paper/coco17_average_clip_score_cliplarge.pdf"

# Load CSV file
df = pd.read_csv(input_csv)  # change this to your actual file path

# Compute average CLIPScore for each caption index across all images
avg_scores = df.groupby("caption_index")["clip_score"].mean().reset_index()

subset = avg_scores[(avg_scores["caption_index"] >= 1) & (avg_scores["caption_index"] <= 5)]
min_val = subset["clip_score"].min()
max_val = subset["clip_score"].max()
min_idx = subset.loc[subset["clip_score"].idxmin(), "caption_index"]
max_idx = subset.loc[subset["clip_score"].idxmax(), "caption_index"]

print(f"📉 Minimum CLIPScore: {min_val:.4f} (Caption Index {min_idx})")
print(f"📈 Maximum CLIPScore: {max_val:.4f} (Caption Index {max_idx})")

# Plot
plt.figure(figsize=(6, 4))
plt.bar(avg_scores["caption_index"], avg_scores["clip_score"], width=0.6)
plt.xlabel("Caption Index")
plt.ylabel("Average CLIPScore")
plt.title("Average CLIPScore per Caption Index")
plt.xticks(avg_scores["caption_index"])
plt.ylim(20, 30)
plt.tight_layout()

# Save to PDF
plt.savefig(output, format="pdf")
plt.close()

print(f"✅ Saved plot to {output}")