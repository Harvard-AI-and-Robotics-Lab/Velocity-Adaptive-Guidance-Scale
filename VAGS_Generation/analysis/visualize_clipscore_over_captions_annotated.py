import pandas as pd
import matplotlib.pyplot as plt

# === Paths ===
# input_csv = "/path/to/datasets/coco17/validation/clip_scores.csv"  # change this to your actual filename
# output = "data4paper/coco17_average_clip_score_cliplarge.pdf"

input_csv = "/path/to/datasets/flickr30k/test/clip_scores.csv"  # change this to your actual filename
output = "data4paper/flickr1k_average_clip_score_cliplarge.pdf"

# === Load ===
df = pd.read_csv(input_csv)

# === Aggregate ===
# Expect columns: 'caption_index' and 'clip_score'
avg_scores = df.groupby("caption_index")["clip_score"].mean().reset_index()

# === Plot ===
plt.figure(figsize=(6, 4))
bars = plt.bar(avg_scores["caption_index"], avg_scores["clip_score"], width=0.6)
plt.xlabel("Caption Index")
plt.ylabel("Average CLIPScore")
plt.title("Average CLIPScore per Caption Index")
plt.xticks(avg_scores["caption_index"])

# dynamic y-limit with headroom for labels
ymax = float(avg_scores["clip_score"].max())
plt.ylim(0, ymax * 1.10)

# annotate each bar with its value
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.2f}",
                 (bar.get_x() + bar.get_width() / 2, height),
                 ha='center', va='bottom', fontsize=8,
                 xytext=(0, 3), textcoords='offset points')

plt.tight_layout()

# === Save ===
plt.savefig(output, format="pdf")
plt.close()
print(f"✅ Saved plot with annotations to {output}")
