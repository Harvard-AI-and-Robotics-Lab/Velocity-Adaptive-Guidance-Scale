import pandas as pd

input_csv = "outputs/coco17_sd35_analysis/vagsgen_k0_guidance_analysis.csv"  # change this to your actual filename
output_csv = "data4paper/prompt_cos_sim_k0.csv"  # change this to your desired output filename

# input_csv = "outputs/coco17_sd35_analysis/vagsgen_k1_guidance_analysis.csv"  # change this to your actual filename
# output_csv = "data4paper/prompt_cos_sim_k1.csv"  # change this to your desired output filename

# Read the input CSV — keep prompt_id as a string to preserve leading zeros
df = pd.read_csv(input_csv, dtype={"prompt_id": str})

# Group by prompt_id and aggregate cos_sim
summary = (
    df.groupby("prompt_id")["cos_sim"]
      .agg(min_cos_sim="min", max_cos_sim="max", avg_cos_sim="mean", std_cos_sim="std")
      .reset_index()
)

# Write to a new CSV
summary.to_csv(output_csv, index=False)

print(summary.head())