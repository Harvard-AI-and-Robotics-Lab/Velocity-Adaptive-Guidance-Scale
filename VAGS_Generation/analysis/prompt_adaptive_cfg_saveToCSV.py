import pandas as pd

input_csv = "outputs/coco17_sd35_analysis/vagsgen_k1_guidance_analysis.csv"  # change this to your actual filename
output_csv = "data4paper/prompt_lambda_i_k1.csv"  # change this to your desired output filename

# Read the input CSV — keep prompt_id as a string to preserve leading zeros
df = pd.read_csv(input_csv, dtype={"prompt_id": str})

# Group by prompt_id and aggregate lambda_i
summary = (
    df.groupby("prompt_id")["lambda_i"]
      .agg(min_lambda_i="min", max_lambda_i="max", avg_lambda_i="mean", std_lambda_i="std")
      .reset_index()
)

# Write to a new CSV
summary.to_csv(output_csv, index=False)

print(summary.head())