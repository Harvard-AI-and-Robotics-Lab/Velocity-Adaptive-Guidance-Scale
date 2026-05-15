import pandas as pd

# input_csv = "outputs/coco17_sd35_analysis/vagsgen_k0_guidance_analysis.csv"  # change this to your actual filename
# output_csv = "data4paper/step_lambda_i_k0.csv"  # change this to your desired output filename

input_csv = "outputs/coco17_sd35_analysis/vagsgen_k1_guidance_analysis.csv"  # change this to your actual filename
output_csv = "data4paper/step_lambda_i_k1.csv"  # change this to your desired output filename


# Read the input CSV
df = pd.read_csv(input_csv, dtype={"prompt_id": str})

step_stats = (
    df.groupby("step_index")["lambda_i"]
      .agg(avg_lambda_i="mean", std_lambda_i="std",
           min_lambda_i="min", max_lambda_i="max", n="count")
      .reset_index()
)
step_stats.to_csv(output_csv, index=False)