import pandas as pd

input_csv = "outputs/coco17_sd35_analysis/vagsgen_k0_guidance_analysis.csv"  # change this to your actual filename
output_csv = "data4paper/step_cos_sim_k0.csv"  # change this to your desired output filename

input_csv = "outputs/coco17_sd35_analysis/vagsgen_k1_guidance_analysis.csv"  # change this to your actual filename
output_csv = "data4paper/step_cos_sim_k1.csv"  # change this to your desired output filename


# Read the input CSV
df = pd.read_csv(input_csv, dtype={"prompt_id": str})

step_stats = (
    df.groupby("step_index")["cos_sim"]
      .agg(avg_cos_sim="mean", std_cos_sim="std",
           min_cos_sim="min", max_cos_sim="max", n="count")
      .reset_index()
)
step_stats.to_csv(output_csv, index=False)