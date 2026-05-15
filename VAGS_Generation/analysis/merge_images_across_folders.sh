#!/bin/bash

source activate base
conda activate vags_gen

# python analysis/merge_images_across_folders.py \
#   --input_folder ./data4paper/aesthetic_images_sd35 \
#   --subfolders generated_images_lookahead_gamma0.9_curv1_step25_g7 generated_images_lookahead_gamma0.95_curv1_step25_g7  generated_images_lookahead_gamma1_curv1_step25_g7 generated_images_lookback_lmb0.1_mid0_step25_g7 generated_images_lookback_lmb0.15_mid0_step25_g7   \
#   --output_dir ./data4paper/merged_images

python analysis/merge_images_across_folders.py \
  --input_folder ./data4paper/aesthetic_images_sd35_medium \
  --subfolders generated_images_lookahead_gamma0.9_curv1_step25_g7 generated_images_lookahead_gamma0.95_curv1_step25_g7 generated_images_lookback_lmb0_mid0_step25_g7 generated_images_lookback_lmb0.1_mid0_step25_g7 generated_images_lookback_lmb0.15_mid0_step25_g7   \
  --output_dir ./data4paper/merged_images_medium



