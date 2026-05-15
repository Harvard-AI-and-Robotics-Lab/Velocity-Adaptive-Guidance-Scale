#!/bin/bash

source activate base
conda activate vags_gen

HUGGINGFACE_TOKEN=''
BATCH_SIZE=2
NUM_PROMPTS_PER_RUN=4
NUM_PROMPTS_PER_RUN=1
IMAGE_WIDTH=512
IMAGE_HEIGHT=512

SCHEDULER="flow_euler" #"flowmatch_euler"
GUIDANCE=( 3.5 )
INFERENCE_STEPS=28

# ORG_IMAGE_DIR="/path/to/datasets/coco17/validation_bestcaption"
# PROMPT_DIR="/path/to/datasets/coco17/validation_bestcaption"
# BASE_OUTPUT_DIR="generated_images_coco17_bestcaption"

ORG_IMAGE_DIR="/path/to/datasets/cub_200/test/"
PROMPT_DIR="/path/to/datasets/cub_200/test/"

OUTPUT_DIRS=(
            # outputs/cub_200_sd35/vagsgen_k0p5
            # outputs/cub_200_sd35/vagsgen_k0p7
            # outputs/cub_200_sd35/vagsgen_k0p9
            outputs/cub_200_sd35/vagsgen_k0
            outputs/cub_200_sd35/vagsgen_k0p3
            outputs/cub_200_sd35/vagsgen_k1
            )



for (( i=0; i<${#OUTPUT_DIRS[@]}; i++ ));
do

python evaluate_on_image_generation.py \
    --prompt_dir=${PROMPT_DIR} \
    --gt_img_dir=${ORG_IMAGE_DIR} \
    --gen_img_dir=${OUTPUT_DIRS[$i]}

python evaluate_with_blip_caption.py \
    --caption_folder=${PROMPT_DIR} \
    --image_folder=${OUTPUT_DIRS[$i]}

parent=$(dirname "${OUTPUT_DIRS[$i]}")
current_dir=$(basename "${OUTPUT_DIRS[$i]}")
python assemble_performance_scores.py \
    --csv_file="${parent}/${current_dir}_img_quality.txt" \
    --json_file="${parent}/${current_dir}_caption_results.json" \
    --output="${parent}/${current_dir}_perf.txt"
done
