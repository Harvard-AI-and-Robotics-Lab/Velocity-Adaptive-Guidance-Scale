#!/bin/bash

source activate base
conda activate vags_gen

HUGGINGFACE_TOKEN=''
BATCH_SIZE=2
NUM_PROMPTS_PER_RUN=4
NUM_PROMPTS_PER_RUN=1
IMAGE_WIDTH=512
IMAGE_HEIGHT=512

SCHEDULER="none" #"flowmatch_euler"
GUIDANCE=( 7 )

# ORG_IMAGE_DIR="/path/to/datasets/coco17/validation_bestcaption"
# PROMPT_DIR="/path/to/datasets/coco17/validation_bestcaption"

# GEN_IMG_DIR=(
#             'output_cvpr/coco17_aflops_sd35/generated_images'
#             'output_cvpr/coco17_selfguidance_sd35/generated_images'
#             )

# ORG_IMAGE_DIR="/path/to/datasets/flickr30k/test_bestcaption"
# PROMPT_DIR="/path/to/datasets/flickr30k/test_bestcaption"

# GEN_IMG_DIR=(
#             'output_cvpr/flickr30k_aflops_sd35/generated_images'
#             'output_cvpr/flickr30k_selfguidance_sd35/generated_images'
#             )

# ORG_IMAGE_DIR="/path/to/datasets/laion_sg/validation/"
# PROMPT_DIR="/path/to/datasets/laion_sg/validation/"

# GEN_IMG_DIR=(
#             'output_cvpr/laion_aflops_sd35/generated_images'
#             'output_cvpr/laion_selfguidance_sd35/generated_images'
#             )

ORG_IMAGE_DIR="/path/to/datasets/cub_200/test/"
PROMPT_DIR="/path/to/datasets/cub_200/test/"

GEN_IMG_DIR=(
            'output_icml/cub200_aflops_sd35/generated_images'
            'output_icml/cud200_selfguidance_sd35/generated_images'
            )


for (( i=0; i<${#GEN_IMG_DIR[@]}; i++ ));
do

    CURRENT_GEN_IMG_DIR="${GEN_IMG_DIR[$i]}"
    PARENTDIR=$(dirname "${GEN_IMG_DIR[$i]}")
    FOLDER=$(basename "${GEN_IMG_DIR[$i]}")

    echo "Parent dir: ${PARENTDIR}, Folder: ${FOLDER}"

    # python inference_sd3.5_scheduler.py \
    # --prompt_dir=${PROMPT_DIR} \
    # --model_path=${MODEL_PATH} \
    # --output_dir="${OUTPUT_DIR}" \
    # --batch_size=${BATCH_SIZE} \
    # --num_prompts_per_run=${NUM_PROMPTS_PER_RUN} \
    # --image_width=${IMAGE_WIDTH} \
    # --image_height=${IMAGE_HEIGHT} \
    # --num_inference_steps=30 \
    # --guidance_scale=${GUIDANCE[$i]} \
    # --scheduler=${SCHEDULER} \
    # --seed 227

    python evaluate_on_image_generation.py \
    --prompt_dir=${PROMPT_DIR} \
    --gt_img_dir=${ORG_IMAGE_DIR} \
    --gen_img_dir="${CURRENT_GEN_IMG_DIR}"

    python evaluate_with_blip_caption.py \
    --caption_folder=${PROMPT_DIR} \
    --image_folder="${CURRENT_GEN_IMG_DIR}"

    python assemble_performance_scores.py \
    --csv_file="${PARENTDIR}/${FOLDER}_img_quality.txt" \
    --json_file="${PARENTDIR}/${FOLDER}_caption_results.json" \
    --output="${PARENTDIR}/${FOLDER}_perf.txt"
done
