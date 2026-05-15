#!/bin/bash

source activate base
conda activate CurveFlow

PROMPT_DIR="/path/to/datasets/coco17/validation"
PROMPT_DIR="/path/to/datasets/coco17/validation"
OUTPUT_JSON='coco17_imgreward_results.json'

GEN_IMAGE_DIR=(
            # 'output/coco17_proposed_lambda0001_1e-5_sigmoid/checkpoint-26000/generated_images_coco17'
            'output_baseline/coco17_sd35_no_reweight/checkpoint-26000/generated_images_coco17'
            'output_baseline/coco17_sd35_lognorm/checkpoint-26000/generated_images_coco17'
            'output_baseline/coco17_sd35_modesample/checkpoint-26000/generated_images_coco17'
            'output_baseline/coco17_sd35_cosmap/checkpoint-26000/generated_images_coco17'
                 )
                 

for (( i=0; i<${#GEN_IMAGE_DIR[@]}; i++ ));
do
    PARENT_FOLDER=$(dirname "${GEN_IMAGE_DIR[$i]}")
    FULL_PATH="${PARENT_FOLDER}/${OUTPUT_JSON}"
    python evaluate_image_reward.py \
    --caption_dir=${PROMPT_DIR} \
    --image_dir=${GEN_IMAGE_DIR[$i]} \
    --output=${FULL_PATH}
done