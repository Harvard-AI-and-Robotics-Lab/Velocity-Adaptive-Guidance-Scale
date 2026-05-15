#!/bin/bash

source activate base
conda activate CurveFlow

PROMPT_DIR="/path/to/datasets/coco17/validation"
PROMPT_DIR="/path/to/datasets/coco17/validation"
OUTPUT_JSON='coco17_pickscore_results.txt'

# GEN_IMAGE_DIR=(
#                 'output_iccv/coco17_proposed/checkpoint-26000/generated_images_coco17'
#                  )
GEN_IMAGE_DIR=(
            # 'output/coco17_proposed_lambda0001_1e-5_sigmoid/checkpoint-26000/generated_images_coco17'
            # '/path/to/flow_matching_diffusion/flow_reweighting/output_iccv/coco17_no_reweight/checkpoint-26000/generated_images_coco17'
            'output_baseline/coco17_sd35_no_reweight/checkpoint-26000/generated_images_coco17'
            'output_baseline/coco17_sd35_lognorm/checkpoint-26000/generated_images_coco17'
            'output_baseline/coco17_sd35_modesample/checkpoint-26000/generated_images_coco17'
            'output_baseline/coco17_sd35_cosmap/checkpoint-26000/generated_images_coco17'
                 )
                 

for (( i=0; i<${#GEN_IMAGE_DIR[@]}; i++ ));
do
    PARENT_FOLDER=$(dirname "${GEN_IMAGE_DIR[$i]}")
    FULL_PATH="${PARENT_FOLDER}/${OUTPUT_JSON}"
    python evaluate_pickscore.py \
    --caption_folder=${PROMPT_DIR} \
    --image_folder=${GEN_IMAGE_DIR[$i]} \
    --output_file=${FULL_PATH}
done