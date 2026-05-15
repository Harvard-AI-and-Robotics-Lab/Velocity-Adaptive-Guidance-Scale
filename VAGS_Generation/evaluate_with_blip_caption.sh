#!/bin/bash

source activate base
conda activate vags_gen

PROMPT_DIR="/path/to/datasets/coco17/validation"
PROMPT_DIR="/path/to/datasets/coco17/validation"
OUTPUT_JSON='coco17_caption_results.json'

GEN_IMAGE_DIR=(
            # 'output/coco17_sd35_lognorm/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_proposed_RCFM/checkpoint-26000'
            # 'output/coco17_proposed_RCFM/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_proposed_sd35_RCFM_lr1e-4/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_proposed_euc0_riem1/checkpoint-26000/generated_images_coco17'
            'output/coco17_sd35_no_reweight/checkpoint-26000/generated_images_coco17'
            'output/coco17_sd35_proposed__eps1e-1/checkpoint-26000/generated_images_coco17'
            'output/coco17_sd35_proposed__eps1e-4/checkpoint-26000/generated_images_coco17'
            'output/coco17_sd35_proposed__eps5e-1/checkpoint-26000/generated_images_coco17'
            'output/coco17_sd35_proposed__eps5e-4/checkpoint-26000/generated_images_coco17'
            'output/coco17_sd35_proposed__eps1e-2/checkpoint-26000/generated_images_coco17'
            'output/coco17_sd35_proposed__eps1e-3/checkpoint-26000/generated_images_coco17'
            'output/coco17_sd35_proposed__eps5e-2/checkpoint-26000/generated_images_coco17'
            'output/coco17_sd35_proposed__eps1e-1/checkpoint-26000/generated_images_coco17'
            'output/coco17_sd35_proposed_eps1e-2/checkpoint-26000/generated_images_coco17'
            'output/coco17_sd35_proposed_eps1e-3/checkpoint-26000/generated_images_coco17'
            'output/coco17_sd35_proposed_eps5e-2/checkpoint-26000/generated_images_coco17'
                 )

# PROMPT_DIR="/path/to/datasets/coco14/val"
# OUTPUT_JSON='coco14_caption_results.json'
# GEN_IMAGE_DIR=(
#             # '/path/to/flow_matching_diffusion/flow_reweighting/output_iccv/coco17_cosmap/checkpoint-26000/generated_images_coco14'
#             # '/path/to/flow_matching_diffusion/flow_reweighting/output_iccv/coco17_lognorm/checkpoint-26000/generated_images_coco14'
#             # '/path/to/flow_matching_diffusion/flow_reweighting/output_iccv/coco17_modesample/checkpoint-26000/generated_images_coco14'
#             # '/path/to/flow_matching_diffusion/flow_reweighting/output_iccv/coco17_no_reweight/checkpoint-26000/generated_images_coco14'
#             # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_1415_forpaper'
#             '/path/to/flow_matching_diffusion/flow_reweighting/output_iccv/coco17_modesample/checkpoint-26000/generated_images_coco14'
#             )

for (( i=0; i<${#GEN_IMAGE_DIR[@]}; i++ ));
do
    PARENT_FOLDER=$(dirname "${GEN_IMAGE_DIR[$i]}")
    FULL_PATH="${PARENT_FOLDER}/${OUTPUT_JSON}"
    python evaluate_with_blip_caption.py \
    --caption_folder=${PROMPT_DIR} \
    --image_folder=${GEN_IMAGE_DIR[$i]} \
    --output_file=${FULL_PATH}
done