#!/bin/bash

source activate base
conda activate vags_gen


OPENAI_API_KEY=

PROMPT_DIR="/path/to/datasets/coco17/validation"
PROMPT_DIR="/path/to/datasets/coco17/validation"
OUTPUT_JSON='coco17_clair_results.json'
GEN_IMAGE_DIR=(
            # 'output_iccv_rebuttal/coco17_proposed_lambda05/checkpoint-10000/coco17_caption_results.json'
            # 'output_iccv_rebuttal/coco17_proposed_lambda001/checkpoint-10000/coco17_caption_results.json'
            # 'output_iccv_rebuttal/coco17_proposed_lambda1/checkpoint-10000/coco17_caption_results.json'
            # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco17_caption_results.json'
            # 'output_iccv_rebuttal/coco17_proposed_lambda_curva_01_noReg/checkpoint-10000/coco17_caption_results.json'
            # 'output/coco17_proposed_lambda0001_1e-5_sigmoid/checkpoint-26000/coco17_caption_results.json'
            '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco17_1234_forpaper_caption_results.json'
                )

# PROMPT_DIR="/path/to/datasets/coco14/val"
# PROMPT_DIR="/path/to/datasets/coco14/val"
# OUTPUT_JSON='coco14_clair_results.json'
# GEN_IMAGE_DIR=(
#             # 'output_iccv_rebuttal/coco17_proposed_lambda05/checkpoint-10000/coco17_caption_results.json'
#             # 'output_iccv_rebuttal/coco17_proposed_lambda001/checkpoint-10000/coco17_caption_results.json'
#             # 'output_iccv_rebuttal/coco17_proposed_lambda1/checkpoint-10000/coco17_caption_results.json'
#             # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco17_caption_results.json'
#             # 'output_iccv_rebuttal/coco17_proposed_lambda_curva_01_noReg/checkpoint-10000/coco17_caption_results.json'
#             # 'output/coco17_proposed_lambda0001_1e-5_sigmoid/checkpoint-26000/coco17_caption_results.json'
#             # "/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_1415_forpaper_caption_results.json"
#             '/path/to/flow_matching_diffusion/flow_reweighting/output_iccv/coco17_modesample/checkpoint-26000/generated_images_coco14_caption_results.json'
#                 )
                
                 

for (( i=0; i<${#GEN_IMAGE_DIR[@]}; i++ ));
do
    PARENT_FOLDER=$(dirname "${GEN_IMAGE_DIR[$i]}")
    FULL_PATH="${PARENT_FOLDER}/${OUTPUT_JSON}"
    python evaluate_CLAIR_with_generated_captions.py \
    --input ${GEN_IMAGE_DIR[$i]} \
    --api-key ${OPENAI_API_KEY} \
    --output ${FULL_PATH}
done