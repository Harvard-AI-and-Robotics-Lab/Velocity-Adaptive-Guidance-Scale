#!/bin/bash

source activate base
conda activate vags_gen

ORG_IMAGE_DIR="/path/to/datasets/coco17/validation_bestcaption"
PROMPT_DIR="/path/to/datasets/coco17/validation_bestcaption"
RESULT_FILE='img_quality_metrics_coco17.txt'

GEN_IMAGE_DIR=(
            'output_cvpr/coco17_pretrained_sd35/generated_images_coco17'
                 )

# ORG_IMAGE_DIR="/path/to/datasets/coco14/val"
# PROMPT_DIR="/path/to/datasets/coco14/val"
# RESULT_FILE='img_quality_metrics_coco14.txt'
# GEN_IMAGE_DIR=(
#             # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14'
#             # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_334'
#             # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_8791'
#             # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_1234'
#             # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_5678'
#             # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_91011'
#             # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_1213'
#             # '/path/to/flow_matching_diffusion/flow_reweighting/output_iccv/coco17_cosmap/checkpoint-26000/generated_images_coco14'
#             # 'output/coco17_proposed_lambda0001_1e-5_sigmoid/checkpoint-26000/generated_images_coco14'
#             # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_1415'
#             # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_1617'
#             # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_1819'
#             # '/path/to/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_2021'
#             '/path/to/flow_matching_diffusion/flow_reweighting/output_iccv/coco17_modesample/checkpoint-26000/generated_images_coco14'
#         )

# ORG_IMAGE_DIR="/path/to/datasets/flickr30k/test_bestcaption"
# PROMPT_DIR="/path/to/datasets/flickr30k/test_bestcaption"
# RESULT_FILE='generated_images_flickr30k_bestcaption'

# GEN_IMAGE_DIR=(
#             'output_cvpr/flickr30k_pretrained_sd35/generated_images_flickr30k_bestcaption'
#             # 'output_cvpr/flickr30k_sd35_lognorm/checkpoint-20000/generated_images_flickr30k_bestcaption'
#             # 'output_cvpr/flickr30k_sd35_proposed/checkpoint-20000'
#             # 'output_cvpr/flickr30k_sd35_proposed_bspline/checkpoint-20000'
#             # 'output_cvpr/flickr30k_sd35_proposed_geodesicspline/checkpoint-20000'
#             # 'output_cvpr/flickr30k_sd35_proposed_synchronization/checkpoint-20000'
#             )

for (( i=0; i<${#GEN_IMAGE_DIR[@]}; i++ ));
do
    python evaluate_on_image_generation_proportion.py \
    --prompt_dir=${PROMPT_DIR} \
    --gt_img_dir=${ORG_IMAGE_DIR} \
    --gen_img_dir=${GEN_IMAGE_DIR[$i]} \
    --output=${RESULT_FILE}
done