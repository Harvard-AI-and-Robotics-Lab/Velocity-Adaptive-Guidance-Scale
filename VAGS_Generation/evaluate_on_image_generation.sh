#!/bin/bash

source activate base
conda activate vags_gen

ORG_IMAGE_DIR="/path/to/datasets/coco17/validation_bestcaption"
PROMPT_DIR="/path/to/datasets/coco17/validation_bestcaption"
RESULT_FILE='img_quality_metrics_coco17.txt'

GEN_IMAGE_DIR=(
            # 'output/coco17_proposed_lambda01/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_proposed_lambda01/checkpoint-24000/generated_images_coco17'
            # 'output/coco17_sd35_lognorm/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_lognorm/checkpoint-24000/generated_images_coco17'
            # 'output/coco17_proposed_sep12/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_proposed_sep12/checkpoint-24000/generated_images_coco17'
            # 'output/coco17_proposed_RCFM/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_proposed_RCFM/checkpoint-24000/generated_images_coco17'
            # 'output/coco17_flux_lr1e-6/checkpoint-14000/generated_images_coco17'
            # 'output/coco17_sd35_proposed/checkpoint-12250/generated_images_coco17_lambda1'
            # 'output/coco17_sd35_proposed/checkpoint-12250/generated_images_coco17_dynlambda'
            # 'output/coco17_sd35_no_reweight/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_proposed__eps1e-1/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_proposed__eps1e-4/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_proposed__eps5e-1/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_proposed__eps5e-4/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_proposed__eps1e-2/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_proposed__eps1e-3/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_proposed__eps5e-2/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_proposed__eps1e-1/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_proposed_eps1e-2/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_proposed_eps1e-3/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_proposed_eps5e-2/checkpoint-26000/generated_images_coco17'
            'output_cvpr/coco17_pretrained_sd35/calibration_kappa0.9_results/images'
            'output_cvpr/coco17_pretrained_sd35/calibration_kappa0.91_results/images'
            'output_cvpr/coco17_pretrained_sd35/generated_images_coco17_g1'
            'output_cvpr/coco17_pretrained_sd35/generated_images_coco17_g2'
            'output_cvpr/coco17_pretrained_sd35/generated_images_coco17_g3'
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
#             'output_cvpr/flickr30k_pretrained_sd35/generated_images_flickr30k_bestcaption_g1'
#             'output_cvpr/flickr30k_pretrained_sd35/generated_images_flickr30k_bestcaption_g2'
#             'output_cvpr/flickr30k_pretrained_sd35/generated_images_flickr30k_bestcaption_g3'
#             'output_cvpr/flickr30k_pretrained_sd35/generated_images_flickr30k_bestcaption_g4'
#             'output_cvpr/flickr30k_pretrained_sd35/generated_images_flickr30k_bestcaption_g5'
#             # 'output_cvpr/flickr30k_sd35_lognorm/checkpoint-20000/generated_images_flickr30k_bestcaption'
#             # 'output_cvpr/flickr30k_sd35_proposed/checkpoint-20000'
#             # 'output_cvpr/flickr30k_sd35_proposed_bspline/checkpoint-20000'
#             # 'output_cvpr/flickr30k_sd35_proposed_geodesicspline/checkpoint-20000'
#             # 'output_cvpr/flickr30k_sd35_proposed_synchronization/checkpoint-20000'
#             )

for (( i=0; i<${#GEN_IMAGE_DIR[@]}; i++ ));
do
    python evaluate_on_image_generation.py \
    --prompt_dir=${PROMPT_DIR} \
    --gt_img_dir=${ORG_IMAGE_DIR} \
    --gen_img_dir=${GEN_IMAGE_DIR[$i]} \
    --output=${RESULT_FILE}
done