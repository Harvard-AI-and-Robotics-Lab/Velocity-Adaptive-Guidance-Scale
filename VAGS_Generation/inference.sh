#!/bin/bash

source activate base
conda activate CurveFlow

HUGGINGFACE_TOKEN=''
BATCH_SIZE=2
NUM_PROMPTS_PER_RUN=4
IMAGE_WIDTH=512
IMAGE_HEIGHT=512

PROMPT_DIR="/path/to/datasets/coco17/validation"
PROMPT_DIR="/path/to/datasets/coco17/validation"
OUTPUT_DIR='generated_images_coco17'

# MODEL_PATH=(
#             'output_iccv/coco17_no_reweight/checkpoint-26000'
#             'output_iccv/coco17_lognorm/checkpoint-26000'
#             'output_iccv/coco17_modesample/checkpoint-26000'
#             'output_iccv/coco17_lognorm/checkpoint-26000'
#             'output_iccv/coco17_proposed/checkpoint-26000'
#             )
MODEL_PATH=(
            'output_baseline/coco17_sd35_no_reweight/checkpoint-26000'
            'output_baseline/coco17_sd35_lognorm/checkpoint-26000'
            'output_baseline/coco17_sd35_cosmap/checkpoint-26000'
            'output_baseline/coco17_sd35_modesample/checkpoint-26000'
            )
NUM_PROMPTS_PER_RUN=1

for (( i=0; i<${#MODEL_PATH[@]}; i++ ));
do
    python inference_sd3.5.py \
    --prompt_dir=${PROMPT_DIR} \
    --model_path=${MODEL_PATH[$i]} \
    --output_dir=${OUTPUT_DIR} \
    --batch_size=${BATCH_SIZE} \
    --num_prompts_per_run=${NUM_PROMPTS_PER_RUN} \
    --image_width=${IMAGE_WIDTH} \
    --image_height=${IMAGE_HEIGHT}
done