#!/bin/bash

source activate base
conda activate vags_gen

HUGGINGFACE_TOKEN=''
BATCH_SIZE=2
NUM_PROMPTS_PER_RUN=4
NUM_PROMPTS_PER_RUN=1
IMAGE_WIDTH=512
IMAGE_HEIGHT=512

SCHEDULER="flow_euler"
INFERENCE_STEPS=25
# GUIDANCE=( 5.95458 ) # for analysis
GUIDANCE=( 7 )

# seed 42, step 28, guidance 3.5, the fid is not good, ~26.
# seed 227, step 25

ORG_IMAGE_DIR="/path/to/datasets/coco17/validation_bestcaption"
PROMPT_DIR="/path/to/datasets/coco17/validation_bestcaption"
OUTPUT_DIR="outputs/coco17_sd35"

MODEL_PATH=(
            'output_cvpr/coco17_pretrained_sd35/'
            )

# ORG_IMAGE_DIR="/path/to/datasets/flickr30k/test_bestcaption"
# PROMPT_DIR="/path/to/datasets/flickr30k/test_bestcaption"
# OUTPUT_DIR="generated_images_flickr30k_bestcaption_lookahead"

# MODEL_PATH=(
#             'output_cvpr/flickr30k_pretrained_sd35/'
#             )

GAMMA=( 0.95 0.9 )
CURV_TH=( 1.0 ) 

# python inference_sd35_vagsgen_parallel.py \
#     --prompt_dir ${PROMPT_DIR} \
#     --output_dir outputs \
#     --guidance_scale 3.5 \
#     --num_inference_steps 28 \
#     --kappas 0 \
#     --gpus   2

# python inference_sd35_vagsgen_parallel.py \
#     --prompt_dir ${PROMPT_DIR} \
#     --output_dir outputs \
#     --guidance_scale 3.5 \
#     --num_inference_steps 28 \
#     --kappas 0.1 \
#     --gpus   2

# python inference_sd35_vagsgen_parallel.py \
#     --prompt_dir ${PROMPT_DIR} \
#     --output_dir outputs \
#     --seed 227 \
#     --guidance_scale 3.5 \
#     --num_inference_steps 28 \
#     --kappas 0.3 \
#     --gpus   2

# python inference_sd35_vagsgen_parallel.py \
#     --prompt_dir ${PROMPT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --guidance_scale ${GUIDANCE} \
#     --num_inference_steps ${INFERENCE_STEPS} \
#     --seed 227 \
#     --kappas 0.5 0.7 0.9 1.0 \
#     --gpus   2 3 4 5

python inference_sd35_vagsgen_parallel.py \
    --prompt_dir ${PROMPT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --guidance_scale ${GUIDANCE} \
    --num_inference_steps ${INFERENCE_STEPS} \
    --seed 227 \
    --kappas 0 3 \
    --gpus   1 2

python inference_sd35_vagsgen_parallel.py \
    --prompt_dir ${PROMPT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --guidance_scale ${GUIDANCE} \
    --num_inference_steps ${INFERENCE_STEPS} \
    --seed 227 \
    --kappas 4 5 \
    --gpus   1 2

# python inference_sd35_vagsgen_parallel.py \
#     --prompt_dir ${PROMPT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --guidance_scale ${GUIDANCE} \
#     --num_inference_steps ${INFERENCE_STEPS} \
#     --seed 227 \
#     --kappas 0.1 \
#     --gpus   4

# python inference_sd35_vagsgen_parallel.py \
#     --prompt_dir ${PROMPT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --guidance_scale ${GUIDANCE} \
#     --num_inference_steps ${INFERENCE_STEPS} \
#     --seed 227 \
#     --kappas 0.3 \
#     --gpus   4

# python inference_sd35_vagsgen_parallel.py \
#     --prompt_dir ${PROMPT_DIR} \
#     --output_dir outputs \
#     --guidance_scale ${GUIDANCE} \
#     --num_inference_steps ${INFERENCE_STEPS} \
#     --seed 227 \
#     --kappas 1.0 2.0 3.0 4.0 5.0 \
#     --gpus   2 3 4 5 6


# for (( i=0; i<${#GAMMA[@]}; i++ ));
# do
    # OUTPUT_DIR="${BASE_OUTPUT_DIR}_gamma${GAMMA[$i]}_curv${CURV_TH}_step${INFERENCE_STEPS}_g${GUIDANCE}"
    # GEN_IMG_DIR="${MODEL_PATH}/${OUTPUT_DIR}"

    # python inference_sd35_lookahead.py \
    # --prompt_dir=${PROMPT_DIR} \
    # --model_path=${MODEL_PATH} \
    # --output_dir=${OUTPUT_DIR} \
    # --batch_size=${BATCH_SIZE} \
    # --num_prompts_per_run=${NUM_PROMPTS_PER_RUN} \
    # --image_width=${IMAGE_WIDTH} \
    # --image_height=${IMAGE_HEIGHT} \
    # --num_inference_steps=${INFERENCE_STEPS} \
    # --guidance_scale=${GUIDANCE} \
    # --scheduler=${SCHEDULER} \
    # --seed 227 \
    # --lookahead_backtrack_factor ${GAMMA[$i]} \
    # --lookahead_curv_threshold ${CURV_TH}

    # python evaluate_on_image_generation.py \
    # --prompt_dir=${PROMPT_DIR} \
    # --gt_img_dir=${ORG_IMAGE_DIR} \
    # --gen_img_dir=${GEN_IMG_DIR}

    # python evaluate_with_blip_caption.py \
    # --caption_folder=${PROMPT_DIR} \
    # --image_folder=${GEN_IMG_DIR}

    # python assemble_performance_scores.py \
    # --csv_file="${MODEL_PATH}/${OUTPUT_DIR}_img_quality.txt" \
    # --json_file="${MODEL_PATH}/${OUTPUT_DIR}_caption_results.json" \
    # --output="${MODEL_PATH}/${OUTPUT_DIR}_perf.txt"
# done
