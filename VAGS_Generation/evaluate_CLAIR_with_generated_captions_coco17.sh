#!/bin/bash

source activate base
conda activate vags_gen


OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

PROMPT_DIR="/path/to/datasets/coco17/validation"
PROMPT_DIR="/path/to/datasets/coco17/validation_bestcaption"
GEN_IMAGE_DIR=(
            # outputs/coco17_sd35/vagsgen_k0p5_caption_results.json
            # outputs/coco17_sd35/vagsgen_k0p7_caption_results.json
            # outputs/coco17_sd35/vagsgen_k0p9_caption_results.json
            # outputs/coco17_sd35/vagsgen_k1_caption_results.json
            outputs/coco17_sd35/vagsgen_k0_caption_results.json
            outputs/coco17_sd35/vagsgen_k0p1_caption_results.json
            outputs/coco17_sd35/vagsgen_k0p3_caption_results.json
            outputs/coco17_sd35/vagsgen_k1p5_caption_results.json
            outputs/coco17_sd35/vagsgen_k2_caption_results.json
                )

for (( i=0; i<${#GEN_IMAGE_DIR[@]}; i++ ));
do
    PARENT_FOLDER=$(dirname "${GEN_IMAGE_DIR[$i]}")
    FULL_PATH="${PARENT_FOLDER}/${OUTPUT_JSON}"
    python evaluate_CLAIR_with_generated_captions.py \
    --input ${GEN_IMAGE_DIR[$i]} \
    --api-key ${OPENAI_API_KEY} 
done