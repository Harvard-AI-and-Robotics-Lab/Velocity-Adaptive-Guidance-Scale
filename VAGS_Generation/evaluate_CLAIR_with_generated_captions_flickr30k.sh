#!/bin/bash

source activate base
conda activate vags_gen


OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"


PROMPT_DIR="/path/to/datasets/flickr30k/validation"
PROMPT_DIR="/path/to/datasets/flickr30k/test_bestcaption"
GEN_IMAGE_DIR=(
            outputs/flickr30k_sd35/vagsgen_k0p5_caption_results.json
            outputs/flickr30k_sd35/vagsgen_k0p7_caption_results.json
            outputs/flickr30k_sd35/vagsgen_k0p9_caption_results.json
            outputs/flickr30k_sd35/vagsgen_k1_caption_results.json
                )

for (( i=0; i<${#GEN_IMAGE_DIR[@]}; i++ ));
do
    PARENT_FOLDER=$(dirname "${GEN_IMAGE_DIR[$i]}")
    FULL_PATH="${PARENT_FOLDER}/${OUTPUT_JSON}"
    python evaluate_CLAIR_with_generated_captions.py \
    --input ${GEN_IMAGE_DIR[$i]} \
    --api-key ${OPENAI_API_KEY}
done