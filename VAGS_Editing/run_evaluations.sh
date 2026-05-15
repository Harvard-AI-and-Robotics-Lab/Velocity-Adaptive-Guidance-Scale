#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Evaluate edited images and compute metrics
#
# Usage:
#   bash run_evaluations.sh <edited_images_dir> <output_csv> [--div2k]
#
# Examples:
#   # PIE-Bench (uses foreground masks)
#   bash run_evaluations.sh outputs/pie_bench outputs/pie_bench/results.csv
#
#   # DIV2K (full image, no mask)
#   bash run_evaluations.sh outputs/div2k outputs/div2k/results.csv --div2k
# ─────────────────────────────────────────────────────────────────────────────

conda activate image_editing

EDITED_DIR="${1:-outputs/pie_bench}"
RESULT_CSV="${2:-outputs/pie_bench/results.csv}"
MODE="${3:-}"

PIE_MAPPING="Data/PIE-Bench_v1/mapping_file.json"
PIE_IMAGES="Data/PIE-Bench_v1/annotation_images"

if [[ "$MODE" == "--div2k" ]]; then
    # DIV2K / YAML evaluation (no masks)
    python evaluate.py \
        --yaml_file          Data/flowedit.yaml \
        --src_image_folder   Data/ \
        --edited_images_dir  "$EDITED_DIR" \
        --tgt_methods        flowedit_sd35_conflictaware_cosine \
        --no_pie_mask \
        --result_path        "$RESULT_CSV"
else
    # PIE-Bench evaluation (with masks)
    python evaluate.py \
        --annotation_mapping_file "$PIE_MAPPING" \
        --src_image_folder        "$PIE_IMAGES" \
        --edited_images_dir       "$EDITED_DIR" \
        --tgt_methods             flowedit_sd35_conflictaware_cosine \
        --result_path             "$RESULT_CSV"
fi
