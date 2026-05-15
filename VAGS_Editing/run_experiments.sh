#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Run benchmark: all methods on PIE-Bench or DIV2K
#
# Usage:
#   bash run_experiments.sh               # PIE-Bench (default)
#   bash run_experiments.sh --div2k       # DIV2K
#
# Method indices (--methods flag):
#   9  = FlowEdit (SD 3.5, no scheduler)
#   10 = FlowEdit + Interval
#   11 = FlowEdit + Monotone
#   12 = FlowEdit + Zero-Init
#   13 = FlowEdit + VAGS  ← ours
#   14 = SplitFlow (SD 3.5, no scheduler)
#   17 = SplitFlow + Zero-Init
#   18 = SplitFlow + VAGS ← ours
#
# GPU mapping: one GPU per method (--gpu_map)
# ─────────────────────────────────────────────────────────────────────────────

conda activate image_editing

OUTDIR="outputs/pie_bench"
DATASET_FLAGS="--pie_bench"

if [[ "$1" == "--div2k" ]]; then
    OUTDIR="outputs/div2k"
    DATASET_FLAGS="--yaml Data/flowedit.yaml --images Data/"
fi

python benchmark.py \
    $DATASET_FLAGS \
    --outdir "$OUTDIR" \
    --methods 9 11 12 13 \
    --gpu_map 0 1 2 3
