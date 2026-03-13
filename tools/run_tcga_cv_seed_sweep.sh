#!/usr/bin/env bash

set -euo pipefail

CV_ROOT="${1:-outputs/tcga_wsi_cv_seed_sweep_tv_resnet50_cls_ep10_avg10}"
FOLDS="${2:-0 1 2 3 4}"
SEEDS="${3:-42 52 62 72 82}"
BACKBONE="${4:-tv_resnet50}"
EPOCHS="${5:-10}"
BATCH_SIZE="${6:-1}"
NUM_WORKERS="${7:-4}"
NUM_TILES="${8:-16}"
DEVICE="${9:-cuda:0}"
TASK="${10:-classification}"
SELECTION_METRIC="${11:-val_auc}"
INFERENCE_REPEATS="${12:-10}"
INFERENCE_SEED_BASE="${13:-1000}"
TOP_K="${14:-2}"

mkdir -p "$CV_ROOT"

for FOLD in $FOLDS; do
  FOLD_ROOT="${CV_ROOT}/fold${FOLD}"
  mkdir -p "$FOLD_ROOT"
  echo "[INFO] Starting CV fold ${FOLD} -> ${FOLD_ROOT}"
  bash tools/run_tcga_fold_seed_sweep.sh \
    "$FOLD_ROOT" \
    "$SEEDS" \
    "$FOLD" \
    "$BACKBONE" \
    "$EPOCHS" \
    "$BATCH_SIZE" \
    "$NUM_WORKERS" \
    "$NUM_TILES" \
    "$DEVICE" \
    "$TASK" \
    "$SELECTION_METRIC" \
    "$INFERENCE_REPEATS" \
    "$INFERENCE_SEED_BASE" \
    ""

  /home/ubuntu/miniconda3/envs/starf/bin/python \
    code/build_topk_seed_ensemble.py \
    --run-root "$FOLD_ROOT" \
    --top-k "$TOP_K" \
    --predictions-file "diagnostics_avg${INFERENCE_REPEATS}/val_predictions_mean.csv" \
    --output-dir "$FOLD_ROOT/top${TOP_K}_ensemble"
done

/home/ubuntu/miniconda3/envs/starf/bin/python \
  code/summarize_cv_topk_ensemble.py \
  --cv-root "$CV_ROOT" \
  --ensemble-subdir "top${TOP_K}_ensemble" \
  --output-csv "$CV_ROOT/top${TOP_K}_cv_summary.csv" \
  --output-json "$CV_ROOT/top${TOP_K}_cv_summary.json"
