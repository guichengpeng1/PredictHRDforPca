#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${1:-outputs/tcga_wsi_fold2_seed_sweep_tv_resnet50_cls_ep10}"
SEEDS="${2:-42 52 62 72 82}"
FOLD="${3:-2}"
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
FOCUS_PATIENT="${14:-TCGA-EJ-7330}"

mkdir -p "$ROOT_DIR"

for SEED in $SEEDS; do
  OUT_DIR="${ROOT_DIR}/seed${SEED}"
  mkdir -p "$OUT_DIR"
  echo "[INFO] Starting fold ${FOLD} seed ${SEED} -> ${OUT_DIR}"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 /home/ubuntu/miniconda3/envs/starf/bin/python \
    code/train_tcga_wsi_mil.py \
    --manifest outputs/tcga_wsi_manifest.csv \
    --output-dir "$OUT_DIR" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --num-tiles "$NUM_TILES" \
    --tile-size 224 \
    --stride 224 \
    --target-mag 10 \
    --device "$DEVICE" \
    --fold "$FOLD" \
    --task "$TASK" \
    --selection-metric "$SELECTION_METRIC" \
    --seed "$SEED" \
    --pretrained \
    --amp

  DIAG_ARGS=(
    /home/ubuntu/miniconda3/envs/starf/bin/python
    code/diagnose_tcga_fold.py
    --run-dir "$OUT_DIR"
    --split val
    --device "$DEVICE"
    --num-workers 0
    --dataset-seed "$INFERENCE_SEED_BASE"
    --repeats "$INFERENCE_REPEATS"
    --output-dir "$OUT_DIR/diagnostics_avg${INFERENCE_REPEATS}"
  )
  if [[ -n "$FOCUS_PATIENT" ]]; then
    DIAG_ARGS+=(--focus-patient "$FOCUS_PATIENT")
  fi
  "${DIAG_ARGS[@]}"
done

SUMMARY_ARGS=(
  /home/ubuntu/miniconda3/envs/starf/bin/python
  code/summarize_seed_sweep.py
  --run-root "$ROOT_DIR"
  --split val
  --diagnostics-subdir "diagnostics_avg${INFERENCE_REPEATS}"
  --output-csv "$ROOT_DIR/seed_sweep_summary.csv"
  --output-json "$ROOT_DIR/seed_sweep_summary.json"
)
if [[ -n "$FOCUS_PATIENT" ]]; then
  SUMMARY_ARGS+=(--focus-patient "$FOCUS_PATIENT")
fi
"${SUMMARY_ARGS[@]}"
