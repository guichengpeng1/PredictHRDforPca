#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${1:-outputs/tcga_wsi_mil_cv_resnet18_tiles16_ep8}"
FOLDS="${2:-0 1 2 3 4}"
BACKBONE="${3:-resnet18}"
EPOCHS="${4:-3}"
BATCH_SIZE="${5:-2}"
NUM_WORKERS="${6:-4}"
NUM_TILES="${7:-16}"
DEVICE="${8:-cuda:0}"

mkdir -p "$ROOT_DIR"

for FOLD in $FOLDS; do
  OUT_DIR="${ROOT_DIR}/fold${FOLD}"
  mkdir -p "$OUT_DIR"
  echo "[INFO] Starting fold ${FOLD} -> ${OUT_DIR}"
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
    --pretrained \
    --amp
done
