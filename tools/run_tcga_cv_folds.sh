#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${1:-outputs/tcga_wsi_mil_cv_resnet18_tiles16_ep8}"
FOLDS="${2:-1 2 3 4}"

mkdir -p "$ROOT_DIR"

for FOLD in $FOLDS; do
  OUT_DIR="${ROOT_DIR}/fold${FOLD}"
  mkdir -p "$OUT_DIR"
  echo "[INFO] Starting fold ${FOLD} -> ${OUT_DIR}"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 /home/ubuntu/miniconda3/envs/starf/bin/python \
    code/train_tcga_wsi_mil.py \
    --manifest outputs/tcga_wsi_manifest.csv \
    --output-dir "$OUT_DIR" \
    --backbone resnet18 \
    --epochs 8 \
    --batch-size 2 \
    --num-workers 4 \
    --num-tiles 16 \
    --tile-size 224 \
    --stride 224 \
    --target-mag 10 \
    --device cuda:0 \
    --fold "$FOLD" \
    --pretrained \
    --amp
done
