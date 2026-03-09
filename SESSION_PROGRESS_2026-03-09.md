# Session Progress and Machine Snapshot

Date: 2026-03-09
Project: 2025年度粤港科技创新联合资助专题
Workspace: /Users/gui/Desktop/项目归并_2026-03-07/国自然/2025年度粤港科技创新联合资助专题

## Research progress

### 1. Materials reviewed
- 2025粤港科技创新联合资助专题可行性研究报告v2.桂程鹏.docx
- 申请伦理及IIT系统/广州医科大学附属第一医院 临床研究方案v4.20260202-观察性研究-clean.docx
- code/model_wsi_backbone.py
- code/trainer.py
- code/ensemble_predict.py
- code/dataloader.py

### 2. Research framework extracted
- Study target: prostate cancer precision diagnosis and treatment.
- Core task 1: build a multi-center WSI-HRD prediction model.
- Core task 2: fuse WSI and clinical variables to predict PARP inhibitor benefit.
- Core task 3: interpret key image regions and mechanism-related morphology.
- Core task 4: validate across centers and support clinical translation.

### 3. Model framework extracted
- Input: multi-magnification WSI + HRD labels from WES + clinical variables.
- Preprocessing: tissue detection, patch extraction, background filtering, normalization.
- Feature extractors: multiple backbones such as ResNet, EfficientNet, Swin, ConvNeXt, ViT, UniNet.
- Fusion: multi-magnification feature fusion + model ensemble.
- Prediction heads:
  - HRD score regression
  - HRD status classification
- Downstream model: fuse image outputs with PSA, Gleason, TNM, age and other variables for PARPi efficacy prediction.

### 4. Current codebase status
- Current implementation is a simplified dual-magnification prototype, not a full WSI MIL pipeline yet.
- code/model_wsi_backbone.py:
  - one backbone for 10x
  - one backbone for 20x
  - feature concatenation
  - two heads: HRD score + HRD status
- code/trainer.py:
  - multitask loss = MSELoss + BCELoss
- code/ensemble_predict.py:
  - average ensemble across 6 backbones
- code/dataloader.py:
  - currently reads sample-level 10x/20x jpg files

### 5. Feasibility conclusions already reached
- 391 WSI cases: trainable on this machine if data are stored on external SSD.
- 1000 WSI cases: also feasible on this machine, but only with a pragmatic pipeline.
- Recommended route:
  - external SSD storage
  - precompute patch features first
  - then run MIL / bag-level aggregation
  - start with 10x or 10x+20x
  - start with 1 backbone, then expand to 2-3 backbones if needed
- Not recommended as first pass:
  - full end-to-end training from raw WSI across all magnifications
  - many backbones trained repeatedly on raw patches
  - storing millions of tiny png/jpg files as the main dataset format

### 6. Main blockers on current machine
- Internal disk is almost exhausted for this task.
- External SSD is required for raw WSI, patch/features, caches, and checkpoints.
- Conda Python is usable, but key packages are still missing:
  - timm
  - openslide-python

## Current machine snapshot

### Hardware
- Model Name: MacBook Pro
- Model Identifier: Mac16,5
- Chip: Apple M4 Max
- CPU cores: 16 (12 performance + 4 efficiency)
- Memory: 128 GB

### OS
- macOS 26.4
- Build: 25E5223i
- uname: Darwin 25.4.0 arm64

### Storage
- Internal system disk: 926 GiB
- Used: 12 GiB
- Available: 7.5 GiB
- Current workspace size: 1.1G

### Python / training environment
- Preferred Python: /Users/gui/miniconda3/bin/python
- Python version: 3.13.2
- PyTorch: 2.10.0
- CUDA available: false
- MPS available: true
- Installed and confirmed:
  - pandas
  - scikit-learn
  - pillow
- Missing and needed:
  - timm
  - openslide-python

### Training interpretation for this machine
- Good fit for: feature extraction, MIL training, moderate model development, cross-validation, prototype ensemble.
- Weak point: slower than Linux + NVIDIA CUDA for heavy end-to-end WSI experiments.
- Storage is the main practical limit, not RAM.

## Suggested comparison points for the next machine
- CPU / GPU type
- RAM size
- Internal and external SSD speed and free space
- Python path and version
- Torch version
- CUDA or MPS availability
- openslide-python available or not
- timm available or not
- Ability to read WSI directly
- Expected pipeline: raw WSI end-to-end vs precomputed features

## Next-step checklist when resuming on another machine
- Mount external SSD.
- Run ./tools/collect_machine_snapshot.sh and save the output.
- Compare the new snapshot with this file.
- Recreate or verify the Python environment.
- Install timm and openslide-python before training.
- Decide whether to start from:
  - 391-case pilot training
  - 1000-case scaled training

## Resume on Linux workstation (2026-03-09)

### Snapshot file generated
- current_machine_snapshot_2026-03-09_linux.md

### Linux workstation summary
- OS: Ubuntu 22.04.5 LTS
- CPU: 2 x Intel Xeon Gold 6252, 96 logical CPUs total
- RAM: 2.0 TiB
- GPU:
  - NVIDIA GeForce RTX 5090, 32 GB
  - NVIDIA GeForce RTX 2080 Ti, 11 GB
- Storage:
  - system disk `/`: 271G total, 198G free
  - external dataset disk `/media/ubuntu/CosMx`: 4.6T total, 2.7T free
- Preferred training Python:
  - `/home/ubuntu/miniconda3/envs/starf/bin/python`
- ML stack in `starf`:
  - torch 2.10.0.dev20251122+cu128
  - CUDA visible: true
  - openslide: OK
  - pandas / sklearn / PIL: OK
  - timm: missing

### Direct comparison vs previous MacBook Pro
- This Linux workstation is much better suited for actual WSI training than the MacBook Pro.
- Main improvements:
  - real CUDA GPUs instead of MPS only
  - much larger RAM headroom
  - much more practical external storage capacity already mounted
  - better fit for feature extraction, MIL training, and repeated experiments
- Remaining blocker:
  - `timm` is still missing in the active training environment, so the current prototype code will not run yet.

### Research conclusion after machine comparison
- 391-case pilot training is fully feasible on this Linux workstation.
- 1000-case scaled training is also feasible if the workflow is kept pragmatic:
  - raw WSI on external disk
  - patch or tile features precomputed first
  - MIL / bag-level training after feature extraction
  - prioritize RTX 5090 as the main training GPU
- Compared with the Mac, this machine removes the main hardware bottleneck. The limiting factor is now pipeline maturity, not compute.

### Codebase status re-confirmed on Linux
- Current code is still a prototype for sample-level dual-magnification JPG input, not a production WSI MIL pipeline.
- `code/dataloader.py` expects one `sample_id.jpg` per magnification.
- `code/model_wsi_backbone.py` depends on `timm` and concatenates 10x/20x features directly.
- Therefore, before large-scale training, the next real engineering step should be dataset/pipeline conversion rather than only rerunning the current scripts.

### Most practical next steps from here
- Install `timm` into `starf`.
- Keep using `tools/collect_machine_snapshot.sh`, which has now been updated to support both macOS and Linux.
- Build a WSI-oriented data manifest:
  - case ID
  - slide path
  - magnification
  - HRD score / HRD status
  - train/val/test split
- Start with one backbone on the RTX 5090.
- Prefer feature precomputation + MIL aggregation before attempting any raw end-to-end multi-backbone training.

## Training continuation on Linux (2026-03-09 evening)

### What was implemented
- Installed `timm` into `/home/ubuntu/miniconda3/envs/starf`.
- Added a runnable TCGA WSI baseline pipeline:
  - `code/build_tcga_wsi_manifest.py`
  - `code/wsi_mil_dataset.py`
  - `code/wsi_mil_model.py`
  - `code/train_tcga_wsi_mil.py`
- Added a minimal `.gitignore` to avoid pushing `outputs/`, `.DS_Store`, and `._*` resource-fork files to GitHub.

### Current training assumption
- Use TCGA as the supervised HRD source.
- Do not mix MRI and pathology into the same model.
- Do not auto-fill labels into unlabeled TCIA data during supervised training.
- Current baseline is therefore:
  - modality: pathology WSI
  - source: `tcga-PRAD-SLIDE`
  - task: HRD score regression + HRD status classification
  - model: single-magnification attention MIL
  - backbone: `resnet18`

### Labeled manifest built
- Output file: `outputs/tcga_wsi_manifest.csv`
- Summary:
  - 391 rows
  - 391 patients
  - 10 HRD-positive
  - 381 HRD-negative
- Selection rule:
  - one preferred slide per patient
  - prioritize lower `DX` rank, e.g. `DX1`

### Smoke training result
- Output dir: `outputs/smoke_tcga_wsi_mil`
- Purpose:
  - verify end-to-end training on real WSI data
  - check tile sampling, GPU forward/backward, metrics logging, checkpoint saving
- Status:
  - passed
- Result from the smoke run:
  - epoch 1
  - train_loss: 135.0009
  - val_loss: 373.6496
  - val_auc: 0.3333
  - val_mae: 18.0817
- Interpretation:
  - pipeline is now runnable
  - these metrics are not meaningful for performance judgment because the smoke run used only a tiny subset

### Important practical note
- A first attempt to detach a longer training job into the background did not stay alive, so the preferred next action is to run the full baseline in a persistent session or supervisor instead of relying on one-shot `nohup`.

### Immediate next target
- Run a full fold baseline on TCGA with the new MIL pipeline and collect real held-out metrics.
- After that, decide whether unlabeled TCIA pathology slides should be used for pseudo-labeling or external inference only.
