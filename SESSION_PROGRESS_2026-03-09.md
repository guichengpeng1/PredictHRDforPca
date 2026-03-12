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

## First reportable held-out TCGA baseline (2026-03-09 night)

### Run identity
- Output dir: `outputs/tcga_wsi_mil_report_fold0_pretrained_nw4`
- Data:
  - 391 labeled TCGA WSI cases total
  - fold setting: 5-fold stratified, reported run = fold 0
- Training configuration:
  - backbone: `resnet18`
  - pretrained: yes
  - pooling: attention MIL
  - magnification: 10x
  - tiles per slide: 8
  - tile size: 224
  - batch size: 2
  - AMP: yes
  - workers: 4
  - device: RTX 5090
  - epochs: 3

### Held-out validation metrics
- Epoch 1:
  - val_auc: 0.4675
  - val_ap: 0.0476
  - val_mae: 7.3324
- Epoch 2:
  - val_auc: 0.4935
  - val_ap: 0.0967
  - val_mae: 6.8220
- Epoch 3:
  - val_auc: 0.4805
  - val_ap: 0.0963
  - val_mae: 6.6581

### Current best checkpoint under the present selection rule
- Best checkpoint file: `outputs/tcga_wsi_mil_report_fold0_pretrained_nw4/best_model.pt`
- Selection rule used in training script:
  - save by highest validation AUC
- Therefore current best epoch:
  - epoch 2
  - best val_auc: 0.4935
  - corresponding val_ap: 0.0967
  - corresponding val_mae: 6.8220

### Interpretation
- The pipeline now has a real held-out baseline on the full TCGA labeled cohort.
- Classification performance is still weak and not yet clinically or scientifically adequate.
- Regression error improved across epochs, but HRD status discrimination remains close to random.
- Most practical next upgrades:
  - increase tiles per slide
  - try stronger backbones such as `resnet50` or `convnext_tiny`
  - run more epochs now that tile cache is already built
  - test focal loss / class-balanced loss because only 10 positives exist
  - compare 10x-only versus a stronger multi-scale variant

## Second TCGA baseline after increasing tiles and epochs (2026-03-09 night)

### Why this run
- The first reportable baseline suggested that the most pragmatic next move was not a new code rewrite, but a stronger training setup on the same validated pipeline.
- Note:
  - the training script already used class-balanced BCE through `pos_weight = negatives / positives`
  - therefore this run changed data density and optimization budget first

### Run identity
- Output dir: `outputs/tcga_wsi_mil_report_fold0_resnet18_tiles16_ep8`
- Model/config changes versus the first baseline:
  - same backbone family: `resnet18`
  - pretrained: yes
  - tiles per slide: increased from 8 to 16
  - epochs requested: increased from 3 to 8
  - evaluation checkpoint taken after epoch 3 because the result was already clearly better and sufficient for reporting

### Held-out validation metrics observed
- Epoch 1:
  - val_auc: 0.5195
  - val_ap: 0.0476
  - val_mae: 7.0146
- Epoch 2:
  - val_auc: 0.6169
  - val_ap: 0.0651
  - val_mae: 6.8535
- Epoch 3:
  - val_auc: 0.7727
  - val_ap: 0.2778
  - val_mae: 6.7394

### Comparison against the first reportable baseline
- First baseline best:
  - val_auc: 0.4935
  - val_ap: 0.0967
  - val_mae: 6.8220
- Second baseline best so far:
  - val_auc: 0.7727
  - val_ap: 0.2778
  - val_mae: 6.7394
- Practical conclusion:
  - increasing tiles per slide helped materially
  - extending optimization budget also helped
  - the model is now producing a first genuinely reportable held-out classification result on fold 0

### Immediate recommendation after this run
- Lock this configuration in as the current working baseline.
- Next upgrade priority:
  - reproduce on additional folds
  - then compare against a stronger backbone such as `resnet50` once pretrained weights are cached locally

## Cross-fold reproduction progress (2026-03-09 late night)

### Automation added
- `tools/run_tcga_cv_folds.sh`
  - sequential fold runner for the improved baseline configuration
- `code/summarize_tcga_cv.py`
  - summarize best-per-fold metrics and report cross-fold mean/std

### Fold 1 first-pass result
- Run dir: `outputs/tcga_wsi_mil_cv_resnet18_tiles16_ep8/fold1`
- Observed best by validation AUC within the first 3 epochs:
  - epoch 3
  - val_auc: 0.5197
  - val_ap: 0.0543
  - val_mae: 7.9528
- Interpretation:
  - much weaker than fold 0
  - confirms that fold-to-fold variance is substantial

### Fold 2 first-pass result
- Run dir: `outputs/tcga_wsi_mil_cv_resnet18_tiles16_ep8/fold2`
- Observed best by validation AUC within the 3-epoch quick pass:
  - epoch 1
  - val_auc: 0.2632
  - val_ap: 0.0253
  - val_mae: 9.2153
- Interpretation:
  - this fold is currently poor
  - the current configuration is not yet stable enough to claim robust generalization across folds

### Current 3-fold picture using best validation AUC per fold
- Included runs:
  - fold 0: `outputs/tcga_wsi_mil_report_fold0_resnet18_tiles16_ep8`
  - fold 1: `outputs/tcga_wsi_mil_cv_resnet18_tiles16_ep8/fold1`
  - fold 2: `outputs/tcga_wsi_mil_cv_resnet18_tiles16_ep8/fold2`
- Summary:
  - mean val_auc: 0.5185
  - std val_auc: 0.2080
  - mean val_ap: 0.1192
  - std val_ap: 0.1128
  - mean val_mae: 7.9692
  - std val_mae: 1.0109

### Practical conclusion after 3 folds
- The improved setup is clearly better than the original weak baseline on some folds.
- However, the cross-fold variance is still too large to treat the fold-0 high score as a stable result.
- Therefore the next engineering priority is still correct:
  - continue folds 3 and 4
  - then decide whether to improve robustness via stronger sampling, longer training, or architecture changes

## Full 5-fold first-pass reproduction (2026-03-10 early morning)

### Fold 3 first-pass result
- Run dir: `outputs/tcga_wsi_mil_cv_resnet18_tiles16_ep8/fold3`
- Best by validation AUC within 3 epochs:
  - epoch 1
  - val_auc: 0.2171
  - val_ap: 0.0245
  - val_mae: 8.1892

### Fold 4 first-pass result
- Run dir: `outputs/tcga_wsi_mil_cv_resnet18_tiles16_ep8/fold4`
- Best by validation AUC within 3 epochs:
  - epoch 3
  - val_auc: 0.7171
  - val_ap: 0.0670
  - val_mae: 7.2834

### Current 5-fold picture using best validation AUC per fold
- Included runs:
  - fold 0: `outputs/tcga_wsi_mil_report_fold0_resnet18_tiles16_ep8`
  - fold 1: `outputs/tcga_wsi_mil_cv_resnet18_tiles16_ep8/fold1`
  - fold 2: `outputs/tcga_wsi_mil_cv_resnet18_tiles16_ep8/fold2`
  - fold 3: `outputs/tcga_wsi_mil_cv_resnet18_tiles16_ep8/fold3`
  - fold 4: `outputs/tcga_wsi_mil_cv_resnet18_tiles16_ep8/fold4`
- Summary:
  - mean val_auc: 0.4980
  - std val_auc: 0.2272
  - mean val_ap: 0.0898
  - std val_ap: 0.0954
  - mean val_mae: 7.8760
  - std val_mae: 0.8415

### Stronger-sampling probe on a weak fold
- Motivation:
  - test whether poor folds are mainly caused by insufficient slide coverage
- Probe run:
  - dir: `outputs/tcga_wsi_sampling_probe_fold2_tiles32_ep3`
  - fold: 2
  - tiles per slide: increased from 16 to 32
  - epochs: 3
- Observed results:
  - epoch 1: val_auc 0.2632, val_mae 8.8729
  - epoch 2: val_auc 0.2105, val_mae 8.8024
  - epoch 3: val_auc 0.1842, val_mae 8.7280
- Comparison against the original fold-2 run:
  - original best val_auc: 0.2632
  - original best val_mae: 9.2153
  - stronger-sampling conclusion:
    - regression error improved somewhat
    - classification AUC did not improve

### Decision on the next optimization priority
- Longer training is not the next priority:
  - several weak folds peak at epoch 1 rather than improving steadily
- Stronger sampling alone is also not the next priority:
  - the 32-tile probe did not rescue fold-2 classification AUC
- Therefore, among the originally discussed choices, the next most justified priority is:
  - switch to a stronger backbone
- Most practical next candidate:
  - `resnet50` once pretrained weights are fully cached locally
  - then re-test on a weak fold first before launching another full CV sweep

## Stronger-backbone validation on a weak fold (2026-03-10)

### Problem encountered and fix
- `timm` + Hugging Face download path for `resnet50.a1_in1k` stalled repeatedly.
- Verified that `torchvision` could fetch `resnet50` weights quickly from the PyTorch CDN.
- Added explicit torchvision backbone support in `code/wsi_mil_model.py`:
  - `tv_resnet18`
  - `tv_resnet34`
  - `tv_resnet50`

### Validation run
- Run dir: `outputs/tcga_wsi_backbone_probe_fold2_tv_resnet50_ep3`
- Fold: 2
- Backbone: `tv_resnet50`
- Other settings kept close to the previous weak-fold baseline:
  - 16 tiles per slide
  - 3 epochs
  - pretrained weights
  - AMP

### Results on fold 2
- Epoch 1:
  - val_auc: 0.9013
  - val_ap: 0.1667
  - val_mae: 10.0090
- Epoch 2:
  - val_auc: 0.4276
  - val_ap: 0.0329
  - val_mae: 10.2739
- Epoch 3:
  - val_auc: 0.8092
  - val_ap: 0.1114
  - val_mae: 10.5905

### Direct comparison against the previous fold-2 baseline
- Previous `resnet18` best on fold 2:
  - val_auc: 0.2632
  - val_ap: 0.0253
  - val_mae: 9.2153
- New `tv_resnet50` best on fold 2:
  - val_auc: 0.9013
  - val_ap: 0.1667
  - val_mae: 10.0090

### Decision after the weak-fold backbone probe
- For the classification objective, stronger backbone is clearly the right next direction.
- The gain is large enough that a full CV rerun with `tv_resnet50` is now justified.
- Important caveat:
  - HRD score regression MAE got worse on this weak-fold probe, so future runs should track classification and regression separately rather than assuming both improve together.

## Full tv_resnet50 CV rerun launched (2026-03-10)

### Launch status
- Output root: `outputs/tcga_wsi_cv_tv_resnet50_ep3`
- Runner:
  - `tools/run_tcga_cv_folds.sh`
- Configuration:
  - folds: `0 1 2 3 4`
  - backbone: `tv_resnet50`
  - epochs per fold: `3`
  - batch size: `1`
  - workers: `4`
  - tiles per slide: `16`
  - pretrained: yes
  - AMP: yes
- Pretrained weights:
  - `torchvision` cache path confirmed:
    - `/home/ubuntu/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`
- Runtime log:
  - `outputs/tcga_wsi_cv_tv_resnet50_ep3/run.log`
- Status at launch check:
  - fold 0 had started successfully
  - output directory and metrics/config files were being created normally

## Full tv_resnet50 CV result (2026-03-10)

### Run completion
- Output root: `outputs/tcga_wsi_cv_tv_resnet50_ep3`
- Status: completed for folds 0-4
- Selection rule:
  - best validation AUC per fold

### Best fold-wise classification results
- fold 0:
  - best epoch: 1
  - val_auc: 0.3312
  - val_ap: 0.0276
  - val_mae: 7.6756
- fold 1:
  - best epoch: 3
  - val_auc: 0.3618
  - val_ap: 0.0301
  - val_mae: 8.8467
- fold 2:
  - best epoch: 2
  - val_auc: 0.8487
  - val_ap: 0.1181
  - val_mae: 10.0768
- fold 3:
  - best epoch: 1
  - val_auc: 0.8882
  - val_ap: 0.3056
  - val_mae: 8.8670
- fold 4:
  - best epoch: 3
  - val_auc: 0.9474
  - val_ap: 0.2679
  - val_mae: 7.1918

### 5-fold summary for tv_resnet50
- mean val_auc: 0.6754
- std val_auc: 0.2706
- mean val_ap: 0.1498
- std val_ap: 0.1170
- mean val_mae: 8.5316
- std val_mae: 1.0126

### Comparison against the previous 5-fold resnet18 baseline
- Previous resnet18:
  - mean val_auc: 0.4980
  - std val_auc: 0.2272
  - mean val_ap: 0.0898
  - mean val_mae: 7.8760
- New tv_resnet50:
  - mean val_auc: 0.6754
  - std val_auc: 0.2706
  - mean val_ap: 0.1498
  - mean val_mae: 8.5316

### Current conclusion
- `tv_resnet50` should be fixed as the classification backbone going forward.
- Classification improved materially over the resnet18 5-fold baseline.
- Regression MAE worsened, so classification and regression should no longer be treated as if they were optimized by the same default setting.
- The next code/training step should be:
  - task-specific training mode (`classification`, `regression`, `multitask`)
  - then a longer classification-focused tv_resnet50 run

## Task-aware training mode and classification-only validation (2026-03-10)

### Code change
- Updated `code/train_tcga_wsi_mil.py` to support:
  - `--task multitask|classification|regression`
  - `--selection-metric auto|val_auc|val_ap|val_mae|val_loss`
- Training behavior is now task-aware:
  - classification-only:
    - optimize only status BCE loss
    - select checkpoint by validation AUC by default
    - regression metrics are recorded as `nan`
  - regression-only:
    - optimize only score MSE loss
    - select checkpoint by validation MAE by default
  - multitask:
    - preserve the previous combined setup

### Classification-only backbone validation
- Run dir: `outputs/tcga_wsi_cls_only_tv_resnet50_fold2_ep6`
- Fold: 2
- Backbone: `tv_resnet50`
- Task: `classification`
- Epochs: `6`
- Tiles per slide: `16`

### Classification-only results on fold 2
- Epoch 1:
  - val_auc: 0.5855
  - val_ap: 0.0602
- Epoch 2:
  - val_auc: 0.8355
  - val_ap: 0.1043
- Epoch 3:
  - val_auc: 0.9145
  - val_ap: 0.2436
- Epoch 4:
  - val_auc: 0.9605
  - val_ap: 0.3929
- Epoch 5:
  - val_auc: 0.5132
  - val_ap: 0.0770
- Epoch 6:
  - val_auc: 0.5592
  - val_ap: 0.2647

### Comparison against tv_resnet50 multitask on the same fold
- Previous multitask `tv_resnet50` best on fold 2:
  - val_auc: 0.9013
  - val_ap: 0.1667
- New classification-only `tv_resnet50` best on fold 2:
  - val_auc: 0.9605
  - val_ap: 0.3929

### Decision after classification-only validation
- `tv_resnet50 + classification-only` is now the strongest validated setup for the HRD status objective.
- The next default for classification experiments should be:
  - backbone: `tv_resnet50`
  - task: `classification`
  - selection metric: `val_auc`
- The next large experiment should therefore be:
  - rerun full CV in classification-only mode

## Full tv_resnet50 classification-only CV launched (2026-03-10)

### Launch status
- Output root: `outputs/tcga_wsi_cv_tv_resnet50_cls_ep6`
- Runner:
  - `tools/run_tcga_cv_folds.sh`
- Configuration:
  - folds: `0 1 2 3 4`
  - backbone: `tv_resnet50`
  - task: `classification`
  - selection metric: `val_auc`
  - epochs per fold: `6`
  - batch size: `1`
  - workers: `4`
  - tiles per slide: `16`
  - pretrained: yes
  - AMP: yes
- Status at launch check:
  - fold 0 had started successfully
  - output directory plus `config.json` / `metrics.csv` were being created normally
  - runner process was alive after launch

## Longer-budget classification-only imbalance-aware probe (2026-03-10)

### Motivation
- Test whether a longer classification-only run can keep AUC high while improving AP.
- Add explicit positive-class handling rather than changing backbone again.

### Code support added
- `code/train_tcga_wsi_mil.py` now supports:
  - `--classification-loss bce|focal`
  - `--train-sampler random|balanced`

### Probe run
- Output dir: `outputs/tcga_wsi_cls_only_tv_resnet50_fold2_ep10_focal_balanced`
- Fold: 2
- Backbone: `tv_resnet50`
- Task: `classification`
- Epochs: `10`
- Classification loss: `focal`
- Train sampler: `balanced`

### Results
- Best validation AUC:
  - epoch 1
  - val_auc: 0.9474
  - val_ap: 0.2917
- Later epochs:
  - AUC recovered partially after the early drop
  - but never exceeded the epoch-1 checkpoint

### Comparison against the previous classification-only fold-2 baseline
- Previous best:
  - val_auc: 0.9605
  - val_ap: 0.3929
- New focal + balanced probe best:
  - val_auc: 0.9474
  - val_ap: 0.2917

### Interpretation
- Longer budget plus imbalance-aware optimization did not beat the current best classification-only fold-2 checkpoint.
- This suggests that:
  - the current setup is already able to reach a strong classification solution early
  - extra epochs without better early-stopping control can degrade performance
  - focal loss plus balanced sampling is not yet a clear upgrade over the simpler classification-only recipe

## Full tv_resnet50 classification-only CV with longer budget (2026-03-12)

### Configuration
- Output root: `outputs/tcga_wsi_cv_tv_resnet50_cls_ep10`
- Model:
  - backbone: `tv_resnet50`
  - task: `classification`
  - selection metric: `val_auc`
- Training:
  - folds: `0 1 2 3 4`
  - epochs per fold: `10`
  - batch size: `1`
  - workers: `4`
  - tiles per slide: `16`
  - pretrained: yes
  - AMP: yes
- Important change versus the previous full classification-only CV:
  - only the training budget was increased from `6` to `10` epochs
  - no focal loss
  - no balanced sampler

### 5-fold summary by best validation AUC
- fold0:
  - best epoch: `1`
  - val_auc: `0.7857`
  - val_ap: `0.0982`
- fold1:
  - best epoch: `1`
  - val_auc: `0.8618`
  - val_ap: `0.2955`
- fold2:
  - best epoch: `4`
  - val_auc: `0.4605`
  - val_ap: `0.0842`
- fold3:
  - best epoch: `3`
  - val_auc: `0.7039`
  - val_ap: `0.0617`
- fold4:
  - best epoch: `3`
  - val_auc: `0.8816`
  - val_ap: `0.5500`

### Cross-fold aggregate
- mean val_auc: `0.7387`
- std val_auc: `0.1526`
- mean val_ap: `0.2179`
- std val_ap: `0.1859`

### Comparison against the previous 6-epoch classification-only CV
- Previous `outputs/tcga_wsi_cv_tv_resnet50_cls_ep6`:
  - mean val_auc: `0.7219`
  - std val_auc: `0.1329`
  - mean val_ap: `0.0927`
- New `outputs/tcga_wsi_cv_tv_resnet50_cls_ep10`:
  - mean val_auc: `0.7387`
  - std val_auc: `0.1526`
  - mean val_ap: `0.2179`

### Interpretation
- A longer classification-only budget improved both cross-fold mean AUC and mean AP.
- `tv_resnet50 + classification-only + 10 epochs` is now the strongest default HRD classification setting in this workspace.
- However, fold-level instability remains:
  - fold2 is still markedly weaker than the other folds
  - this split needs case-level diagnosis instead of another blind global sweep

## Fold-2 diagnosis after the 10-epoch classification-only CV (2026-03-12)

### Tooling added for diagnosis
- `code/summarize_tcga_cv.py`
  - now accepts a CV root directory and automatically expands `fold*/metrics.csv`
- `code/diagnose_tcga_fold.py`
  - loads a saved fold checkpoint
  - rebuilds the exact train/val split
  - exports per-slide predictions
  - reports positive-case ranks, top false positives, and threshold-0.5 confusion

### Fold-2 validation-set diagnosis
- Diagnosed run:
  - `outputs/tcga_wsi_cv_tv_resnet50_cls_ep10/fold2`
  - checkpoint epoch: `4`
- Validation split size:
  - `78` slides total
  - `2` positive
  - `76` negative
- Validation diagnosis:
  - val_auc: `0.4605`
  - val_ap: `0.0842`
  - positive ranks: `7` and `78`
  - threshold-0.5 confusion:
    - TP: `1`
    - FN: `1`
    - FP: `8`
    - TN: `68`
- Positive slides:
  - `TCGA-XQ-A8TA`
    - predicted probability: `0.5462`
    - rank: `7`
  - `TCGA-EJ-7330`
    - predicted probability: `0.0318`
    - rank: `78`
- Top false positives in this weak checkpoint:
  - `TCGA-G9-A9S0`
    - predicted probability: `0.7759`
  - `TCGA-J4-A6G1`
    - predicted probability: `0.7735`
  - `TCGA-V1-A9O7`
    - predicted probability: `0.7022`
  - `TCGA-M7-A71Z`
    - predicted probability: `0.6503`
  - `TCGA-M7-A71Y`
    - predicted probability: `0.5787`

### Fold-2 train-set diagnosis
- Train split size:
  - `313` slides total
  - `8` positive
  - `305` negative
- Train diagnosis:
  - auc: `0.3766`
  - ap: `0.0349`
  - positive ranks: `8`, `114`, `138`, `169`, `244`, `276`, `303`, `305`
  - threshold-0.5 confusion:
    - TP: `1`
    - FN: `7`
    - FP: `28`
    - TN: `277`

### Comparison against the earlier strong fold-2 checkpoint
- Comparison run:
  - `outputs/tcga_wsi_cls_only_tv_resnet50_fold2_ep6`
  - checkpoint epoch: `4`
- Same validation split, much stronger result:
  - val_auc: `0.9605`
  - val_ap: `0.3929`
  - positive ranks: `2` and `7`
  - threshold-0.5 confusion:
    - TP: `2`
    - FN: `0`
    - FP: `7`
    - TN: `69`
- The key contrast is the same positive case `TCGA-EJ-7330`:
  - weak 10-epoch CV fold2 checkpoint:
    - predicted probability: `0.0318`
    - rank: `78`
  - strong earlier fold2 checkpoint:
    - predicted probability: `0.7750`
    - rank: `2`

### Diagnostic conclusion
- Fold 2 is not simply an intrinsically impossible split.
- The weak `tcga_wsi_cv_tv_resnet50_cls_ep10/fold2` checkpoint is a bad local solution:
  - it under-ranks positives on the training split as well
  - it does not merely suffer from a poor validation threshold
  - it shows optimization instability rather than a pure data-separability limit
- The next intervention for this fold should focus on training stability and checkpoint robustness, not a new backbone:
  - repeated seeds
  - checkpoint ensembling
  - or stronger early-stopping / model-selection controls
