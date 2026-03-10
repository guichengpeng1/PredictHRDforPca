import argparse
import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from build_tcga_wsi_manifest import build_manifest
from wsi_mil_dataset import WSITileBagDataset
from wsi_mil_model import WSIAttentionMIL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(train: bool, image_size: int):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.10,
                    hue=0.02,
                ),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(ops)


def safe_auc(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def determine_selection_metric(task: str, selection_metric: str) -> str:
    if selection_metric != "auto":
        return selection_metric
    if task == "regression":
        return "val_mae"
    return "val_auc"


def metric_sort_value(metric_name: str, metrics: dict) -> float:
    value = metrics[metric_name]
    if np.isnan(value):
        return float("-inf") if metric_name not in {"val_mae", "val_loss"} else float("inf")
    if metric_name in {"val_mae", "val_loss"}:
        return -float(value)
    return float(value)


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(
            targets > 0.5,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1.0 - self.alpha),
        )
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)
        return (focal_weight * bce).mean()


def build_train_sampler(train_df: pd.DataFrame, sampler_name: str):
    if sampler_name != "balanced":
        return None
    labels = train_df["hrd_status"].astype(int).to_numpy()
    class_counts = np.bincount(labels, minlength=2)
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    nonzero = class_counts > 0
    class_weights[nonzero] = 1.0 / class_counts[nonzero]
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def run_epoch(
    model,
    loader,
    device,
    optimizer,
    criterion_score,
    criterion_status,
    alpha: float,
    beta: float,
    amp_enabled: bool,
    task: str,
):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_loss = 0.0
    total_count = 0
    all_scores_true, all_scores_pred = [], []
    all_status_true, all_status_pred = [], []
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if train_mode else None

    for bags, hrd_score, hrd_status, _, _ in loader:
        bags = bags.to(device, non_blocking=True)
        hrd_score = hrd_score.to(device, non_blocking=True)
        hrd_status = hrd_status.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with autocast_context:
            outputs = model(bags)
            loss_terms = []
            if task in {"multitask", "regression"}:
                score_loss = criterion_score(outputs["score"], hrd_score)
                loss_terms.append(alpha * score_loss)
            if task in {"multitask", "classification"}:
                status_loss = criterion_status(outputs["status_logits"], hrd_status)
                loss_terms.append(beta * status_loss)
            loss = sum(loss_terms)

        if train_mode:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        probs = torch.sigmoid(outputs["status_logits"])
        batch_size = bags.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        all_scores_true.append(hrd_score.detach().cpu())
        all_scores_pred.append(outputs["score"].detach().cpu())
        all_status_true.append(hrd_status.detach().cpu())
        all_status_pred.append(probs.detach().cpu())

    score_true = torch.cat(all_scores_true).numpy()
    score_pred = torch.cat(all_scores_pred).numpy()
    status_true = torch.cat(all_status_true).numpy()
    status_pred = torch.cat(all_status_pred).numpy()

    metrics = {
        "loss": total_loss / max(total_count, 1),
        "mae": float(mean_absolute_error(score_true, score_pred))
        if task in {"multitask", "regression"}
        else float("nan"),
        "auc": safe_auc(status_true, status_pred)
        if task in {"multitask", "classification"}
        else float("nan"),
        "ap": safe_ap(status_true, status_pred)
        if task in {"multitask", "classification"}
        else float("nan"),
    }
    return metrics


def select_fold(
    manifest_df: pd.DataFrame,
    fold: int,
    n_splits: int,
):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(splitter.split(manifest_df, manifest_df["hrd_status"]))
    train_idx, val_idx = splits[fold]
    return (
        manifest_df.iloc[train_idx].reset_index(drop=True),
        manifest_df.iloc[val_idx].reset_index(drop=True),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TCGA-PRAD WSI MIL baseline.")
    parser.add_argument(
        "--slide-root",
        type=Path,
        default=Path("/media/ubuntu/Sandisk22T/tcga-PRAD-SLIDE"),
    )
    parser.add_argument(
        "--clinical-csv",
        type=Path,
        default=Path("TCGA-PRAD391/clinical.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/tcga_wsi_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tcga_wsi_mil_baseline"),
    )
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-tiles", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--target-mag", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--task",
        type=str,
        default="multitask",
        choices=["multitask", "classification", "regression"],
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="auto",
        choices=["auto", "val_auc", "val_ap", "val_mae", "val_loss"],
    )
    parser.add_argument(
        "--classification-loss",
        type=str,
        default="bce",
        choices=["bce", "focal"],
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--train-sampler",
        type=str,
        default="random",
        choices=["random", "balanced"],
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/tile_cache"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--max-train-slides", type=int, default=None)
    parser.add_argument("--max-val-slides", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.manifest.exists():
        build_manifest(
            slide_root=args.slide_root,
            clinical_csv=args.clinical_csv,
            output_csv=args.manifest,
            include_all_slides=False,
        )

    manifest_df = pd.read_csv(args.manifest)
    train_df, val_df = select_fold(manifest_df, args.fold, args.n_splits)
    if args.max_train_slides is not None:
        train_df = (
            train_df.sample(n=args.max_train_slides, random_state=args.seed)
            .reset_index(drop=True)
        )
    if args.max_val_slides is not None:
        val_df = (
            val_df.sample(n=args.max_val_slides, random_state=args.seed)
            .reset_index(drop=True)
        )

    train_dataset = WSITileBagDataset(
        train_df,
        num_tiles=args.num_tiles,
        tile_size=args.tile_size,
        target_magnification=args.target_mag,
        stride=args.stride,
        transform=build_transforms(train=True, image_size=args.tile_size),
        cache_dir=args.cache_dir,
        training=True,
        seed=args.seed,
    )
    val_dataset = WSITileBagDataset(
        val_df,
        num_tiles=args.num_tiles,
        tile_size=args.tile_size,
        target_magnification=args.target_mag,
        stride=args.stride,
        transform=build_transforms(train=False, image_size=args.tile_size),
        cache_dir=args.cache_dir,
        training=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.train_sampler == "random",
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        sampler=build_train_sampler(train_df, args.train_sampler),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    model = WSIAttentionMIL(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion_score = nn.MSELoss()
    train_pos = int(train_df["hrd_status"].sum())
    train_neg = int(len(train_df) - train_pos)
    pos_weight = torch.tensor(
        [train_neg / max(train_pos, 1)],
        dtype=torch.float32,
        device=device,
    )
    if args.classification_loss == "focal":
        criterion_status = BinaryFocalLossWithLogits(
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
        )
    else:
        criterion_status = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    selection_metric = determine_selection_metric(args.task, args.selection_metric)

    config_path = args.output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    metrics_path = args.output_dir / "metrics.csv"
    metric_fields = [
        "epoch",
        "train_loss",
        "train_mae",
        "train_auc",
        "train_ap",
        "val_loss",
        "val_mae",
        "val_auc",
        "val_ap",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=metric_fields,
        )
        writer.writeheader()

    best_score = float("-inf")
    best_ckpt = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            criterion_score=criterion_score,
            criterion_status=criterion_status,
            alpha=args.alpha,
            beta=args.beta,
            amp_enabled=amp_enabled,
            task=args.task,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            criterion_score=criterion_score,
            criterion_status=criterion_status,
            alpha=args.alpha,
            beta=args.beta,
            amp_enabled=False,
            task=args.task,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_auc": train_metrics["auc"],
            "train_ap": train_metrics["ap"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_auc": val_metrics["auc"],
            "val_ap": val_metrics["ap"],
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=metric_fields)
            writer.writerow(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_auc={train_metrics['auc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_mae={val_metrics['mae']:.4f}"
        )

        selection_payload = {
            "val_auc": val_metrics["auc"],
            "val_ap": val_metrics["ap"],
            "val_mae": val_metrics["mae"],
            "val_loss": val_metrics["loss"],
        }
        current_score = metric_sort_value(selection_metric, selection_payload)
        if current_score > best_score:
            best_score = current_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "selection_metric": selection_metric,
                },
                best_ckpt,
            )
            print(f"Saved checkpoint to {best_ckpt} by {selection_metric}")


if __name__ == "__main__":
    main()
