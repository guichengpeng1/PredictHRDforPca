import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from train_tcga_wsi_mil import build_transforms, select_fold, set_seed
from wsi_mil_dataset import WSITileBagDataset
from wsi_mil_model import WSIAttentionMIL


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose a saved TCGA fold checkpoint.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training output directory that contains best_model.pt and config.json.",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["val"],
        choices=["train", "val"],
        help="One or more data splits to diagnose.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional override for the manifest csv path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device. Falls back to cpu if CUDA is unavailable.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers for deterministic diagnostics.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save diagnostic csv/json outputs. Defaults to <run-dir>/diagnostics.",
    )
    return parser.parse_args()


def load_checkpoint_args(run_dir: Path) -> dict:
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    args = dict(checkpoint["args"])
    args["_checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
    args["_checkpoint_val_metrics"] = checkpoint.get("val_metrics", {})
    args["_selection_metric"] = checkpoint.get("selection_metric")
    return args


def build_split_dataframe(manifest_df: pd.DataFrame, ckpt_args: dict, split_name: str) -> pd.DataFrame:
    train_df, val_df = select_fold(
        manifest_df=manifest_df,
        fold=int(ckpt_args["fold"]),
        n_splits=int(ckpt_args["n_splits"]),
    )
    if split_name == "train":
        return train_df
    return val_df


def build_dataset(split_df: pd.DataFrame, ckpt_args: dict) -> WSITileBagDataset:
    return WSITileBagDataset(
        split_df,
        num_tiles=int(ckpt_args["num_tiles"]),
        tile_size=int(ckpt_args["tile_size"]),
        target_magnification=float(ckpt_args["target_mag"]),
        stride=int(ckpt_args["stride"]),
        transform=build_transforms(train=False, image_size=int(ckpt_args["tile_size"])),
        cache_dir=Path(ckpt_args["cache_dir"]),
        training=False,
        seed=int(ckpt_args["seed"]),
    )


def run_inference(
    model: WSIAttentionMIL,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (bags, hrd_score, hrd_status, patients, slide_paths) in enumerate(loader):
            bags = bags.to(device, non_blocking=True)
            outputs = model(bags)
            probs = torch.sigmoid(outputs["status_logits"]).detach().cpu().numpy()
            logits = outputs["status_logits"].detach().cpu().numpy()
            scores = outputs["score"].detach().cpu().numpy()
            attention = outputs["attention"].detach().cpu().numpy()
            for item_idx in range(len(patients)):
                attn = attention[item_idx]
                attn_entropy = float(-(attn * np.log(np.clip(attn, 1e-8, 1.0))).sum())
                rows.append(
                    {
                        "batch_index": batch_idx,
                        "patient_barcode": patients[item_idx],
                        "slide_path": slide_paths[item_idx],
                        "hrd_score": float(hrd_score[item_idx].item()),
                        "hrd_status": int(hrd_status[item_idx].item()),
                        "pred_score": float(scores[item_idx]),
                        "pred_logit": float(logits[item_idx]),
                        "pred_prob": float(probs[item_idx]),
                        "pred_label_0p5": int(probs[item_idx] >= 0.5),
                        "attention_max": float(attn.max()),
                        "attention_entropy": attn_entropy,
                    }
                )
    return pd.DataFrame(rows)


def summarize_predictions(pred_df: pd.DataFrame) -> dict:
    pred_df = pred_df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    pred_df["rank_desc"] = np.arange(1, len(pred_df) + 1)
    positives = pred_df[pred_df["hrd_status"] == 1].copy()
    negatives = pred_df[pred_df["hrd_status"] == 0].copy()

    threshold = 0.5
    tp = int(((pred_df["hrd_status"] == 1) & (pred_df["pred_prob"] >= threshold)).sum())
    fn = int(((pred_df["hrd_status"] == 1) & (pred_df["pred_prob"] < threshold)).sum())
    fp = int(((pred_df["hrd_status"] == 0) & (pred_df["pred_prob"] >= threshold)).sum())
    tn = int(((pred_df["hrd_status"] == 0) & (pred_df["pred_prob"] < threshold)).sum())

    return {
        "num_slides": int(len(pred_df)),
        "num_positive": int((pred_df["hrd_status"] == 1).sum()),
        "num_negative": int((pred_df["hrd_status"] == 0).sum()),
        "auc": safe_auc(pred_df["hrd_status"].to_numpy(), pred_df["pred_prob"].to_numpy()),
        "ap": safe_ap(pred_df["hrd_status"].to_numpy(), pred_df["pred_prob"].to_numpy()),
        "pred_prob_mean": float(pred_df["pred_prob"].mean()),
        "positive_prob_mean": float(positives["pred_prob"].mean()) if not positives.empty else float("nan"),
        "negative_prob_mean": float(negatives["pred_prob"].mean()) if not negatives.empty else float("nan"),
        "positive_ranks_desc": positives["rank_desc"].astype(int).tolist(),
        "positive_probs": [float(x) for x in positives["pred_prob"].tolist()],
        "top_false_positives": negatives.head(5)[
            ["patient_barcode", "pred_prob", "rank_desc", "slide_path"]
        ].to_dict(orient="records"),
        "positive_cases": positives[
            ["patient_barcode", "pred_prob", "rank_desc", "slide_path"]
        ].to_dict(orient="records"),
        "threshold_0p5_confusion": {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        },
    }


def main() -> None:
    args = parse_args()
    ckpt_args = load_checkpoint_args(args.run_dir)
    set_seed(int(ckpt_args["seed"]))

    manifest_path = args.manifest or Path(ckpt_args["manifest"])
    manifest_df = pd.read_csv(manifest_path)
    output_dir = args.output_dir or (args.run_dir / "diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = WSIAttentionMIL(
        backbone_name=str(ckpt_args["backbone"]),
        pretrained=False,
    ).to(device)
    checkpoint = torch.load(args.run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    aggregate = {
        "run_dir": str(args.run_dir),
        "checkpoint_epoch": int(ckpt_args["_checkpoint_epoch"]),
        "selection_metric": ckpt_args["_selection_metric"],
        "fold": int(ckpt_args["fold"]),
        "task": str(ckpt_args["task"]),
        "backbone": str(ckpt_args["backbone"]),
        "splits": {},
    }

    for split_name in args.split:
        split_df = build_split_dataframe(manifest_df, ckpt_args, split_name)
        dataset = build_dataset(split_df, ckpt_args)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        pred_df = run_inference(model, loader, device)
        pred_df = pred_df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        pred_df["rank_desc"] = np.arange(1, len(pred_df) + 1)

        pred_path = output_dir / f"{split_name}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        aggregate["splits"][split_name] = summarize_predictions(pred_df)
        aggregate["splits"][split_name]["predictions_csv"] = str(pred_path)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, ensure_ascii=False)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
