import argparse
import json
from pathlib import Path
from typing import Iterable

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
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        help="Override the base dataset seed used for deterministic tile sampling.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated deterministic bag samplings to average at inference time.",
    )
    parser.add_argument(
        "--focus-patient",
        type=str,
        nargs="*",
        default=[],
        help="Optional patient barcodes to highlight in the summary output.",
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


def build_dataset(split_df: pd.DataFrame, ckpt_args: dict, dataset_seed: int) -> WSITileBagDataset:
    return WSITileBagDataset(
        split_df,
        num_tiles=int(ckpt_args["num_tiles"]),
        tile_size=int(ckpt_args["tile_size"]),
        target_magnification=float(ckpt_args["target_mag"]),
        stride=int(ckpt_args["stride"]),
        transform=build_transforms(train=False, image_size=int(ckpt_args["tile_size"])),
        cache_dir=Path(ckpt_args["cache_dir"]),
        training=False,
        seed=int(dataset_seed),
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


def summarize_predictions(
    pred_df: pd.DataFrame,
    prob_col: str,
    prob_std_col: str | None = None,
    focus_patients: Iterable[str] | None = None,
) -> dict:
    pred_df = pred_df.sort_values(prob_col, ascending=False).reset_index(drop=True)
    pred_df["rank_desc"] = np.arange(1, len(pred_df) + 1)
    positives = pred_df[pred_df["hrd_status"] == 1].copy()
    negatives = pred_df[pred_df["hrd_status"] == 0].copy()
    focus_patients = list(focus_patients or [])

    threshold = 0.5
    tp = int(((pred_df["hrd_status"] == 1) & (pred_df[prob_col] >= threshold)).sum())
    fn = int(((pred_df["hrd_status"] == 1) & (pred_df[prob_col] < threshold)).sum())
    fp = int(((pred_df["hrd_status"] == 0) & (pred_df[prob_col] >= threshold)).sum())
    tn = int(((pred_df["hrd_status"] == 0) & (pred_df[prob_col] < threshold)).sum())

    focus_cols = ["patient_barcode", "slide_path", "rank_desc", prob_col]
    if prob_std_col is not None and prob_std_col in pred_df.columns:
        focus_cols.append(prob_std_col)
    focus_cases = []
    if focus_patients:
        focus_df = pred_df[pred_df["patient_barcode"].isin(focus_patients)][focus_cols].copy()
        for record in focus_df.to_dict(orient="records"):
            for key, value in list(record.items()):
                if isinstance(value, (np.floating, float)):
                    record[key] = float(value)
                if isinstance(value, (np.integer, int)):
                    record[key] = int(value)
            focus_cases.append(record)

    return {
        "num_slides": int(len(pred_df)),
        "num_positive": int((pred_df["hrd_status"] == 1).sum()),
        "num_negative": int((pred_df["hrd_status"] == 0).sum()),
        "auc": safe_auc(pred_df["hrd_status"].to_numpy(), pred_df[prob_col].to_numpy()),
        "ap": safe_ap(pred_df["hrd_status"].to_numpy(), pred_df[prob_col].to_numpy()),
        "pred_prob_mean": float(pred_df[prob_col].mean()),
        "positive_prob_mean": float(positives[prob_col].mean()) if not positives.empty else float("nan"),
        "negative_prob_mean": float(negatives[prob_col].mean()) if not negatives.empty else float("nan"),
        "positive_ranks_desc": positives["rank_desc"].astype(int).tolist(),
        "positive_probs": [float(x) for x in positives[prob_col].tolist()],
        "top_false_positives": negatives.head(5)[
            focus_cols
        ].to_dict(orient="records"),
        "positive_cases": positives[
            focus_cols
        ].to_dict(orient="records"),
        "focus_cases": focus_cases,
        "threshold_0p5_confusion": {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        },
    }


def aggregate_repeat_predictions(pred_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(pred_dfs, ignore_index=True)
    group_cols = ["patient_barcode", "slide_path", "hrd_score", "hrd_status"]
    agg_df = (
        combined.groupby(group_cols, as_index=False)
        .agg(
            pred_score_mean=("pred_score", "mean"),
            pred_score_std=("pred_score", "std"),
            pred_logit_mean=("pred_logit", "mean"),
            pred_logit_std=("pred_logit", "std"),
            pred_prob_mean=("pred_prob", "mean"),
            pred_prob_std=("pred_prob", "std"),
            attention_max_mean=("attention_max", "mean"),
            attention_max_std=("attention_max", "std"),
            attention_entropy_mean=("attention_entropy", "mean"),
            attention_entropy_std=("attention_entropy", "std"),
        )
        .sort_values("pred_prob_mean", ascending=False)
        .reset_index(drop=True)
    )
    std_cols = [col for col in agg_df.columns if col.endswith("_std")]
    agg_df[std_cols] = agg_df[std_cols].fillna(0.0)
    agg_df["rank_desc"] = np.arange(1, len(agg_df) + 1)
    agg_df["pred_label_0p5"] = (agg_df["pred_prob_mean"] >= 0.5).astype(int)
    return agg_df


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
        "dataset_seed_base": int(args.dataset_seed if args.dataset_seed is not None else ckpt_args["seed"]),
        "repeats": int(args.repeats),
        "focus_patients": list(args.focus_patient),
        "splits": {},
    }

    for split_name in args.split:
        split_df = build_split_dataframe(manifest_df, ckpt_args, split_name)
        dataset_seed_base = int(args.dataset_seed if args.dataset_seed is not None else ckpt_args["seed"])
        if args.repeats <= 1:
            dataset = build_dataset(split_df, ckpt_args, dataset_seed_base)
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

            aggregate["splits"][split_name] = summarize_predictions(
                pred_df,
                prob_col="pred_prob",
                focus_patients=args.focus_patient,
            )
            aggregate["splits"][split_name]["predictions_csv"] = str(pred_path)
            continue

        repeat_dfs = []
        for repeat_idx in range(args.repeats):
            repeat_seed = dataset_seed_base + repeat_idx
            dataset = build_dataset(split_df, ckpt_args, repeat_seed)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            repeat_df = run_inference(model, loader, device)
            repeat_df["repeat_index"] = repeat_idx
            repeat_df["repeat_seed"] = repeat_seed
            repeat_dfs.append(repeat_df)

        combined_df = pd.concat(repeat_dfs, ignore_index=True)
        combined_path = output_dir / f"{split_name}_predictions_repeats.csv"
        combined_df.to_csv(combined_path, index=False)

        agg_df = aggregate_repeat_predictions(repeat_dfs)
        agg_path = output_dir / f"{split_name}_predictions_mean.csv"
        agg_df.to_csv(agg_path, index=False)

        aggregate["splits"][split_name] = summarize_predictions(
            agg_df,
            prob_col="pred_prob_mean",
            prob_std_col="pred_prob_std",
            focus_patients=args.focus_patient,
        )
        aggregate["splits"][split_name]["repeats"] = int(args.repeats)
        aggregate["splits"][split_name]["repeat_seed_start"] = dataset_seed_base
        aggregate["splits"][split_name]["predictions_repeats_csv"] = str(combined_path)
        aggregate["splits"][split_name]["predictions_mean_csv"] = str(agg_path)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, ensure_ascii=False)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
