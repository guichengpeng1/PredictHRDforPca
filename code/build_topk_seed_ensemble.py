import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def parse_args():
    parser = argparse.ArgumentParser(description="Build a top-k seed ensemble from a completed seed sweep.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--predictions-file",
        type=str,
        default="diagnostics_avg10/val_predictions_mean.csv",
    )
    parser.add_argument(
        "--seed-summary-csv",
        type=Path,
        default=None,
        help="Optional override for the seed sweep summary csv.",
    )
    parser.add_argument("--focus-patient", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_seed_rankings(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise RuntimeError(f"Empty seed summary: {summary_csv}")
    return df.sort_values(["auc", "ap"], ascending=False).reset_index(drop=True)


def load_prediction_table(run_root: Path, seed: int, predictions_file: str) -> pd.DataFrame:
    pred_path = run_root / f"seed{seed}" / predictions_file
    pred_df = pd.read_csv(pred_path)
    return pred_df.rename(
        columns={
            "pred_prob_mean": f"pred_prob_seed_{seed}",
            "pred_prob_std": f"pred_prob_std_seed_{seed}",
        }
    )


def merge_prediction_tables(run_root: Path, seeds: list[int], predictions_file: str) -> pd.DataFrame:
    merged = None
    for seed in seeds:
        seed_df = load_prediction_table(run_root, seed, predictions_file)
        keep_cols = [
            "patient_barcode",
            "slide_path",
            "hrd_score",
            "hrd_status",
            f"pred_prob_seed_{seed}",
        ]
        std_col = f"pred_prob_std_seed_{seed}"
        if std_col in seed_df.columns:
            keep_cols.append(std_col)
        seed_df = seed_df[keep_cols].copy()
        if merged is None:
            merged = seed_df
        else:
            merged = merged.merge(
                seed_df,
                on=["patient_barcode", "slide_path", "hrd_score", "hrd_status"],
                how="inner",
            )
    if merged is None:
        raise RuntimeError("No prediction tables loaded for the requested seeds.")
    return merged


def summarize_ensemble(df: pd.DataFrame, focus_patient: str | None) -> dict:
    df = df.sort_values("ensemble_prob", ascending=False).reset_index(drop=True)
    df["rank_desc"] = np.arange(1, len(df) + 1)
    positives = df[df["hrd_status"] == 1].copy()
    negatives = df[df["hrd_status"] == 0].copy()
    tp = int(((df["hrd_status"] == 1) & (df["ensemble_prob"] >= 0.5)).sum())
    fn = int(((df["hrd_status"] == 1) & (df["ensemble_prob"] < 0.5)).sum())
    fp = int(((df["hrd_status"] == 0) & (df["ensemble_prob"] >= 0.5)).sum())
    tn = int(((df["hrd_status"] == 0) & (df["ensemble_prob"] < 0.5)).sum())

    summary = {
        "num_slides": int(len(df)),
        "num_positive": int((df["hrd_status"] == 1).sum()),
        "num_negative": int((df["hrd_status"] == 0).sum()),
        "auc": safe_auc(df["hrd_status"].to_numpy(), df["ensemble_prob"].to_numpy()),
        "ap": safe_ap(df["hrd_status"].to_numpy(), df["ensemble_prob"].to_numpy()),
        "positive_ranks_desc": positives["rank_desc"].astype(int).tolist(),
        "positive_cases": positives[
            ["patient_barcode", "slide_path", "rank_desc", "ensemble_prob"]
        ].to_dict(orient="records"),
        "top_false_positives": negatives.head(5)[
            ["patient_barcode", "slide_path", "rank_desc", "ensemble_prob"]
        ].to_dict(orient="records"),
        "threshold_0p5_confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }
    if focus_patient:
        focus = df[df["patient_barcode"] == focus_patient]
        if not focus.empty:
            focus_row = focus.iloc[0]
            summary["focus_case"] = {
                "patient_barcode": focus_patient,
                "rank_desc": int(focus_row["rank_desc"]),
                "ensemble_prob": float(focus_row["ensemble_prob"]),
            }
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.run_root / f"top{args.top_k}_ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.seed_summary_csv or (args.run_root / "seed_sweep_summary.csv")
    ranking_df = load_seed_rankings(summary_csv)
    selected = ranking_df.head(args.top_k).copy()
    selected_seeds = [int(seed) for seed in selected["seed"].tolist()]

    merged = merge_prediction_tables(args.run_root, selected_seeds, args.predictions_file)
    prob_cols = [f"pred_prob_seed_{seed}" for seed in selected_seeds]
    merged["ensemble_prob"] = merged[prob_cols].mean(axis=1)
    ensemble_df = merged.sort_values("ensemble_prob", ascending=False).reset_index(drop=True)
    ensemble_df["rank_desc"] = np.arange(1, len(ensemble_df) + 1)

    summary = {
        "run_root": str(args.run_root),
        "top_k": int(args.top_k),
        "selected_seeds": selected_seeds,
        "selected_seed_rows": selected.to_dict(orient="records"),
        "focus_patient": args.focus_patient,
        "metrics": summarize_ensemble(ensemble_df, args.focus_patient),
    }

    ensemble_csv = output_dir / "ensemble_predictions.csv"
    ensemble_df.to_csv(ensemble_csv, index=False)
    summary["ensemble_predictions_csv"] = str(ensemble_csv)

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
