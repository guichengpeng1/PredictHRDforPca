import argparse
import json
from pathlib import Path

import pandas as pd

from search_seed_ensembles import (
    evaluate_combo,
    load_seed_predictions,
    merge_predictions,
    safe_ap,
    safe_auc,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the best conditional ensemble from top-1/top-2/top-3 seed combinations."
    )
    parser.add_argument("--run-root", type=Path, required=True)
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
    parser.add_argument(
        "--candidate-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Allowed ensemble sizes evaluated from the top-ranked seeds.",
    )
    parser.add_argument("--focus-patient", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_seed_rankings(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise RuntimeError(f"Empty seed summary: {summary_csv}")
    return df.sort_values(["auc", "ap"], ascending=False).reset_index(drop=True)


def build_ensemble_predictions(
    merged_df: pd.DataFrame,
    selected_seeds: list[int],
) -> pd.DataFrame:
    prob_cols = [f"pred_prob_seed_{seed}" for seed in selected_seeds]
    ensemble_df = merged_df.copy()
    ensemble_df["ensemble_prob"] = ensemble_df[prob_cols].mean(axis=1)
    ensemble_df = ensemble_df.sort_values("ensemble_prob", ascending=False).reset_index(drop=True)
    ensemble_df["rank_desc"] = range(1, len(ensemble_df) + 1)
    return ensemble_df


def summarize_ensemble(df: pd.DataFrame, focus_patient: str | None) -> dict:
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
    output_dir = args.output_dir or (args.run_root / "conditional_ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.seed_summary_csv or (args.run_root / "seed_sweep_summary.csv")
    ranking_df = load_seed_rankings(summary_csv)
    seed_dfs = load_seed_predictions(args.run_root, args.predictions_file)
    merged_df = merge_predictions(seed_dfs)

    available_seeds = ranking_df["seed"].astype(int).tolist()
    combo_rows = []
    for size in sorted(set(int(x) for x in args.candidate_sizes)):
        if size < 1 or size > len(available_seeds):
            continue
        selected_seeds = available_seeds[:size]
        combo = tuple(selected_seeds)
        combo_row = evaluate_combo(merged_df, combo, args.focus_patient)
        combo_rows.append(combo_row)

    if not combo_rows:
        raise RuntimeError("No valid conditional ensemble candidates were evaluated.")

    candidates_df = pd.DataFrame(combo_rows).sort_values(["auc", "ap"], ascending=False).reset_index(drop=True)
    best_row = candidates_df.iloc[0].to_dict()
    best_seeds = [int(seed) for seed in str(best_row["combo"]).split(",")]
    selected_seed_rows = ranking_df[ranking_df["seed"].isin(best_seeds)].copy()
    selected_seed_rows["_selection_order"] = selected_seed_rows["seed"].apply(best_seeds.index)
    selected_seed_rows = selected_seed_rows.sort_values("_selection_order").drop(columns="_selection_order")

    ensemble_df = build_ensemble_predictions(merged_df, best_seeds)
    metrics = summarize_ensemble(ensemble_df, args.focus_patient)

    payload = {
        "run_root": str(args.run_root),
        "candidate_sizes": [int(x) for x in sorted(set(args.candidate_sizes))],
        "best_combo": best_row,
        "selected_seeds": best_seeds,
        "selected_seed_rows": selected_seed_rows.to_dict(orient="records"),
        "focus_patient": args.focus_patient,
        "metrics": metrics,
    }

    candidates_csv = output_dir / "candidate_summary.csv"
    candidates_df.to_csv(candidates_csv, index=False)
    payload["candidate_summary_csv"] = str(candidates_csv)

    ensemble_csv = output_dir / "ensemble_predictions.csv"
    ensemble_df.to_csv(ensemble_csv, index=False)
    payload["ensemble_predictions_csv"] = str(ensemble_csv)

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
