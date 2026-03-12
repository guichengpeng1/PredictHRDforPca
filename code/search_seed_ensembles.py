import argparse
import itertools
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
    parser = argparse.ArgumentParser(description="Search ensemble combinations across a seed sweep.")
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Root directory containing seed*/diagnostics_avg*/val_predictions_mean.csv",
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        default="diagnostics_avg10/val_predictions_mean.csv",
        help="Relative prediction csv path inside each seed directory.",
    )
    parser.add_argument(
        "--focus-patient",
        type=str,
        default=None,
        help="Optional patient barcode to track through ensemble combinations.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1,
        help="Minimum ensemble size to evaluate.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum ensemble size to evaluate. Defaults to all seeds.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def load_seed_predictions(run_root: Path, predictions_file: str) -> list[tuple[int, pd.DataFrame]]:
    loaded = []
    for config_path in sorted(run_root.glob("seed*/config.json")):
        run_dir = config_path.parent
        pred_path = run_dir / predictions_file
        if not pred_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        seed = int(config["seed"])
        pred_df = pd.read_csv(pred_path)
        pred_df = pred_df.rename(
            columns={
                "pred_prob_mean": f"pred_prob_seed_{seed}",
                "pred_prob_std": f"pred_prob_std_seed_{seed}",
            }
        )
        keep_cols = [
            "patient_barcode",
            "slide_path",
            "hrd_score",
            "hrd_status",
            f"pred_prob_seed_{seed}",
        ]
        if f"pred_prob_std_seed_{seed}" in pred_df.columns:
            keep_cols.append(f"pred_prob_std_seed_{seed}")
        loaded.append((seed, pred_df[keep_cols].copy()))
    return loaded


def merge_predictions(seed_dfs: list[tuple[int, pd.DataFrame]]) -> pd.DataFrame:
    base = None
    for _, seed_df in seed_dfs:
        if base is None:
            base = seed_df
            continue
        base = base.merge(
            seed_df,
            on=["patient_barcode", "slide_path", "hrd_score", "hrd_status"],
            how="inner",
        )
    if base is None:
        raise RuntimeError("No seed prediction files found.")
    return base


def evaluate_combo(
    merged_df: pd.DataFrame,
    combo: tuple[int, ...],
    focus_patient: str | None,
) -> dict:
    prob_cols = [f"pred_prob_seed_{seed}" for seed in combo]
    combo_df = merged_df[
        ["patient_barcode", "slide_path", "hrd_score", "hrd_status", *prob_cols]
    ].copy()
    combo_df["ensemble_prob"] = combo_df[prob_cols].mean(axis=1)
    combo_df = combo_df.sort_values("ensemble_prob", ascending=False).reset_index(drop=True)
    combo_df["rank_desc"] = np.arange(1, len(combo_df) + 1)
    positives = combo_df[combo_df["hrd_status"] == 1].copy()

    row = {
        "combo": ",".join(str(seed) for seed in combo),
        "ensemble_size": len(combo),
        "auc": safe_auc(combo_df["hrd_status"].to_numpy(), combo_df["ensemble_prob"].to_numpy()),
        "ap": safe_ap(combo_df["hrd_status"].to_numpy(), combo_df["ensemble_prob"].to_numpy()),
        "positive_ranks": ",".join(str(int(x)) for x in positives["rank_desc"].tolist()),
    }
    if focus_patient is not None:
        focus_df = combo_df[combo_df["patient_barcode"] == focus_patient]
        if not focus_df.empty:
            focus_row = focus_df.iloc[0]
            row["focus_patient"] = focus_patient
            row["focus_rank_desc"] = int(focus_row["rank_desc"])
            row["focus_prob"] = float(focus_row["ensemble_prob"])
    return row


def main() -> None:
    args = parse_args()
    seed_dfs = load_seed_predictions(args.run_root, args.predictions_file)
    if not seed_dfs:
        raise RuntimeError("No seed predictions were found.")
    merged_df = merge_predictions(seed_dfs)
    seeds = [seed for seed, _ in seed_dfs]

    max_size = args.max_size or len(seeds)
    rows = []
    for combo_size in range(args.min_size, max_size + 1):
        for combo in itertools.combinations(seeds, combo_size):
            rows.append(evaluate_combo(merged_df, combo, args.focus_patient))

    summary_df = pd.DataFrame(rows).sort_values(["auc", "ap"], ascending=False).reset_index(drop=True)
    aggregate = {
        "run_root": str(args.run_root),
        "num_seed_runs": len(seeds),
        "num_combinations": int(len(summary_df)),
        "best_combo_by_auc": summary_df.iloc[0].to_dict(),
    }
    best_ap_row = summary_df.sort_values(["ap", "auc"], ascending=False).iloc[0].to_dict()
    aggregate["best_combo_by_ap"] = best_ap_row

    print(summary_df.to_string(index=False))
    print()
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.output_csv, index=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
