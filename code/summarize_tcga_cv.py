import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize TCGA CV runs from metrics.csv files.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more training output directories that contain metrics.csv.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_auc",
        choices=["val_auc", "val_ap", "val_mae"],
        help="Metric used to select the best epoch for each fold.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path to save the summary.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path to save the per-fold best rows.",
    )
    return parser.parse_args()


def best_row(metrics_df: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "val_mae":
        idx = metrics_df[metric].astype(float).idxmin()
    else:
        idx = metrics_df[metric].astype(float).idxmax()
    return metrics_df.loc[idx]


def expand_run_dirs(run_dirs: list[Path]) -> list[Path]:
    expanded = []
    seen = set()
    for run_dir in run_dirs:
        candidates = []
        if (run_dir / "metrics.csv").exists():
            candidates = [run_dir]
        else:
            candidates = sorted(
                path.parent for path in run_dir.glob("fold*/metrics.csv") if path.is_file()
            )
        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str in seen:
                continue
            seen.add(candidate_str)
            expanded.append(candidate)
    return expanded


def main() -> None:
    args = parse_args()
    rows = []
    for run_dir in expand_run_dirs(args.run_dir):
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            print(f"SKIP missing metrics: {metrics_path}")
            continue
        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.empty:
            print(f"SKIP empty metrics: {metrics_path}")
            continue
        row = best_row(metrics_df, args.metric).to_dict()
        row["run_dir"] = str(run_dir)
        rows.append(row)

    if not rows:
        raise RuntimeError("No valid runs found for summarization.")

    summary_df = pd.DataFrame(rows)
    numeric_cols = [
        "train_loss",
        "train_mae",
        "train_auc",
        "train_ap",
        "val_loss",
        "val_mae",
        "val_auc",
        "val_ap",
    ]
    for col in numeric_cols + ["epoch"]:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    aggregate = {
        "selection_metric": args.metric,
        "num_runs": int(len(summary_df)),
    }
    for col in numeric_cols:
        if col in summary_df.columns:
            aggregate[f"{col}_mean"] = float(summary_df[col].mean())
            aggregate[f"{col}_std"] = float(summary_df[col].std(ddof=0))

    print(summary_df.to_string(index=False))
    print()
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.output_csv, index=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
