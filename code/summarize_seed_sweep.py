import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize a fold seed sweep from diagnostic summaries.")
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Root directory containing seed*/ outputs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
    )
    parser.add_argument(
        "--diagnostics-subdir",
        type=str,
        default="diagnostics_avg10",
        help="Relative diagnostics subdir inside each seed run directory.",
    )
    parser.add_argument(
        "--focus-patient",
        type=str,
        default=None,
        help="Optional patient barcode to surface in the summary.",
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


def load_seed_row(run_dir: Path, diagnostics_subdir: str, split_name: str, focus_patient: str | None) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary_path = run_dir / diagnostics_subdir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split = summary["splits"][split_name]

    row = {
        "seed": int(config["seed"]),
        "run_dir": str(run_dir),
        "checkpoint_epoch": int(summary["checkpoint_epoch"]),
        "auc": float(split["auc"]),
        "ap": float(split["ap"]),
        "pred_prob_mean": float(split["pred_prob_mean"]),
        "positive_prob_mean": float(split["positive_prob_mean"]),
        "negative_prob_mean": float(split["negative_prob_mean"]),
    }
    if focus_patient is not None:
        focus_cases = [case for case in split.get("focus_cases", []) if case["patient_barcode"] == focus_patient]
        if focus_cases:
            case = focus_cases[0]
            row["focus_patient"] = focus_patient
            row["focus_rank_desc"] = int(case["rank_desc"])
            if "pred_prob_mean" in case:
                row["focus_pred_prob_mean"] = float(case["pred_prob_mean"])
            elif "pred_prob" in case:
                row["focus_pred_prob_mean"] = float(case["pred_prob"])
            if "pred_prob_std" in case:
                row["focus_pred_prob_std"] = float(case["pred_prob_std"])
    return row


def main() -> None:
    args = parse_args()
    run_dirs = sorted(path for path in args.run_root.glob("seed*/config.json"))
    rows = [
        load_seed_row(path.parent, args.diagnostics_subdir, args.split, args.focus_patient)
        for path in run_dirs
        if (path.parent / args.diagnostics_subdir / "summary.json").exists()
    ]
    if not rows:
        raise RuntimeError("No completed seed runs with diagnostic summaries were found.")

    summary_df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    aggregate = {
        "run_root": str(args.run_root),
        "split": args.split,
        "num_runs": int(len(summary_df)),
        "auc_best": float(summary_df["auc"].max()),
        "auc_mean": float(summary_df["auc"].mean()),
        "auc_std": float(summary_df["auc"].std(ddof=0)),
        "ap_best": float(summary_df["ap"].max()),
        "ap_mean": float(summary_df["ap"].mean()),
        "ap_std": float(summary_df["ap"].std(ddof=0)),
        "best_seed": int(summary_df.iloc[0]["seed"]),
    }
    if "focus_rank_desc" in summary_df.columns:
        aggregate["focus_patient"] = args.focus_patient
        aggregate["focus_rank_best"] = float(summary_df["focus_rank_desc"].min())
        aggregate["focus_rank_mean"] = float(summary_df["focus_rank_desc"].mean())
        aggregate["focus_rank_std"] = float(summary_df["focus_rank_desc"].std(ddof=0))
    if "focus_pred_prob_mean" in summary_df.columns:
        aggregate["focus_prob_best"] = float(summary_df["focus_pred_prob_mean"].max())
        aggregate["focus_prob_mean"] = float(summary_df["focus_pred_prob_mean"].mean())
        aggregate["focus_prob_std"] = float(summary_df["focus_pred_prob_mean"].std(ddof=0))

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
