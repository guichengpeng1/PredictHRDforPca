import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize per-fold top-k seed ensembles across a CV run.")
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--ensemble-subdir", type=str, default="top2_ensemble")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for summary_path in sorted(args.cv_root.glob(f"fold*/{args.ensemble_subdir}/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        row = {
            "fold_dir": str(summary_path.parent.parent),
            "top_k": int(payload["top_k"]),
            "selected_seeds": ",".join(str(x) for x in payload["selected_seeds"]),
            "auc": float(metrics["auc"]),
            "ap": float(metrics["ap"]),
            "positive_ranks_desc": ",".join(str(x) for x in metrics["positive_ranks_desc"]),
        }
        focus_case = metrics.get("focus_case")
        if focus_case is not None:
            row["focus_patient"] = focus_case["patient_barcode"]
            row["focus_rank_desc"] = int(focus_case["rank_desc"])
            row["focus_prob"] = float(focus_case["ensemble_prob"])
        rows.append(row)

    if not rows:
        raise RuntimeError("No per-fold top-k ensemble summaries were found.")

    df = pd.DataFrame(rows)
    aggregate = {
        "cv_root": str(args.cv_root),
        "num_folds": int(len(df)),
        "auc_mean": float(df["auc"].mean()),
        "auc_std": float(df["auc"].std(ddof=0)),
        "ap_mean": float(df["ap"].mean()),
        "ap_std": float(df["ap"].std(ddof=0)),
    }
    if "focus_rank_desc" in df.columns:
        aggregate["focus_rank_mean"] = float(df["focus_rank_desc"].mean())
        aggregate["focus_rank_std"] = float(df["focus_rank_desc"].std(ddof=0))

    print(df.to_string(index=False))
    print()
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
