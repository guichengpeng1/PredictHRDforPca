import argparse
import re
from pathlib import Path

import pandas as pd


SLIDE_SUFFIXES = {".svs", ".tif", ".tiff"}
DX_PATTERN = re.compile(r"DX(\d+)", re.IGNORECASE)


def parse_patient_barcode(slide_name: str) -> str:
    stem = Path(slide_name).stem
    parts = stem.split("-")
    if len(parts) < 3 or parts[0] != "TCGA":
        raise ValueError(f"Cannot parse TCGA patient barcode from: {slide_name}")
    return "-".join(parts[:3])


def parse_slide_barcode(slide_name: str) -> str:
    stem = Path(slide_name).stem
    parts = stem.split("-")
    if len(parts) < 4 or parts[0] != "TCGA":
        raise ValueError(f"Cannot parse TCGA slide barcode from: {slide_name}")
    return "-".join(parts[:4])


def parse_dx_rank(slide_name: str) -> int:
    match = DX_PATTERN.search(slide_name)
    if match is None:
        return 9999
    return int(match.group(1))


def build_manifest(
    slide_root: Path,
    clinical_csv: Path,
    output_csv: Path,
    include_all_slides: bool = False,
) -> pd.DataFrame:
    clinical_df = pd.read_csv(clinical_csv).copy()
    clinical_df["patient_barcode"] = clinical_df["Tumor_Sample_Barcode"].astype(str)
    clinical_df["hrd_status"] = (
        clinical_df["HRD_Binary"].astype(str).str.upper().eq("MUT").astype(int)
    )
    clinical_df["hrd_score"] = pd.to_numeric(
        clinical_df["HRD_Score"], errors="coerce"
    )
    clinical_df = clinical_df.dropna(subset=["patient_barcode", "hrd_score"])

    patient_to_clinical = clinical_df.set_index("patient_barcode").to_dict("index")
    rows = []
    for slide_path in sorted(slide_root.rglob("*")):
        if slide_path.suffix.lower() not in SLIDE_SUFFIXES:
            continue
        try:
            patient_barcode = parse_patient_barcode(slide_path.name)
            slide_barcode = parse_slide_barcode(slide_path.name)
        except ValueError:
            continue

        clinical_row = patient_to_clinical.get(patient_barcode)
        if clinical_row is None:
            continue

        rows.append(
            {
                "source": "TCGA-PRAD-SLIDE",
                "patient_barcode": patient_barcode,
                "slide_barcode": slide_barcode,
                "slide_name": slide_path.name,
                "slide_path": str(slide_path.resolve()),
                "dx_rank": parse_dx_rank(slide_path.name),
                "hrd_status": int(clinical_row["hrd_status"]),
                "hrd_binary": str(clinical_row["HRD_Binary"]),
                "hrd_score": float(clinical_row["hrd_score"]),
                "gleason_score": clinical_row.get("gleason_score"),
                "age": clinical_row.get("age"),
                "ajcc_pathologic_t": clinical_row.get("ajcc_pathologic_t"),
                "ajcc_pathologic_n": clinical_row.get("ajcc_pathologic_n"),
                "ajcc_pathologic_m": clinical_row.get("ajcc_pathologic_m"),
            }
        )

    manifest_df = pd.DataFrame(rows)
    if manifest_df.empty:
        raise RuntimeError(
            f"No overlapping labeled slides found under {slide_root} using {clinical_csv}"
        )

    manifest_df = manifest_df.sort_values(
        ["patient_barcode", "dx_rank", "slide_name"]
    ).reset_index(drop=True)
    if not include_all_slides:
        manifest_df = (
            manifest_df.groupby("patient_barcode", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_csv, index=False)
    return manifest_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a labeled TCGA-PRAD WSI manifest for HRD training."
    )
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
        "--output",
        type=Path,
        default=Path("outputs/tcga_wsi_manifest.csv"),
    )
    parser.add_argument(
        "--include-all-slides",
        action="store_true",
        help="Keep every slide per patient instead of selecting a single preferred slide.",
    )
    args = parser.parse_args()

    manifest_df = build_manifest(
        slide_root=args.slide_root,
        clinical_csv=args.clinical_csv,
        output_csv=args.output,
        include_all_slides=args.include_all_slides,
    )

    print(
        {
            "rows": int(len(manifest_df)),
            "patients": int(manifest_df["patient_barcode"].nunique()),
            "positives": int(manifest_df["hrd_status"].sum()),
            "negatives": int((1 - manifest_df["hrd_status"]).sum()),
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
