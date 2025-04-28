import argparse
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from batch_spot_analysis import batch_process_images, plot_summary_statistics
from post_processing import analyze_nuclei_and_spots
import random
from visualization import render_nuclei_3d

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch spot analysis on denoised images + nuclei masks"
    )
    p.add_argument("--base-dir",        type=Path, required=True)
    p.add_argument("--input-dir",       type=Path, required=True)
    p.add_argument("--nuclei-mask-dir", type=Path, required=True)
    p.add_argument("--nuclei-csv-dir",  type=Path, required=True)
    p.add_argument("--output-dir",      type=Path, required=True)
    return p.parse_args()

def random_qc_visualization(spots_dir: Path,
                             nuclei_dir: Path,
                             mask_dir: Path,
                             qc_out: Path,
                             n_samples: int = 3):
    """
    Pick n_samples random spot CSVs from spots_dir, find the matching
    nuclei CSV and mask TIFF in nuclei_dir and mask_dir, and render each.
    """
    # collect all spot files
    spot_files = list(spots_dir.glob("*_spots.csv"))
    if not spot_files:
        print("No spot CSV files found for QC.")
        return

    # sample up to n_samples
    samples = random.sample(spot_files, min(n_samples, len(spot_files)))
    qc_out.mkdir(parents=True, exist_ok=True)

    for spot_fp in samples:
        # assume the base name before "_spots.csv" matches nuclei CSV & mask
        base = spot_fp.stem.replace("_spots", "")
        nuclei_fp = nuclei_dir / f"{base}.csv"
        # assume mask files end in .tif
        mask_fp = next(mask_dir.glob(f"{base}*.tif"), None)
        if not nuclei_fp.exists() or mask_fp is None:
            print(f"Skipping QC for {base}: missing CSV or mask")
            continue

        out_subdir = qc_out / base
        render_nuclei_3d(
            spots_path=str(spot_fp),
            mask_path=str(mask_fp),
            csv_path=str(nuclei_fp),
            output_path=str(out_subdir)
        )


if __name__ == "__main__":
    args = parse_args()

    BASE_DIR        = args.base_dir
    INPUT_DIR       = args.input_dir
    NUCLEI_MASK_DIR = args.nuclei_mask_dir
    NUCLEI_CSV_DIR  = args.nuclei_csv_dir
    OUTPUT_DIR      = args.output_dir

    # Check inputs
    for d in (INPUT_DIR, NUCLEI_MASK_DIR, NUCLEI_CSV_DIR):
        if not d.exists():
            raise FileNotFoundError(f"{d} does not exist")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Batch processing
    combined_results = batch_process_images(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        nuclei_mask_dir=NUCLEI_MASK_DIR,
        nuclei_csv_dir=NUCLEI_CSV_DIR,
    )

    # Final combine
    analyze_nuclei_and_spots(
        nuclei_dir=NUCLEI_CSV_DIR,
        spots_dir=OUTPUT_DIR
    )

        # after all analyses, do a quick 3‐sample QC:
    QC_OUTPUT = BASE_DIR / "qc_3d"
    random_qc_visualization(
        spots_dir=OUTPUT_DIR,
        nuclei_dir=NUCLEI_CSV_DIR,
        mask_dir=NUCLEI_MASK_DIR,
        qc_out=QC_OUTPUT,
        n_samples=3
    )
    print(f"QC 3D renders saved to {QC_OUTPUT}")

