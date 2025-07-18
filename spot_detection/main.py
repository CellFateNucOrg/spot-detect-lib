import argparse
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from batch_spot_analysis import batch_process_images, split_5d_tiff_to_timepoints

from post_processing import analyze_nuclei_and_spots
import random
from visualization import batch_qc_visualization, batch_render_spots_z_slider

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
    count_spots_dir = OUTPUT_DIR / "count_spots"
    count_spots_dir.mkdir(parents=True, exist_ok=True)

    # split_5d_tiff_to_timepoints(
    #     input_path=INPUT_DIR,
    #     output_dir=OUTPUT_DIR
    # )

    combined_results = batch_process_images(
        input_dir=OUTPUT_DIR / "raw_images_timelapse",
        #input_dir=INPUT_DIR ,
        output_dir=count_spots_dir,
        nuclei_mask_dir=NUCLEI_MASK_DIR,
        nuclei_csv_dir=NUCLEI_CSV_DIR
    )

    # Final combine
    analyze_nuclei_and_spots(
        nuclei_dir=NUCLEI_CSV_DIR,
        spots_dir=count_spots_dir,
        mask_dir=NUCLEI_MASK_DIR
    )

    # # #do a QC
    QC_OUTPUT = BASE_DIR / "qc_3d"
    QC_OUTPUT.mkdir(parents=True, exist_ok=True)

    batch_render_spots_z_slider(
        spots_dir=count_spots_dir,
        mask_dir=NUCLEI_MASK_DIR,
        raw_img_dir=OUTPUT_DIR / "raw_images_timelapse",
        #raw_img_dir=INPUT_DIR,
        domains_path= None,
        qc_out=QC_OUTPUT,
        n_samples=5,
        z_range=(10, 30),
    )
    print(f"QC 3D renders saved to {QC_OUTPUT}")


    # batch_qc_visualization(
    #     spots_dir=count_spots_dir,
    #     nuclei_dir=NUCLEI_CSV_DIR,
    #     mask_dir=NUCLEI_MASK_DIR,
    #     raw_img_dir=OUTPUT_DIR / "raw_images_timelapse",
    #     #raw_img_dir=INPUT_DIR,
    #     qc_out=QC_OUTPUT,
    #     n_samples=15,
    #     z_range=(15, 35)
    # )