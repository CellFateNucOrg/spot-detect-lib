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
    p.add_argument("--nuclei-mask-dir", type=Path, required=False)
    p.add_argument("--nuclei-csv-dir",  type=Path, required=False)
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

    # If you have a 5D TIFF file (TCZYX), then this function will split 
    # the file to 4D for each timepoint (CZYX)
    # it helps the algorithm to process the data more efficiently and avoid memory crash

    # split_5d_tiff_to_timepoints(
    #     input_path="/mnt/external.data/MeisterLab/mvolosko/image_project/high_res_images/1268/",
    #     output_dir="/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1268_fast_imaging_01/raw_images_timelapse_all/"
    # )

    # Main processing step
    batch_process_images(
        input_dir=INPUT_DIR ,
        output_dir=OUTPUT_DIR,
        nuclei_mask_dir=NUCLEI_MASK_DIR,
        nuclei_csv_dir=NUCLEI_CSV_DIR
    )

    # Final combine
    analyze_nuclei_and_spots(
        nuclei_dir=NUCLEI_CSV_DIR,
        spots_dir=OUTPUT_DIR,
        mask_dir=NUCLEI_MASK_DIR
    )

    # # #do a QC
    QC_OUTPUT = BASE_DIR / "qc_3d"
    QC_OUTPUT.mkdir(parents=True, exist_ok=True)

    batch_render_spots_z_slider(
        spot_mask_parent_dir=OUTPUT_DIR,
        mask_dir=NUCLEI_MASK_DIR,
        raw_img_dir=INPUT_DIR,
        domains_dir= OUTPUT_DIR / "domains", #or None if you don't want to detect them
        qc_out=QC_OUTPUT,
        n_samples=5, # how many qc images you need
        z_range=(10, 30), # middle 20 slides
    )
