import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.batch_spot_analysis import batch_process_images, plot_summary_statistics
from utils.post_processing import analyze_nuclei_and_spots


if __name__ == "__main__":
    # Configure paths
    # BASE_DIR = Path("/mnt/external.data/MeisterLab/mvolosko/image_project/sdc1/SDC1/1273/20240813_e")
    # INPUT_DIR = Path('/mnt/external.data/MeisterLab/Dario/SDC1/1273/20240813_e/N2V_sdc1_dpy27_mSG_emr1_mCh/denoised/')
    BASE_DIR = Path('/mnt/external.data/MeisterLab/mvolosko/image_project/dpy27/DPY27/1268/20240808_e/')
    INPUT_DIR = Path("/mnt/external.data/MeisterLab/Dario/DPY27/1268/20240808_e/N2V_dpy27_mSG_emr1_mCh/denoised/")
    OUTPUT_DIR = BASE_DIR / "spots/"
    NUCLEI_MASK_DIR = BASE_DIR / "segmentation/"
    NUCLEI_CSV_DIR = BASE_DIR / "nuclei/"
    # Check if input directories exist
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory {INPUT_DIR} does not exist.")
    if not NUCLEI_MASK_DIR.exists():
        raise FileNotFoundError(f"Nuclei mask directory {NUCLEI_MASK_DIR} does not exist.")
    if not NUCLEI_CSV_DIR.exists():
        raise FileNotFoundError(f"Nuclei CSV directory {NUCLEI_CSV_DIR} does not exist.")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run batch processing
    combined_results = batch_process_images(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        nuclei_mask_dir=NUCLEI_MASK_DIR,
        nuclei_csv_dir=NUCLEI_CSV_DIR
    )

    # # Generate and save plots
    # if not combined_results.empty:
    #     plot_summary_statistics(combined_results)
    #     plot_nuclei_stats(combined_results, OUTPUT_DIR)
    #     print(f"Analysis complete! Results saved to {OUTPUT_DIR}")
    # else:
    #     print("No results generated - check input paths and data!")

    # Run nuclei+spot combination analysis
    analyze_nuclei_and_spots(
        nuclei_dir=NUCLEI_CSV_DIR,
        spots_dir=OUTPUT_DIR
    )
