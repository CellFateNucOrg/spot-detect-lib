import os
import re
import time
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bioio
import bioio_tifffile
from pathlib import Path
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aicsimageio import AICSImage
from spot_analysis import (
    detect_spots_MPHD,
    analyze_spots,
    visualize_results,
    load_nuclei_data,
    count_spots_in_nuclei,
    calculate_spatial_metrics,
    visualize_distribution
)
import tifffile


def split_5d_tiff_to_timepoints(input_path: Path, output_dir: Path) -> None:
    """
    Splits all 5D (T, C, Z, Y, X) TIFFs in a folder into OME-TIFFs per timepoint.
    Outputs to output_dir/raw_images_timelapse/{basename}_t{tt}.ome.tif
    """
    from aicsimageio import AICSImage
    import tifffile
    import numpy as np
    from pathlib import Path

    input_dir = Path(input_path)
    raw_root = Path(output_dir) / "raw_images_timelapse"
    raw_root.mkdir(parents=True, exist_ok=True)

    ext = {".tif", ".tiff"}

    for img_path in sorted(input_dir.iterdir()):
        if img_path.suffix.lower() not in ext or img_path.stem.endswith('_max'):
            continue

        base_name = img_path.stem

        # lazy open
        img = AICSImage(str(img_path))
        assert img.dims.order == "TCZYX", f"Unexpected axes: {img.dims.order}"
        n_time, n_chan, n_z, n_y, n_x = img.shape

        for t in range(n_time):
            # grab channels+Z for this timepoint as (C,Z,Y,X)
            czyx = img.get_image_data("CZYX", T=t)

            czyx_5d = czyx[np.newaxis, ...] # shape = (1, C, Z, Y, X)

            # Write directly (no need to move axes now!)
            out_path = raw_root / f"{base_name}_t{t:02d}.tif"
            tifffile.imwrite(
                str(out_path),
                czyx_5d,
                photometric="minisblack",
                metadata={"axes": "TCZYX"},
                ome=True,
            )
            print(f"Saved OME-TIFF: {out_path}")



def process_single_image(image_path, nuclei_mask_path, nuclei_csv_path, output_dir, spot_channel=1):
    """Process a single image and save results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename for output files
    base_name = Path(image_path).stem
    
    # Run spot detection
    print(f"Processing {base_name}...")
    spots_df, img, hdome, labels = detect_spots_MPHD(
        image_path=image_path,
        spot_channel=spot_channel,
        sigma=1,
        h_percentile=55,
        peak_percentile=70,
        min_distance=1,
        min_volume=5,
        max_volume=100,
        intensity_percentile=80,
        nuclei_mask_path=nuclei_mask_path
    )
    
    # Analyze spots in nuclei
    spots_mapped, nuclei_analysis = analyze_spots(
        spots_df=spots_df,
        nuclei_csv_path=nuclei_csv_path,
        mask_path=nuclei_mask_path
    )
    
    # Save results
    spots_output = os.path.join(output_dir, f"{base_name}_spots.csv")
    nuclei_output = os.path.join(output_dir, f"{base_name}_nuclei.csv")
    
    spots_mapped.to_csv(spots_output, index=False)
    nuclei_analysis.to_csv(nuclei_output, index=False)
    
    return spots_mapped, nuclei_analysis, img


def batch_process_images(input_dir, output_dir, nuclei_mask_dir, nuclei_csv_dir, spot_channel=1):
    """Process all images in the input directory"""

    os.makedirs(output_dir, exist_ok=True)
    
    summary_data = []
    # Filter out *_max.tif* files and non-TIFF files
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.tif', '.tiff')) 
        and not os.path.splitext(f)[0].endswith('_max')
    ]

    for image_file in tqdm(image_files, desc="Processing images"):
        
        image_path = os.path.join(input_dir, image_file)
        base_name = re.sub(r'n2v_', '', os.path.splitext(image_file)[0])

        # Construct nuclei filenames based on naming convention
        nuclei_base = base_name
        
        # Verify if segmentation mask file exists
        mask_filename = f"{base_name}.tif"
        mask_path = os.path.join(nuclei_mask_dir, mask_filename)
        mask_exists = os.path.isfile(mask_path) 

        if not mask_exists:
            print(f"Segmentation mask directory is empty or not found: {mask_path}")
            continue
        
        # Verify if nuclei CSV file exists
        csv_filename = f"{nuclei_base}.csv"
        csv_path = os.path.join(nuclei_csv_dir, csv_filename)
        csv_exists = os.path.isfile(csv_path)
        if not csv_exists:
            print(f"Nuclei CSV file not found: {csv_path}")
            continue

        # Detect spots
        try:
            spots_df, _, _, _ = detect_spots_MPHD(
                image_path, 
                nuclei_mask_path=mask_path if mask_exists else None,
                spot_channel=spot_channel
            )
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
        
        # Save spots data
        spots_output = os.path.join(output_dir, f"{base_name}_spots.csv")
        spots_df.to_csv(spots_output, index=False)
        
        summary_entry = {
            'image': image_file,
            'total_spots': len(spots_df),
            'nuclei_data_available': mask_exists and csv_exists
        }
        
        # Analyze nuclei if data exists
        if mask_exists and csv_exists:
            try:
                nuclei_df = load_nuclei_data(csv_path)
                spots_mapped, nuclei_metrics = analyze_spots(
                    spots_output, csv_path, mask_path, plot=True
                )

                
                metrics_output = os.path.join(output_dir, f"{base_name}_nuclei_metrics.csv")
                nuclei_metrics.to_csv(metrics_output, index=False)
                
                summary_entry.update({
                    'mean_spots_per_nuclei': nuclei_metrics['spot_count'].mean(),
                    'median_spots_per_nuclei': nuclei_metrics['spot_count'].median(),
                    'total_nuclei': len(nuclei_metrics)
                })
            except Exception as e:
                print(f"Error in nuclei analysis for {image_file}: {e}")
        
        summary_data.append(summary_entry)
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    return summary_df



def plot_summary_statistics(combined_nuclei, output_dir):
    """Create summary plots for all processed images and save to file"""
    import os

    plt.figure(figsize=(15, 10))
    
    # Plot 1: Distribution of spots per nucleus across images
    plt.subplot(2, 2, 1)
    sns.boxplot(x='image', y='spot_count', data=combined_nuclei)
    plt.xticks(rotation=45)
    plt.title('Spots per Nucleus Distribution by Image')
    
    # Plot 2: Volume vs Spot Count
    plt.subplot(2, 2, 2)
    sns.scatterplot(x='volume', y='spot_count', data=combined_nuclei, alpha=0.5)
    plt.title('Volume vs Spot Count')
    
    # Plot 3: Spots per Volume Distribution
    plt.subplot(2, 2, 3)
    sns.histplot(combined_nuclei['spots_per_volume'], bins=30)
    plt.title('Distribution of Spots per Volume')
    
    # Plot 4: Aspect Ratio vs Spot Count
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='xy_ratio', y='spot_count', data=combined_nuclei, alpha=0.5)
    plt.title('Aspect Ratio vs Spot Count')
    
    plt.tight_layout()

    # Save the plot
    output_path = OUTPUT_PATH / "summary_statistics.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    
    # Optional: show if running interactively
    # plt.show()
    plt.close()

