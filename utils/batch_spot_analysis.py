import os
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
from utils.spot_analysis import (
    detect_spots_MPHD,
    analyze_spots,
    visualize_results
)

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

def batch_process_images(input_dir, output_dir, 
                   nuclei_mask_dir, nuclei_csv_dir,
                   nuclei_prefix='SDC1_e_', mask_time_suffix='_t00'):
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
        base_name = os.path.splitext(image_file)[0]
        
        # Construct nuclei filenames based on naming convention
        nuclei_base = f"{nuclei_prefix}{base_name}"
        mask_filename = f"{nuclei_base}{mask_time_suffix}.tif"
        csv_filename = f"{nuclei_base}.csv"
        
        mask_path = os.path.join(nuclei_mask_dir, mask_filename)
        csv_path = os.path.join(nuclei_csv_dir, csv_filename)
        
        mask_exists = os.path.exists(mask_path)
        csv_exists = os.path.exists(csv_path)
        
        # Detect spots
        try:
            spots_df, _, _, _ = detect_spots_MPHD(
                image_path, 
                nuclei_mask_path=mask_path if mask_exists else None
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
                    spots_df, csv_path, mask_path, plot=False
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

def analyze_spots(spots_df, nuclei_csv_path, mask_path, plot=True):
    """Integrated analysis pipeline with optional plotting"""
    nuclei_df = load_nuclei_data(nuclei_csv_path)
    spots_mapped, nuclei_counts = count_spots_in_nuclei(spots_df, mask_path, nuclei_df)
    
    # Calculate aspect ratios and filter
    nuclei_counts['xy_ratio'] = nuclei_counts.apply(
        lambda r: min(r['bb_dimX'], r['bb_dimY']) / max(r['bb_dimX'], r['bb_dimY']), axis=1)
    nuclei_counts['xz_ratio'] = nuclei_counts.apply(
        lambda r: min(r['bb_dimX'], r['bb_dimZ']) / max(r['bb_dimX'], r['bb_dimZ']), axis=1)
    nuclei_counts['yz_ratio'] = nuclei_counts.apply(
        lambda r: min(r['bb_dimY'], r['bb_dimZ']) / max(r['bb_dimY'], r['bb_dimZ']), axis=1)
    
    filtered_nuclei = nuclei_counts[
        (nuclei_counts['xy_ratio'] >= 0.5) &
        (nuclei_counts['xz_ratio'] >= 0.2) &
        (nuclei_counts['yz_ratio'] >= 0.2)
    ].copy()
    
    nuclei_metrics = calculate_spatial_metrics(filtered_nuclei)
    
    if plot:
        visualize_distribution(nuclei_metrics)
    
    return spots_mapped, nuclei_metrics

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

