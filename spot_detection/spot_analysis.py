import os
import pandas as pd
import matplotlib.pyplot as plt
import bioio
import bioio_tifffile
import numpy as np
from skimage import filters, feature, measure, morphology, segmentation
import re
import ast
from scipy import ndimage
from scipy.stats import norm
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def detect_spots_MPHD(image_path, spot_channel=1, 
                     sigma=1.3, h_percentile=85,  
                     min_distance=2,              
                     min_volume=10,                
                     max_volume=100, 
                     intensity_percentile=80,     
                     peak_percentile=70,
                     nuclei_mask_path=None):         
    """
    MPHD spot detection with adaptive h-dome transformation.
    Parameters:
    - image_path: Path to the 3D TIFF image.
    - spot_channel: Channel index for spot detection.
    - sigma: Gaussian smoothing parameter.
    - h_percentile: Percentile for adaptive h value.
    - min_distance: Minimum distance between detected spots.
    - min_volume: Minimum volume of detected spots.
    - max_volume: Maximum volume of detected spots.
    - intensity_percentile: Percentile for intensity filtering.
    - peak_percentile: Percentile for peak thresholding.
    - nuclei_mask_path: Optional path to nuclei mask for final filtering
    """
    # Load image
    img_4d = bioio.BioImage(image_path, reader=bioio_tifffile.Reader)
    img_raw = img_4d.get_image_data("ZYX", C=spot_channel).astype(np.float32)
    
    # Normalize to [0,1] based on image statistics
    img = (img_raw - img_raw.min()) / (img_raw.max() - img_raw.min() + 1e-6)
    
    # Load nuclei mask if provided
    nuclei_mask = None
    if nuclei_mask_path is not None:
        mask_4d = bioio.BioImage(nuclei_mask_path, reader=bioio_tifffile.Reader)
        nuclei_mask = mask_4d.get_image_data("ZYX", T=0, C=0).astype(bool)
        if nuclei_mask.shape != img.shape:
            raise ValueError("Nuclei mask shape does not match image shape.")

    # Gaussian smoothing with 3D kernel
    smoothed = filters.gaussian(img, sigma=sigma)
    
    # Calculate adaptive h value
    non_zero = smoothed[smoothed > 0]
    if len(non_zero) == 0:
        raise ValueError("No signal detected after smoothing!")
        
    h = np.percentile(non_zero, h_percentile)
    
    # 3D h-dome transformation
    seed = np.clip(smoothed - h, 0, None)
    footprint = morphology.ball(5)
    reconstructed = morphology.reconstruction(
        seed=seed,
        mask=smoothed,
        method='dilation',
        footprint=footprint
    )

    hdome = smoothed - reconstructed
    
    # Adaptive peak threshold
    non_zero_hdome = hdome[hdome > 0]
    if len(non_zero_hdome) > 0:
        peak_thresh = np.percentile(non_zero_hdome, peak_percentile)
    else:
        raise ValueError("H-dome transformation failed - check h value!")
    
    # Detect regional maxima
    coordinates = feature.peak_local_max(
        hdome,
        min_distance=min_distance,
        threshold_abs=peak_thresh,
        exclude_border=False
    )

    # Create markers
    mask = np.zeros_like(hdome, dtype=bool)
    mask[tuple(coordinates.T)] = True
    markers = measure.label(mask)

    # Watershed segmentation
    if hdome.max() > 0:
        mask_thresh = hdome > filters.threshold_otsu(hdome)
    else:
        mask_thresh = np.zeros_like(hdome, dtype=bool)
    
    labels = segmentation.watershed(-hdome, markers, mask=mask_thresh)

    # Measure properties
    props = measure.regionprops_table(
        labels,
        intensity_image=img,
        properties=('label', 'centroid', 'area', 'max_intensity')
    )
    
    df = pd.DataFrame(props).rename(columns={
        'centroid-0': 'centroid_z', 'centroid-1': 'centroid_y', 'centroid-2': 'centroid_x',
        'area': 'volume',
        'max_intensity': 'intensity'
    })
    
    # Add peak coordinates
    peak_df = pd.DataFrame(coordinates, columns=['peak_z', 'peak_y', 'peak_x'])
    peak_df['label'] = markers[tuple(coordinates.T)]
    df = df.merge(peak_df, on='label', how='left')

    # Intensity filtering
    if len(img[img > 0]) > 0:
        intensity_thresh = np.percentile(img[img > 0], intensity_percentile)
        df = df[df['intensity'] > intensity_thresh]
    
    # Volume filtering
    df = df[df['volume'].between(min_volume, max_volume)]

    # Apply nuclei mask filtering * 0.8
    if nuclei_mask is not None:
        print("Applying nuclei mask filtering to final spots")
        # Convert float coordinates to integer indices
        z_coords = np.round(df['centroid_z']).astype(int).clip(0, nuclei_mask.shape[0]-1)
        y_coords = np.round(df['centroid_y']).astype(int).clip(0, nuclei_mask.shape[1]-1)
        x_coords = np.round(df['centroid_x']).astype(int).clip(0, nuclei_mask.shape[2]-1)
        
        # Check if spots are in masked regions
        in_nuclei = nuclei_mask[z_coords, y_coords, x_coords]
        df = df[in_nuclei]
        print(f"After nuclei filtering: {len(df)} spots remaining")

    return df, img, hdome, labels


def load_nuclei_data(nuclei_csv_path):
    """Load nuclei data with array columns parsing"""
    df = pd.read_csv(nuclei_csv_path)
    
    def parse_array(arr_str):
        try:
            cleaned = re.sub(r'[^\d\.,\[\]]', '', str(arr_str))
            cleaned = re.sub(r',+', ',', cleaned)
            return np.array(ast.literal_eval(cleaned))
        except (ValueError, SyntaxError):
            return np.array([])

    # Process array columns
    array_cols = ['zproj_spots', 'zproj_nuclei', 'intensity_dist', 'intensity_dist_spots']
    for col in array_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_array)


    return df

def count_spots_in_nuclei(spots_df, mask_path, nuclei_df):
    """Enhanced spot counting with physical coordinate conversion"""
    # Load mask with pixel size handling
    mask_img = bioio.BioImage(mask_path, reader=bioio_tifffile.Reader)
    mask = mask_img.get_image_data("ZYX", T=0, C=0)
    
    # Handle physical pixel sizes
    # try:
    #     pixel_sizes = mask_img.physical_pixel_sizes()
    # except AttributeError:
    pixel_sizes = (1.0, 1.0, 1.0)
    
    # Convert coordinates with pixel size adjustment
    spots_df = spots_df.copy()
    axis_map = {'z': 0, 'y': 1, 'x': 2}
    
    for axis in ['z', 'y', 'x']:
        ps = pixel_sizes[axis_map[axis]]
        dim = mask.shape[axis_map[axis]]
        ps_val = ps if ps is not None and ps > 0 else 1.0

        col_name = f'centroid_{axis}'
        if col_name not in spots_df.columns:
            raise ValueError(f"Missing required column: {col_name}")
            
        spots_df[f'{axis}_idx'] = (
            spots_df[f'centroid_{axis}']
            .div(ps_val)
            .round()
            .astype(int)
            .clip(0, dim - 1)
        )
    
    # Get labels for each spot
    mask = mask.astype(np.uint32)
    spots_df['nucleus_label'] = mask[
        spots_df['z_idx'].values,
        spots_df['y_idx'].values,
        spots_df['x_idx'].values
    ]
    
    # Ensure label column exists before merging
    if 'nucleus_label' not in spots_df.columns:
        raise RuntimeError("Failed to assign nucleus labels to spots")
    
    # Count spots per nucleus with proper merging
    spot_counts = (
        spots_df[spots_df['nucleus_label'] > 0]
        .groupby('nucleus_label', observed=True)
        .size()
        .rename('spot_count')
        .astype(np.uint32)
    )
    
    # Merge with nuclei data
    nuclei_df = nuclei_df.merge(
        spot_counts,
        left_on='label',
        right_index=True,
        how='left'
    ).fillna({'spot_count': 0})
    
    
    return spots_df, nuclei_df, mask

def calculate_spatial_metrics(nuclei_df):
    """Calculate advanced spatial metrics"""
    # 3D density metrics
    if 'volume' in nuclei_df.columns:
        nuclei_df['spots_per_volume'] = nuclei_df['spot_count'] / (nuclei_df['volume'] + 1e-6)
    
    # Surface area approximation
    if all(col in nuclei_df.columns for col in ['major_axis_length', 'solidity', 'anisotropy']):
        a = nuclei_df['major_axis_length']/2
        b = a * nuclei_df['solidity']
        c = b * nuclei_df['anisotropy']
        nuclei_df['surface_area'] = 4 * np.pi * (((a*b)**1.6 + (a*c)**1.6 + (b*c)**1.6)/3)**(1/1.6)
    
    return nuclei_df

def visualize_distribution(nuclei_df):
    """Enhanced visualization with multiple plots and normal distribution fit"""
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Volume vs Spot Count
    ax1 = fig.add_subplot(221)
    if 'volume' in nuclei_df.columns and 'spot_count' in nuclei_df.columns:
        color_data = nuclei_df['anisotropy'] if 'anisotropy' in nuclei_df.columns else 'blue'
        cmap = 'viridis' if 'anisotropy' in nuclei_df.columns else None
        scatter = ax1.scatter(
            nuclei_df['volume'],
            nuclei_df['spot_count'],
            c=color_data,
            cmap=cmap,
            alpha=0.6)
        ax1.set_xlabel('Nuclear Volume (pixels³)')
        ax1.set_ylabel('Spot count')
        # if 'anisotropy' in nuclei_df.columns:
        #     plt.colorbar(scatter, ax=ax1, label='Anisotropy')

    # Plot 2: Spots per Volume with Normal Fit
    ax2 = fig.add_subplot(222)
    if 'spots_per_volume' in nuclei_df.columns:
        spots_per_vol = nuclei_df['spots_per_volume'].dropna()
        if not spots_per_vol.empty:
            # Histogram with counts
            n, bins, patches = ax2.hist(spots_per_vol, bins=30, alpha=0.6, color='green')
            
            # Calculate normal distribution parameters
            mu, sigma = spots_per_vol.mean(), spots_per_vol.std()
            x = np.linspace(spots_per_vol.min(), spots_per_vol.max(), 100)
            
            # Scale normal curve to match histogram
            y = norm.pdf(x, mu, sigma) * len(spots_per_vol) * (bins[1] - bins[0])
            ax2.plot(x, y, 'r--', linewidth=2)
            ax2.set_xlabel('Spots per volume (spots/pixel³)')
            ax2.set_ylabel('Count')
            ax2.set_title(f'Normal distribution fit\nμ={mu:.2f}, σ={sigma:.2f}')

    # Plot 3: Spot Count Distribution
    ax3 = fig.add_subplot(223)
    if 'spot_count' in nuclei_df.columns:
        counts = nuclei_df['spot_count'].value_counts().sort_index()
        counts.plot(kind='bar', ax=ax3, color='teal')
        ax3.set_xlabel('Spot count per nucleus')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of spot counts')

    # Plot 4: Strain Comparison or Density Plot
    ax4 = fig.add_subplot(224)
    if 'strain' in nuclei_df.columns and 'spots_per_volume' in nuclei_df.columns:
        # Boxplot by strain
        strains = nuclei_df['strain'].unique()
        data = [nuclei_df[nuclei_df['strain'] == s]['spots_per_volume'].dropna() for s in strains]
        ax4.boxplot(data, labels=strains, showmeans=True)
        ax4.set_ylabel('Spots per Volume (spots/μm³)')
        ax4.set_title('Distribution by Strain')
    elif 'spots_per_volume' in nuclei_df.columns:
        # Density plot if no strain information
        spots_per_vol = nuclei_df['spots_per_volume'].dropna()
        if not spots_per_vol.empty:
            ax4.hist(spots_per_vol, bins=30, density=True, alpha=0.6, color='purple')
            ax4.set_xlabel('Spots per volume (spots/pixel³)')
            ax4.set_ylabel('Density')
            ax4.set_title('Probability density')

    plt.tight_layout()
    plt.show()

def visualize_results(img, df, hdome, labels):
    """Visualization for MPHD results"""
    max_proj = np.max(img, axis=0)
    hdome_proj = np.max(hdome, axis=0)
    # mid_z = img.shape[0] // 2
    
    # # Filter by PEAK Z-coordinate
    # df_mid = df[df['peak_z'] == mid_z]

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(max_proj, cmap='gray', vmax=np.percentile(max_proj, 99.5))
    plt.scatter(df['peak_x'], df['peak_y'], s=20, facecolors='none', 
                edgecolors='red', linewidth=0.5)
    plt.title(f"Detected spots: {len(df)}")
    
    plt.subplot(1, 2, 2)
    plt.imshow(hdome_proj, cmap='viridis')
    plt.title("H-Dome projection")

def analyze_spots(
    spots_csv_path,
    nuclei_csv_path,
    mask_path,
    output_dir=None,
    plot=False):
    """
    Analyze spots in nuclei
    
    Parameters:
    - spots_csv_path: Path to the CSV file containing spot coordinates
    - nuclei_csv_path: Path to the CSV file containing nuclei data
    - mask_path: Path to the nuclei mask file
    - output_dir: Directory to save output files (optional)
    - plot: Whether to generate and save plots (default: False)
    bioio_tifffile
    Returns:
    - Path to the mapped spots CSV file
    - Path to the nuclei metrics CSV file
    """
    try:
        # Load data from files
        spots_df = pd.read_csv(spots_csv_path)
        nuclei_df = load_nuclei_data(nuclei_csv_path)
        
        # Perform analysis
        spots_mapped, nuclei_counts, _ = count_spots_in_nuclei(spots_df, mask_path, nuclei_df)
        
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
        
        # Save results if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filenames
            base_name = os.path.splitext(os.path.basename(spots_csv_path))[0]
            spots_output = os.path.join(output_dir, f"{base_name}_mapped_spots.csv")
            metrics_output = os.path.join(output_dir, f"{base_name}_nuclei_metrics.csv")
            
            # Save results
            spots_mapped.to_csv(spots_output, index=False)
            nuclei_metrics.to_csv(metrics_output, index=False)
            
            # Generate and save plots if requested
            if plot:
                plot_output = os.path.join(output_dir, f"{base_name}_distribution.png")
                visualize_distribution(nuclei_metrics)
                plt.savefig(plot_output)
                plt.close()
            
            return spots_output, metrics_output
        else:
            return spots_mapped, nuclei_metrics
            
    except Exception as e:
        print(f"Error in analyze_spots: {str(e)}")
        raise 


