import os
import pandas as pd
import matplotlib.pyplot as plt
import bioio
import bioio_tifffile
import numpy as np
from skimage import io, filters, feature, measure, morphology, segmentation
import re
import ast
from scipy import ndimage as ndi
from scipy.stats import norm
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

def detect_spots_watershed(
    image_path: str,
    spot_channel: int = 1,
    selem_radius: int = 4,
    min_peak_distance: int = 10,
    peak_min_intensity_factor: float = 0.30,
    min_volume: int = 50,
    max_volume: int = None,
    nuclei_mask_path: str = None,
    merge_gap: int = 1,
    save_outputs: bool = True
) -> tuple:
    """
    Detects spots using a watershed segmentation algorithm.

    This function binarizes the image, calculates a distance transform, and uses 
    watershed segmentation to identify and label individual spots. It includes 
    steps for filtering by size and intensity, and an optional step to merge 
    objects separated by a small gap.

    Parameters:
    - image_path: Path to the 3D TIFF image.
    - spot_channel: Channel index for spot detection.
    - selem_radius: Radius of the ball structuring element for morphological opening.
    - min_peak_distance: The minimum distance between peaks in the distance map.
    - peak_min_intensity_factor: Factor of the max distance to set the minimum peak intensity.
    - min_volume: Minimum volume of a detected spot.
    - max_volume: Maximum volume of a detected spot.
    - nuclei_mask_path: Optional path to a nuclei mask for final filtering.
    - merge_gap: The radius of the structuring element for merging close objects.
    - save_outputs: If True, saves the final mask and spot data to a 'results_spots' subdirectory.

    Returns:
    - A tuple containing:
        - df (pd.DataFrame): DataFrame with properties of detected spots.
        - image (np.ndarray): The original 3D image data.
        - distance_map (np.ndarray): The distance transform map used for segmentation.
        - final_labels (np.ndarray): The final labeled image of detected spots.
    """
    print(f"\n--- Running Watershed Spot Detection on: {os.path.basename(image_path)} ---")

    # 1. Load Image
    img_4d = bioio.BioImage(image_path, reader=bioio_tifffile.Reader)
    image = img_4d.get_image_data("ZYX", C=spot_channel)
    print(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")

    # 2. Binarization using Otsu's threshold
    threshold_value = filters.threshold_otsu(image)
    binary_image = image > threshold_value
    print(f"Otsu threshold: {threshold_value}")

    # 3. Morphological Opening to remove small noise
    selem = morphology.ball(selem_radius)
    binary_opened = morphology.binary_opening(binary_image, footprint=selem)

    # 4. Distance Transform and Watershed Marker Detection
    distance_map = ndi.distance_transform_edt(binary_opened)
    peak_min_intensity_abs = distance_map.max() * peak_min_intensity_factor

    local_maxi = feature.peak_local_max(
        distance_map,
        footprint=np.ones((min_peak_distance,) * 3),
        labels=binary_opened,
        threshold_abs=peak_min_intensity_abs
    )

    markers = np.zeros(image.shape, dtype=bool)
    if local_maxi.size > 0:
        markers[tuple(local_maxi.T)] = True
    labeled_markers, num_markers = ndi.label(markers)
    print(f"Found {num_markers} initial markers for watershed.")

    if num_markers == 0:
        print("No markers found. Returning empty results.")
        return pd.DataFrame(), image, distance_map, np.zeros_like(image, dtype=np.uint16)

    # 5. Watershed Segmentation
    segmented_labels = segmentation.watershed(
        -distance_map,
        labeled_markers,
        mask=binary_opened,
        watershed_line=True
    )

    unique_labels = np.unique(segmented_labels)
    label_map = np.zeros(segmented_labels.max() + 1, dtype=segmented_labels.dtype)
    valid_labels = []

    for label in unique_labels:
        if label == 0:
            continue # Skip background
        mask = (segmented_labels == label)
        if np.sum(mask) >= min_volume:
            label_map[label] = label # Keep this label
            valid_labels.append(label)
        # else: label_map[label] remains 0

    # Apply the mapping
    final_segmented_image = label_map[segmented_labels]

    # 6. Post-Watershed Merging of objects with small gaps
    if merge_gap > 0:
        merge_selem = morphology.ball(merge_gap)
        binary_from_labels = final_segmented_image > 0
        closed_binary = morphology.binary_closing(binary_from_labels, footprint=merge_selem)
        final_labels, num_merged = ndi.label(closed_binary)
        print(f"Relabeled segments after merging: {num_merged}")
    else:
        final_labels = final_segmented_image
        
    # 7. Measure properties and create DataFrame
    props = measure.regionprops_table(
        final_labels,
        intensity_image=image,
        properties=('label', 'centroid', 'area', 'max_intensity', 'mean_intensity', 'min_intensity')
    )

    # Convert to a DataFrame
    df = pd.DataFrame(props).rename(columns={
        'centroid-0': 'axis-0', 
        'centroid-1': 'axis-1', 
        'centroid-2': 'axis-2',
        'area': 'volume'
    })

    match = re.search(r'_t(\d{2})', os.path.basename(image_path))
    if match:
        timepoint = int(match.group(1))
    else:
        timepoint = 0
    df["timepoint"] = timepoint

    print(f"Initial detected spots: {len(df)}")

    # 8. Filtering

    # Volume filtering
    if max_volume is None:
        df = df[df['volume'] >= min_volume]
    elif min_volume is None:
        df = df[df['volume'] <= max_volume]
    else:
        df = df[df['volume'].between(min_volume, max_volume)]
    print(f"After volume filtering (min: {min_volume}, max: {max_volume}): {len(df)} spots")

    # Filter labels image to match the filtered DataFrame
    if not df.empty:
        kept_labels = df['label'].unique()
        label_map = np.zeros(final_labels.max() + 1, dtype=final_labels.dtype)
        label_map[kept_labels] = kept_labels
        final_labels = label_map[final_labels]
    else:
        # If all spots are filtered out, return an empty label image
        final_labels = np.zeros_like(final_labels)

    # 9. Nuclei Mask Filtering
    if nuclei_mask_path and os.path.exists(nuclei_mask_path):
        print("Applying nuclei mask for final filtering...")
        mask_4d = bioio.BioImage(nuclei_mask_path, reader=bioio_tifffile.Reader)
        nuclei_mask = mask_4d.get_image_data("ZYX", T=0, C=0).astype(bool)
        if nuclei_mask.shape == image.shape:
            z_coords = np.round(df['axis-0']).astype(int).clip(0, nuclei_mask.shape[0] - 1)
            y_coords = np.round(df['axis-1']).astype(int).clip(0, nuclei_mask.shape[1] - 1)
            x_coords = np.round(df['axis-2']).astype(int).clip(0, nuclei_mask.shape[2] - 1)
            
            in_nuclei = nuclei_mask[z_coords, y_coords, x_coords]
            df = df[in_nuclei].copy()
            print(f"After nuclei mask filtering: {len(df)} spots remaining")
        else:
            print("Warning: Nuclei mask shape does not match image shape. Skipping filtering.")

    return df, image, distance_map, final_labels

def analyze_domains_in_shape(image_path: str, 
                            shape_mask: np.ndarray, 
                            domain_min_size: int, 
                            output_dir: str, 
                            base_name: str,
                            spot_channel: int = 1):
    """
    Analyzes bright domains within larger segmented shapes (e.g., chromosomes).

    This function identifies bright, compact regions within each labeled shape
    using intensity thresholding and size filtering.

    Parameters:
    - image_path: The original 3D intensity image.
    - shape_mask: A 3D labeled image where each unique integer corresponds to a segmented shape.
    - domain_min_size: The minimum volume (in voxels) for a bright region to be considered a domain.
    - output_dir: The base directory to save the results.
    - base_name: The base name for the output CSV file.
    """

    print(f"\n--- Analyzing domains within shapes : {os.path.basename(image_path)} ---")

    # Load Image
    img_4d = bioio.BioImage(image_path, reader=bioio_tifffile.Reader)
    image = img_4d.get_image_data("ZYX", C=spot_channel)

    # Create a dedicated directory for domain results
    domain_output_dir = os.path.join(output_dir, 'domains')
    os.makedirs(domain_output_dir, exist_ok=True)
    
    all_domains = []
    
    # Iterate through each unique shape label in the mask
    for label in tqdm(np.unique(shape_mask)[1:], desc="Analyzing domains in shapes", disable=True):  # [1:] to skip background
        
        # Create a mask for the current shape
        current_shape_mask = (shape_mask == label)
        
        # Get the intensity values only from within the current shape
        intensities_in_shape = image[current_shape_mask]
        
        if intensities_in_shape.size == 0:
            continue
            
        # Use Otsu's method on the shape's intensities to find a threshold for "bright" domains
        try:
            domain_threshold = filters.threshold_otsu(intensities_in_shape)
        except ValueError:
            # This can happen if all intensity values in the mask are the same
            print(f"Skipping shape {label}: all intensity values are identical.")
            continue
            
        # Create a binary mask of potential domains within the current shape
        domain_binary_mask = (image > domain_threshold) & current_shape_mask
        
        # Label connected components (the individual domains)
        labeled_domains, num_domains = ndi.label(domain_binary_mask)
        
        if num_domains == 0:
            continue
            
        # Measure properties of each potential domain
        props = measure.regionprops(labeled_domains, intensity_image=image)
        
        for domain_prop in props:
            # Filter domains by the user-provided minimum size
            if domain_prop.area >= domain_min_size:
                all_domains.append({
                    'label': label,
                    'axis-0': domain_prop.centroid[0],
                    'axis-1': domain_prop.centroid[1],
                    'axis-2': domain_prop.centroid[2],
                    'volume': domain_prop.area,
                    'max_intensity': domain_prop.max_intensity,
                    'mean_intensity': domain_prop.mean_intensity,
                    'min_intensity': domain_prop.min_intensity
                })

    if not all_domains:
        print("No domains found meeting the size criteria.")
        return

    # Convert the list of domains to a DataFrame and save to CSV
    domains_df = pd.DataFrame(all_domains)
    domain_csv_path = os.path.join(domain_output_dir, f"{base_name}_domains.csv")
    domains_df.to_csv(domain_csv_path, index=False)
    print(f"Found {len(domains_df)} domains. Saved coordinates to: {domain_csv_path}")


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

    array_cols = ['zproj_spots', 'zproj_nuclei', 'intensity_dist', 'intensity_dist_spots']
    for col in array_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_array)

    return df

def count_spots_in_nuclei(spots_df, mask_path, nuclei_df):
    """Enhanced spot counting with physical coordinate conversion"""
    mask_img = bioio.BioImage(mask_path, reader=bioio_tifffile.Reader)
    mask = mask_img.get_image_data("ZYX", T=0, C=0)

    pixel_sizes = (1.0, 1.0, 1.0)
    
    spots_df = spots_df.copy()
    axis_labels = ['axis-0', 'axis-1', 'axis-2']
    
    for axis in axis_labels:
        ps = pixel_sizes[axis_labels.index(axis)]
        dim = mask.shape[axis_labels.index(axis)]
        ps_val = ps if ps is not None and ps > 0 else 1.0
        if axis not in spots_df.columns:
            raise ValueError(f"Missing required column: {axis}")
        spots_df[f'{axis}_idx'] = (
            spots_df[axis]
            .div(ps_val)
            .round()
            .astype(int)
            .clip(0, dim - 1)
        )
    
    mask = mask.astype(np.uint32)
    spots_df['nucleus_label'] = mask[
        spots_df['axis-0'].values,
        spots_df['axis-1'].values,
        spots_df['axis-2'].values
    ]
    
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
    if 'volume' in nuclei_df.columns:
        nuclei_df['spots_per_volume'] = nuclei_df['spot_count'] / (nuclei_df['volume'] + 1e-6)
    
    if all(col in nuclei_df.columns for col in ['major_axis_length', 'solidity', 'anisotropy']):
        a = nuclei_df['major_axis_length']/2
        b = a * nuclei_df['solidity']
        c = b * nuclei_df['anisotropy']
        nuclei_df['surface_area'] = 4 * np.pi * (((a*b)**1.6 + (a*c)**1.6 + (b*c)**1.6)/3)**(1/1.6)
    
    return nuclei_df


def analyze_spots(
    spots_df: pd.DataFrame,
    final_spot_mask: np.ndarray,
    nuclei_csv_path: str,
    mask_path: str,
    raw_image_path: str,
    output_dir: str,
    spot_channel: int = 1,
    domain_min_size: int = 50
):
    """
    Main analysis function to save spot detection results, count spots, measure intensities, and analyze domains.
    
    Parameters:
    - spots_df: DataFrame with detected spot properties (from detect_spots_watershed).
    - final_spot_mask: Labeled image of detected spots (from detect_spots_watershed).
    - nuclei_csv_path: Path to the CSV file containing nuclei data.
    - mask_path: Path to the nuclei mask file.
    - raw_image_path: Path to the original raw intensity image for analysis.
    - output_dir: Directory to save all output files.
    - spot_channel: The channel from the raw image to use for analysis.
    - domain_min_size: The minimum size for a bright region to be considered a domain. If 0, this step is skipped.
    """
    try:
        # base_name = os.path.splitext(os.path.basename(raw_image_path))[0]
        # os.makedirs(output_dir, exist_ok=True)
        
        # # --- Save the initial spot detection results ---
        # spots_results_dir = os.path.join(output_dir, 'results_spots')
        # os.makedirs(spots_results_dir, exist_ok=True)
        
        # spot_mask_path = os.path.join(spots_results_dir, f'{base_name}_labels.tif')
        # spot_csv_path = os.path.join(spots_results_dir, f'{base_name}.csv')

        # io.imsave(spot_mask_path, final_spot_mask.astype(np.uint16), check_contrast=False)
        # print(f"Saved final segmented spot labels to: {spot_mask_path}")

        # if not spots_df.empty:
        #     spots_df.to_csv(spot_csv_path, index=False)
        #     print(f"Saved spots data to: {spot_csv_path}")
        # else:
        #     pd.DataFrame().to_csv(spot_csv_path, index=False)
        #     print("No spots detected; saved an empty CSV file.")

        # --- Proceed with analysis using the in-memory spot data ---
        nuclei_df = load_nuclei_data(nuclei_csv_path)
        spots_mapped, nuclei_counts, mask = count_spots_in_nuclei(spots_df, mask_path, nuclei_df)
        
        nuclei_counts['xy_ratio'] = nuclei_counts.apply(lambda r: min(r['bb_dimX'], r['bb_dimY']) / max(r['bb_dimX'], r['bb_dimY']), axis=1)
        nuclei_counts['xz_ratio'] = nuclei_counts.apply(lambda r: min(r['bb_dimX'], r['bb_dimZ']) / max(r['bb_dimX'], r['bb_dimZ']), axis=1)
        nuclei_counts['yz_ratio'] = nuclei_counts.apply(lambda r: min(r['bb_dimY'], r['bb_dimZ']) / max(r['bb_dimY'], r['bb_dimZ']), axis=1)
        filtered_nuclei = nuclei_counts[(nuclei_counts['xy_ratio'] >= 0.2) & (nuclei_counts['xz_ratio'] >= 0.1) & (nuclei_counts['yz_ratio'] >= 0.1)].copy()
        
        nuclei_metrics = calculate_spatial_metrics(filtered_nuclei)
        
        print("\n--- Measuring Intensity Properties within Shapes ---")
        raw_img_4d = bioio.BioImage(raw_image_path, reader=bioio_tifffile.Reader)
        image = raw_img_4d.get_image_data("ZYX", C=spot_channel)

        intensity_props = measure.regionprops_table(mask, intensity_image=image, properties=('label', 'min_intensity', 'max_intensity', 'median_intensity'))
        intensity_df = pd.DataFrame(intensity_props)
        print(f"Calculated intensity stats for {len(intensity_df)} shapes.")

        nuclei_metrics = nuclei_metrics.merge(intensity_df, on='label', how='left')
        
        spots_output_path = os.path.join(output_dir, f"{base_name}_spots_mapped.csv")
        spots_mapped.to_csv(spots_output_path, index=False)
        print(f"Saved mapped spots to: {spots_output_path}")

        nuclei_metrics_path = os.path.join(output_dir, f"{base_name}_nuclei_metrics.csv")
        nuclei_metrics.to_csv(nuclei_metrics_path, index=False)
        print(f"Saved enhanced nuclei metrics to: {nuclei_metrics_path}")

        if domain_min_size > 0:
            print(f"\nProceeding with domain analysis (min size: {domain_min_size})...")
            analyze_domains_in_shape(image=image, shape_mask=mask, domain_min_size=domain_min_size, output_dir=output_dir, base_name=base_name)
        else:
            print("\nSkipping domain analysis as 'domain_min_size' is 0.")
            
        return spots_output_path, nuclei_metrics_path
            
    except Exception as e:
        print(f"Error in analyze_spots: {str(e)}")
        raise