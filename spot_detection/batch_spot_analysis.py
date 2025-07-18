import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bioio
import bioio_tifffile
from skimage import io
from pathlib import Path
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aicsimageio import AICSImage
from spot_analysis import (
    detect_spots_watershed,
    analyze_spots,
    load_nuclei_data,
    count_spots_in_nuclei,
    calculate_spatial_metrics
)
import tifffile


def split_5d_tiff_to_timepoints(input_path: Path, output_dir: Path) -> None:
    """
    Splits all 5D (T, C, Z, Y, X) TIFFs in a folder into OME-TIFFs per timepoint.
    Outputs to output_dir/raw_images_timelapse/{basename}_t{tt}.tif
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



def process_single_image(image_file, input_dir, output_dir, nuclei_mask_dir, 
                        nuclei_csv_dir, spot_channel):
    """
    Processes a single image: detects spots, saves results including 
    segmentation masks, and runs analysis.
    """
    image_path = os.path.join(input_dir, image_file)
    name_wo_ext = os.path.splitext(image_file)[0]

    # Adjust name
    if re.search(r'_n2v$', name_wo_ext) and not re.search(r'_t\d\d$', name_wo_ext):
        base_name = re.sub(r'_n2v$', '_t00', name_wo_ext)
    else:
        base_name = re.sub(r'n2v_', '', name_wo_ext)

    # Construct mask and CSV paths
    mask_path = os.path.join(nuclei_mask_dir, f"{base_name}.tif")
    csv_path = os.path.join(nuclei_csv_dir, f"{base_name}.csv")

    # Skip if files missing
    if not os.path.isfile(mask_path):
        return {'image': image_file, 'error': f"Missing mask: {mask_path}"}

    if not os.path.isfile(csv_path):
        return {'image': image_file, 'error': f"Missing nuclei CSV: {csv_path}"}

    # Detect spots
    try:
        spots_df, _, _, spot_labels = detect_spots_watershed(
            image_path=image_path,
            spot_channel=spot_channel,
            selem_radius=4,
            min_peak_distance=20,
            peak_min_intensity_factor=0.20,
            min_volume=100,
            max_volume=1000,
            intensity_percentile=85,
            nuclei_mask_path=mask_path,
            merge_gap=1,
            save_outputs=True
        )

    except Exception as e:
        return {'image': image_file, 'error': f"Detection failed: {str(e)}"}

    # Save spot segmentation mask
    spot_seg_dir = os.path.join(output_dir, 'spot_segmentation')
    os.makedirs(spot_seg_dir, exist_ok=True)
    spot_mask_output_path = os.path.join(spot_seg_dir, f"{base_name}_spot_mask.tif")
    tifffile.imwrite(
        spot_mask_output_path,
        spot_labels[np.newaxis, np.newaxis, ...].astype(np.uint16),
        photometric="minisblack",
        metadata={"axes": "ZYX"}
    )
    print(f"Saved spot segmentation mask to: {spot_mask_output_path}")

    # Run downstream analysis
    try:
        analyze_spots(
            spots_csv_path=spots_csv_output_path, 
            nuclei_csv_path=csv_path, 
            mask_path=mask_path, 
            output_dir=output_dir
        )

    except Exception as e:
        return {'image': image_file, 'error': f"Analysis failed: {str(e)}"}

    return {
        'image': image_file,
        'total_spots': len(spots_df),
        'nuclei_data_available': True,
        'error': None,
        'message': f'Successfully processed. Results in {output_dir}',
        'spot_mask_path': spot_mask_output_path
    }


def batch_process_images(input_dir, output_dir, nuclei_mask_dir, nuclei_csv_dir, spot_channel=1, max_workers=4):
    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.tif', '.tiff')) and not f.endswith('_max.tif')
    ]

    summary_data = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_image, image_file, input_dir, output_dir,
                nuclei_mask_dir, nuclei_csv_dir, spot_channel
            )
            for image_file in image_files
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images in parallel"):
            result = future.result()
            summary_data.append(result)
            if result.get('error'):
                print(f"Error for {result['image']}: {result['error']}")

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    return summary_df
