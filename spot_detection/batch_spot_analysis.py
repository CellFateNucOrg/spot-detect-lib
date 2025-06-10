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



def process_single_image(image_file, input_dir, output_dir, nuclei_mask_dir, nuclei_csv_dir, spot_channel):
    image_path = os.path.join(input_dir, image_file)
    name_wo_ext = os.path.splitext(image_file)[0]

    # Adjust name
    if re.search(r'_n2v$', name_wo_ext) and not re.search(r'_t\d\d$', name_wo_ext):
        base_name = re.sub(r'_n2v$', '_t00', name_wo_ext)
    else:
        base_name = re.sub(r'n2v_', '', name_wo_ext)

    # Construct mask and CSV paths
    mask_filename = f"{base_name}.tif"
    mask_path = os.path.join(nuclei_mask_dir, mask_filename)
    csv_filename = f"{base_name}.csv"
    csv_path = os.path.join(nuclei_csv_dir, csv_filename)

    # Skip if files missing
    if not os.path.isfile(mask_path):
        return {'image': image_file, 'error': f"Missing mask: {mask_path}"}

    if not os.path.isfile(csv_path):
        return {'image': image_file, 'error': f"Missing nuclei CSV: {csv_path}"}

    # Detect spots
    # try:
    #     spots_df, _, _, _ = detect_spots_MPHD(
    #         image_path=image_path,
    #         spot_channel=spot_channel,
    #         sigma=1,
    #         h_percentile=75,
    #         peak_percentile=70,
    #         min_distance=3,
    #         min_volume=10,
    #         max_volume=100,
    #         intensity_percentile=85,
    #         nuclei_mask_path=mask_path
    #     )
    #high resolution
    try:
        spots_df, _, _, _ = detect_spots_MPHD(
            image_path=image_path,
            spot_channel=spot_channel,
            sigma=1.5,
            h_percentile=75,
            peak_percentile=70,
            min_distance=3,
            min_volume=10,
            max_volume=500,
            intensity_percentile=85,
            nuclei_mask_path=mask_path
        )

    except Exception as e:
        return {'image': image_file, 'error': f"Detection failed: {str(e)}"}

    # Save spots
    spots_output = os.path.join(output_dir, f"{base_name}_spots.csv")
    spots_df.to_csv(spots_output, index=False)

    # Analyze spots (optional)
    try:
        spots_mapped, _ = analyze_spots(spots_output, csv_path, mask_path, plot=True)
    except Exception as e:
        return {'image': image_file, 'error': f"Analysis failed: {str(e)}"}

    return {
        'image': image_file,
        'total_spots': len(spots_df),
        'nuclei_data_available': True,
        'error': None
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
