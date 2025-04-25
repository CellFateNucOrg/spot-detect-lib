import numpy as np
import pandas as pd
import plotly.graph_objects as go
from skimage.measure import marching_cubes
from spot_analysis import load_nuclei_data, count_spots_in_nuclei
import os
from pathlib import Path
import bioio
import bioio_tifffile


def render_nuclei_3d(spots_path, mask_path, csv_path, output_path, step_size=2, level=0.5, z_range=None):
    """
    3D rendering with aspect ratio filtering
    """

    nuclei_df = load_nuclei_data(csv_path)
    spots_df = pd.read_csv(spots_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    spots_df, nuclei_df, mask = count_spots_in_nuclei(spots_df, mask_path, nuclei_df)
     
        # Calculate aspect ratios for each plane
    nuclei_df['xy_ratio'] = nuclei_df.apply(
        lambda row: min(row['bb_dimX'], row['bb_dimY']) / max(row['bb_dimX'], row['bb_dimY']), axis=1)
    nuclei_df['xz_ratio'] = nuclei_df.apply(
        lambda row: min(row['bb_dimX'], row['bb_dimZ']) / max(row['bb_dimX'], row['bb_dimZ']), axis=1)
    nuclei_df['yz_ratio'] = nuclei_df.apply(
        lambda row: min(row['bb_dimY'], row['bb_dimZ']) / max(row['bb_dimY'], row['bb_dimZ']), axis=1)
    
    
    # Filter nuclei based on aspect ratios first
    filtered_nuclei = nuclei_df[
        (nuclei_df['xy_ratio'] >= 0.5) &
        (nuclei_df['xz_ratio'] >= 0.2) &
        (nuclei_df['yz_ratio'] >= 0.2)
    ]
    
    mask = mask.astype(np.uint16)
    labels_to_plot = filtered_nuclei['label'].tolist()

    if z_range is not None:
        z_start, z_end = z_range
        mask = mask[z_start:z_end]
        spots_df = spots_df[(spots_df['z_idx'] >= z_start) & (spots_df['z_idx'] < z_end)].copy()
        spots_df['z_idx'] -= z_start

    print(f"Rendering {len(labels_to_plot)} filtered nuclei...")

    fig = go.Figure()

    for label in labels_to_plot:
        binary_mask = (mask == label).astype(np.uint8)
        if np.count_nonzero(binary_mask) == 0:
            continue
        
        try:
            verts, faces, normals, values = marching_cubes(
                binary_mask, 
                level=level, 
                step_size=step_size
            )
        except RuntimeError:
            continue

        z, y, x = verts.T
        i, j, k = faces.T

        # Get matching nucleus data for coloring
        nucleus_data = filtered_nuclei[filtered_nuclei['label'] == label].iloc[0]
        
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.3,
            color='orange',
            name=f'Nucleus {label} (Spots: {nucleus_data["spot_count"]})',
            text=[
                f"Volume: {nucleus_data['volume']}<br>"
                f"Spots: {nucleus_data['spot_count']}<br>"
                f"XY Ratio: {nucleus_data['xy_ratio']:.2f}<br>"
                f"XZ Ratio: {nucleus_data['xz_ratio']:.2f}"
            ],
            hoverinfo='text',
            flatshading=True
        ))

    # Plot spots with same filtering
    filtered_spots = spots_df[spots_df['nucleus_label'].isin(labels_to_plot)]
    fig.add_trace(go.Scatter3d(
        x=filtered_spots['x_idx'],
        y=filtered_spots['y_idx'],
        z=filtered_spots['z_idx'],
        mode='markers',
        marker=dict(
            size=5,
            color=filtered_spots['label'],
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(title='Nucleus ID')
        ),
        name='Spots',
        hovertext=filtered_spots['nucleus_label']
    ))

    fig.update_layout(
        title='3D Filtered Nuclei and Spots',
        scene=dict(
            xaxis_title='X', 
            yaxis_title='Y', 
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    fig.write_html(output_path / 'nuclei_3d_visualization_early.html')
    
    print(f"Figure saved to {output_path}")

# middle DPY27
# if __name__ == "__main__":
#     # Example usage
#     spots_path = "/mnt/external.data/MeisterLab/mvolosko/image_project/dpy27/DPY27/1268/20240808_e/spots/20240809_1268_E_mid_15um_04_n2v_spots.csv"
#     mask = '/mnt/external.data/MeisterLab/mvolosko/image_project/dpy27/DPY27/1268/20240808_e/segmentation/DPY27_e_20240809_1268_E_mid_15um_04_t00.tif'
#     csv_path = "/mnt/external.data/MeisterLab/mvolosko/image_project/dpy27/DPY27/1268/20240808_e/nuclei/DPY27_e_20240809_1268_E_mid_15um_04.csv"
#     output_path = '/mnt/external.data/MeisterLab/mvolosko/image_project/dpy27/DPY27/1268/20240808_e/3d_visualization/'  
#     render_nuclei_3d(spots_path, mask, csv_path, output_path)

# early DPY27
if __name__ == "__main__":
    # Example usage
    spots_path = "/mnt/external.data/MeisterLab/mvolosko/image_project/dpy27/DPY27/1268/20240808_e/spots/20240809_1268_E_early_15um_02_n2v_spots.csv"
    mask = '/mnt/external.data/MeisterLab/mvolosko/image_project/dpy27/DPY27/1268/20240808_e/segmentation/DPY27_e_20240809_1268_E_early_15um_02_t00.tif'
    csv_path = "/mnt/external.data/MeisterLab/mvolosko/image_project/dpy27/DPY27/1268/20240808_e/nuclei/DPY27_e_20240809_1268_E_early_15um_02.csv"
    output_path = '/mnt/external.data/MeisterLab/mvolosko/image_project/dpy27/DPY27/1268/20240808_e/3d_visualization/'  
    render_nuclei_3d(spots_path, mask, csv_path, output_path)

def plot_nuclei_stats(combined_df, output_dir):
    """Create and save nuclei statistics plots"""
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Spots per nucleus distribution
    plt.subplot(1, 2, 1)
    sns.histplot(combined_df['spot_count'], bins=30, kde=True)
    plt.title('Spots per Nucleus Distribution')
    plt.xlabel('Number of Spots')
    plt.ylabel('Count')

    # Plot 2: Nuclear volume vs spot count
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=combined_df, x='volume', y='spot_count', alpha=0.6)
    plt.title('Nuclear Volume vs Spot Count')
    plt.xlabel('Nuclear Volume (μm³)')
    plt.ylabel('Number of Spots')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nuclei_statistics.png'))
    plt.close()