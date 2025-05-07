import pandas as pd
import os
import glob
import re
import bioio
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_and_process_mask(mask_path):
    """Load segmentation mask with physical coordinates handling"""
    mask_img = bioio.BioImage(mask_path)
    mask = mask_img.get_image_data("ZYX", T=0, C=0).astype(np.uint32)
    
    try:
        pixel_sizes = mask_img.physical_pixel_sizes
    except AttributeError:
        pixel_sizes = (1.0, 1.0, 1.0)
        
    return mask, pixel_sizes

def convert_coordinates(df, pixel_sizes, mask_shape):
    """Convert physical coordinates to pixel indices"""
    axis_map = {'z': 0, 'y': 1, 'x': 2}
    coord_df = df.copy()
    
    for axis in ['z', 'y', 'x']:
        ps = pixel_sizes[axis_map[axis]] or 1.0
        dim = mask_shape[axis_map[axis]]
        coord_df[f'{axis}_idx'] = (coord_df[f'centroid_{axis}'] / ps).round().astype(int)
        coord_df[f'{axis}_idx'] = coord_df[f'{axis}_idx'].clip(0, dim - 1)
    
    return coord_df

def assign_nuclei_labels(spots_df, mask):
    """Vectorized label assignment using segmentation mask"""
    spots_df['nucleus_label'] = mask[
        spots_df['z_idx'].values,
        spots_df['y_idx'].values,
        spots_df['y_idx'].values  # This should be spots_df['y_idx'].values
    ]
    return spots_df

def analyze_nuclei_and_spots(nuclei_dir, spots_dir, mask_dir):
    results_dir = os.path.join(spots_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    combined_df = pd.DataFrame()

    nuclei_files = glob.glob(os.path.join(nuclei_dir, '*.csv'))
    
    for nuclei_path in nuclei_files:
        # File matching logic
        filename = os.path.basename(nuclei_path)
        base_id = re.sub(r'\.csv$', '', filename)
        
        # Load corresponding mask
        mask_path = os.path.join(mask_dir, f"{base_id}.tif")
        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            continue
            
        # Load and process mask
        mask, pixel_sizes = load_and_process_mask(mask_path)
        
        # Load nuclei data
        nuclei_df = pd.read_csv(nuclei_path)
        
        # Load spots data
        spot_pattern = os.path.join(spots_dir, f"{base_id}_spots.csv")
        spot_files = glob.glob(spot_pattern)
        if not spot_files:
            print(f"No spots found for {base_id}")
            continue
            
        spots_df = pd.concat([pd.read_csv(f) for f in spot_files], ignore_index=True)
        
        # Coordinate conversion
        spots_conv = convert_coordinates(spots_df, pixel_sizes, mask.shape)
        
        # Assign nuclei labels using mask
        spots_labeled = assign_nuclei_labels(spots_conv, mask)
        
        # Filter out background (label 0)
        valid_spots = spots_labeled[spots_labeled['nucleus_label'] > 0]
        
        # Aggregate spot data
        if not valid_spots.empty:
            agg_stats = valid_spots.groupby('nucleus_label').agg(
                spot_count=('label', 'size'),
                mean_intensity=('intensity', 'mean'),
                max_intensity=('intensity', 'max')
            ).reset_index()
        else:
            agg_stats = pd.DataFrame(columns=['nucleus_label', 'spot_count', 
                                            'mean_intensity', 'max_intensity'])
        
        # Merge with nuclei data
        merged_df = nuclei_df.merge(
            agg_stats,
            left_on='label',
            right_on='nucleus_label',
            how='left'
        ).fillna({
            'spot_count': 0,
            'mean_intensity': 0,
            'max_intensity': 0
        })
        
        # Add metadata
        match = re.search(r'_t(\d{2})_', base_id)
        if match:
            category = match.group(1)  # the 2 numbers
        else:
            category = None

        merged_df['category'] = category
        combined_df = pd.concat([combined_df, merged_df], ignore_index=True)

    # Save and plot results
    combined_csv_path = os.path.join(results_dir, 'combined_nuclei_spots.csv')
    combined_df.to_csv(combined_csv_path, index=False)
    # Plotting
    plot_path = lambda name: os.path.join(results_dir, name)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_df, x='category', y='spot_count')
    plt.title('Spots per nucleus across categories')
    plt.savefig(plot_path('boxplot_spots_per_nucleus.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=combined_df, x='volume', y='spot_count', hue='category')
    plt.title('Nucleus volume vs spot count')
    plt.savefig(plot_path('scatter_volume_vs_spotcount.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=combined_df, x='category', y='mean_intensity', estimator='mean')
    plt.title('Average spot intensity per category')
    plt.savefig(plot_path('barplot_mean_intensity.png'))
    plt.close()

    # Boxplot: add all traces
    fig1 = px.box(combined_df, x='category', y='spot_count', color='category',
                title='Spots per nucleus',
                points="all", template='plotly_white')

    # Scatter: fix overlapping with opacity + marker size
    fig2 = px.scatter(combined_df, x='volume', y='spot_count', color='category',
                    title='Nucleus volume vs spot count',
                    hover_data=['label', 'mean_intensity'],
                    opacity=0.6, size_max=10, template='plotly_white')

    # Barplot: as before
    fig3 = px.bar(combined_df, x='category', y='mean_intensity', color='category',
                title='Average spot intensity per category',
                barmode='group', template='plotly_white')

    # Combine into a dashboard
    dashboard = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Spots per nucleus", "Volume vs spot count", "Mean spot intensity"),
        specs=[[{"type": "box"}, {"type": "scatter"}],
            [{"colspan": 2}, None]]
    )

    # Add ALL boxplot traces
    for trace in fig1['data']:
        dashboard.add_trace(trace, row=1, col=1)

    # Add ALL scatterplot traces
    for trace in fig2['data']:
        dashboard.add_trace(trace, row=1, col=2)

    # Add ALL barplot traces
    for trace in fig3['data']:
        dashboard.add_trace(trace, row=2, col=1)

    # Final layout tweaks
    dashboard.update_layout(
        height=850,
        width=1300,
        title_text="Spots analysis Dashboard",
        template='plotly_white',
        showlegend=True
    )

    # Save and show
    dashboard.write_html(plot_path("interactive_dashboard.html"))
    print(f"Dashboard saved to: {results_dir}")