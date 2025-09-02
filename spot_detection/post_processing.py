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

    for i, axis_name in zip(range(3), ['z', 'y', 'x']):
        ps = pixel_sizes[i] or 1.0
        dim = mask_shape[i]
        coord_df[f'{axis_name}_idx'] = (coord_df[f'axis-{i}'] / ps).round().astype(int)
        coord_df[f'{axis_name}_idx'] = coord_df[f'{axis_name}_idx'].clip(0, dim - 1)
    return coord_df

def assign_nuclei_labels(spots_df, mask):
    """label assignment using segmentation mask"""
    spots_df['nucleus_label'] = mask[
        spots_df['z_idx'].values,
        spots_df['y_idx'].values,
        spots_df['x_idx'].values 
    ]
    return spots_df

def analyze_nuclei_and_spots(nuclei_dir, spots_dir, mask_dir):
    """Main function to analyze nuclei and spots
    and generate summary statistics and visualizations."""

    results_dir = os.path.join(spots_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    combined_df = pd.DataFrame()

    nuclei_files = glob.glob(os.path.join(nuclei_dir, '*.csv'))
    
    for nuclei_path in nuclei_files:
        filename = os.path.basename(nuclei_path)
        base_id = re.sub(r'\.csv$', '', filename)
        
        # Load corresponding mask
        mask_path = os.path.join(mask_dir, f"{base_id}.tif")
        if not os.path.exists(mask_path):
            # Try alternative mask filename with _t00.tif
            alt_mask_path = os.path.join(mask_dir, f"{base_id}_t00.tif")
            if os.path.exists(alt_mask_path):
                mask_path = alt_mask_path
            else:
                print(f"Mask not found: {mask_path}")
            continue
            
        # Load and process mask
        mask, pixel_sizes = load_and_process_mask(mask_path)
        
        # Load nuclei data (ensure timepoint is integer)
        nuclei_df = pd.read_csv(nuclei_path)
        nuclei_df['timepoint'] = nuclei_df['timepoint'].astype(int)
        
        # Load spots data
        spot_pattern = os.path.join(spots_dir, 'spot_segmentation', f"{base_id}_spots.csv")
        spot_files = glob.glob(spot_pattern)
        if not spot_files:
            print(f"No spots found for {spot_pattern}")
            continue
            
        spots_df = pd.concat([pd.read_csv(f) for f in spot_files], ignore_index=True)
        
        # Coordinate conversion and label assignment
        spots_conv = convert_coordinates(spots_df, pixel_sizes, mask.shape)
        spots_labeled = assign_nuclei_labels(spots_conv, mask)
        valid_spots = spots_labeled[spots_labeled['nucleus_label'] > 0].copy()
        
        # Aggregate spot data
        if not valid_spots.empty:
            agg_stats = valid_spots.groupby('nucleus_label').agg(
                spot_count=('label', 'size'),
                mean_intensity=('mean_intensity', 'mean'),
                max_intensity=('max_intensity', 'max')
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
        
        combined_df = pd.concat([combined_df, merged_df], ignore_index=True)

    # timepoints sort
    combined_df = combined_df.sort_values('timepoint')
    combined_df['timepoint_str'] = combined_df['timepoint'].astype(str)  # For categorical plotting

    
    plot_path = lambda name: os.path.join(results_dir, name)

    # --- Plotting ---
    # Seaborn boxplot (numerical timepoints)
    if not combined_df.empty and combined_df['spot_count'].notnull().any():
        # Optional: filter timepoints that have at least some data
        non_empty_df = combined_df.groupby('timepoint').filter(lambda g: len(g) > 0)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=non_empty_df, x='timepoint', y='spot_count')
        plt.title('Spots per nucleus over time')
        plt.savefig(plot_path('boxplot_spots_over_time.png'))
        plt.close()
    else:
        print("Warning: No data available for Seaborn boxplot — skipping plot.")

    # Plotly dashboard
    fig1 = px.box(combined_df, x='timepoint_str', y='spot_count', 
                 color='timepoint_str', points="all",
                 title='Spots per nucleus over timepoints',
                 category_orders={"timepoint_str": sorted(combined_df['timepoint_str'].unique(), key=int)})

    fig2 = px.scatter(combined_df, x='volume', y='spot_count', 
                     color='timepoint_str',
                     title='Nucleus volume vs spot count',
                     opacity=0.6, size_max=10)

    fig3 = px.bar(combined_df, x='timepoint_str', y='mean_intensity',
                 color='timepoint_str',
                 title='Average spot intensity',
                 category_orders={"timepoint_str": sorted(combined_df['timepoint_str'].unique(), key=int)})

    # Create dashboard
    dashboard = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Spots per nucleus", "Volume vs spot count", "Mean spot intensity"),
        specs=[[{"type": "box"}, {"type": "scatter"}],
               [{"colspan": 2}, None]]
    )

    for trace in fig1['data']:
        dashboard.add_trace(trace, row=1, col=1)
        
    for trace in fig2['data']:
        dashboard.add_trace(trace, row=1, col=2)

    for trace in fig3['data']:
        dashboard.add_trace(trace, row=2, col=1)

    dashboard.update_layout(
        height=850,
        width=1300,
        title_text="Spots analysis dashboard",
        template='plotly_white',
        showlegend=False  # Optional: cleaner view if many timepoints
    )

    dashboard.write_html(plot_path("timecourse_dashboard.html"))
    print(f"Analysis saved to: {results_dir}")
