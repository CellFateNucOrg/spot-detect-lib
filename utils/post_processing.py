import pandas as pd
import os
import glob
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re


def analyze_nuclei_and_spots(nuclei_dir, spots_dir):
    results_dir = os.path.join(spots_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    combined_df = pd.DataFrame()

    # Process all nuclei files
    nuclei_files = glob.glob(os.path.join(nuclei_dir, '*.csv'))

    for nuclei_path in nuclei_files:

        filename = os.path.basename(nuclei_path)
        match = re.search(r'_e_(.+)\.csv$', filename)

        if match:
            base_id = match.group(1)
        else:
            raise ValueError(f"Filename '{filename}' does not match the expected pattern.")
            
        nuclei_df = pd.read_csv(nuclei_path)
        nuclei_df['min_z'] = nuclei_df['centroid_z'] - nuclei_df['bb_dimZ']/2
        nuclei_df['max_z'] = nuclei_df['centroid_z'] + nuclei_df['bb_dimZ']/2
        nuclei_df['min_y'] = nuclei_df['centroid_y'] - nuclei_df['bb_dimY']/2
        nuclei_df['max_y'] = nuclei_df['centroid_y'] + nuclei_df['bb_dimY']/2
        nuclei_df['min_x'] = nuclei_df['centroid_x'] - nuclei_df['bb_dimX']/2
        nuclei_df['max_x'] = nuclei_df['centroid_x'] + nuclei_df['bb_dimX']/2

        spot_path = os.path.join(spots_dir, f"{base_id}_n2v_spots.csv")
        if not os.path.exists(spot_path):
1 ЛИПНЯ 2025 Р. 490 БЕРН
￼
2/3
￼
￼
Ctrl+M

            print(f"No spots found for {base_id}")
            continue
        spots_df = pd.read_csv(spot_path)

        spots_assigned = []
        for _, spot in spots_df.iterrows():
            sz, sy, sx = spot[['centroid_z', 'centroid_y', 'centroid_x']]
            mask = (
                (nuclei_df['min_z'] <= sz) & (sz <= nuclei_df['max_z']) &
                (nuclei_df['min_y'] <= sy) & (sy <= nuclei_df['max_y']) &
                (nuclei_df['min_x'] <= sx) & (sx <= nuclei_df['max_x'])
            )
            if mask.any():
                nuclei_label = nuclei_df.loc[mask, 'label'].values[0]
                spots_assigned.append({**spot.to_dict(), 'nuclei_label': nuclei_label})

        spots_assigned_df = pd.DataFrame(spots_assigned)
        if not spots_assigned_df.empty:
            agg_stats = spots_assigned_df.groupby('nuclei_label').agg(
                spot_count=('label', 'size'),
                mean_intensity=('intensity', 'mean'),
                max_intensity=('intensity', 'max')
            ).reset_index()
            merged_df = nuclei_df.merge(agg_stats, left_on='label', right_on='nuclei_label', how='left')
        else:
            merged_df = nuclei_df.copy()
            merged_df[['spot_count', 'mean_intensity', 'max_intensity']] = 0

        merged_df['category'] = base_id.split('_')[3]
        combined_df = pd.concat([combined_df, merged_df], ignore_index=True)

    # Save merged data
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