import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tifffile import imread
from scipy.ndimage import zoom

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from skimage.measure import marching_cubes, find_contours, block_reduce
from skimage.transform import downscale_local_mean

from matplotlib import cm
from matplotlib.colors import Normalize, to_hex

from aicsimageio import AICSImage

from spot_analysis import load_nuclei_data, count_spots_in_nuclei

def render_nuclei_3d(spots_path,
                     original_img_path,
                     mask_path,
                     csv_path,
                     output_path,
                     step_size=2,
                     level=0.5,
                     z_range=None,
                     base_id=None):
    """
    3D rendering with aspect-ratio filtering and
    interactive z-stack explorer (no contours).
    """
    # --- load & filter ----------------------------------------
    nuclei_df = load_nuclei_data(csv_path)
    spots_df = pd.read_csv(spots_path)
    spots_df, nuclei_df, mask = count_spots_in_nuclei(
        spots_df, mask_path, nuclei_df
    )
    # compute aspect ratios
    for name, (d0, d1) in [
        ('xy_ratio', ('bb_dimX','bb_dimY')),
        ('xz_ratio', ('bb_dimX','bb_dimZ')),
        ('yz_ratio', ('bb_dimY','bb_dimZ'))
    ]:
        nuclei_df[name] = nuclei_df.apply(
            lambda r: min(r[d0], r[d1]) / max(r[d0], r[d1]), axis=1
        )
    filtered = nuclei_df.query(
        "xy_ratio>=0.3 & xz_ratio>=0.1 & yz_ratio>=0.1"
    )
    labels = filtered['label'].tolist()

    # optional z-cropping
    if z_range is not None:
        z0, z1 = z_range
        mask = mask[z0:z1]
        spots_df = (
            spots_df
            .query("@z0 <= z_idx < @z1")
            .assign(z_idx=lambda df: df.z_idx - z0)
        )

    print(f"Rendering {len(labels)} filtered nuclei…")

    # --- set up subplots --------------------------------------
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type':'xy'}, {'type':'scene'}]],
        subplot_titles=("Z-slice explorer", "3D Reconstruction")
    )

    # load raw ZYX stack
    raw = AICSImage(original_img_path).get_image_data("ZYX", T=0, C=1)
    n_slices = raw.shape[0]
    mid = n_slices // 2

    # add one heatmap per z (no contours)
    for z in range(n_slices):
        fig.add_trace(
            go.Heatmap(
                z=raw[z],
                colorscale='Gray',
                showscale=False,
                visible=(z == mid)
            ),
            row=1, col=1
        )

    # --- 3D meshes & spots -------------------------------------
    for lbl in labels:
        bm = (mask == lbl).astype(np.uint8)
        if not bm.any():
            continue
        try:
            verts, faces, *_ = marching_cubes(
                bm, level=level, step_size=step_size
            )
        except RuntimeError:
            print(f"Skipping label {lbl} due to marching_cubes failure.")
            continue

        z, y, x = verts.T
        i, j, k = faces.T
        meta = filtered.query("label==@lbl").iloc[0]

        fig.add_trace(
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=0.3,
                color='orange',
                name=f"Nuc {lbl} (spots {meta.spot_count})",
                text=[
                    f"Vol: {meta.volume}<br>"
                    f"XY: {meta.xy_ratio:.2f}<br>"
                    f"XZ: {meta.xz_ratio:.2f}"
                ],
                hoverinfo='text',
                flatshading=True
            ),
            row=1, col=2
        )

    # spots cloud
    sp = spots_df[spots_df.nucleus_label.isin(labels)]
    fig.add_trace(
        go.Scatter3d(
            x=sp.x_idx, y=sp.y_idx, z=sp.z_idx,
            mode='markers',
            marker=dict(
                size=4,
                color=sp.label,
                colorscale='Viridis',
                opacity=0.6
            ),
            name='Spots'
        ),
        row=1, col=2
    )

    # --- build slider ------------------------------------------
    # first n_slices traces are heatmaps,
    # the rest are the 3D and spots (always visible)
    base_traces = n_slices
    static_traces = len(fig.data) - base_traces

    steps = []
    for z in range(n_slices):
        vis = [i == z for i in range(base_traces)]
        vis += [True] * static_traces
        steps.append(dict(
            method='restyle',
            args=['visible', vis],
            label=str(z)
        ))

    fig.update_layout(
        sliders=[dict(active=mid, pad={'t': 30}, steps=steps)],
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600,
        width=1200,
        title=f"QC 3D – {base_id or ''}"
    )

    # --- write HTML -------------------------------------
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    fname = f"{base_id}_3d_qc.html" if base_id else "nuclei_3d_qc.html"
    fig.write_html(
        output_path / fname,
        include_plotlyjs='cdn',
        full_html=False
    )
    print("Saved →", output_path / fname)

    return fig



def batch_qc_visualization(
    spots_dir: Path,
    nuclei_dir: Path,
    mask_dir: Path,
    raw_img_dir: Path,
    qc_out: Path,
    z_range: tuple = None,
    n_samples: int = None
): 
    """
    For each spots CSV in spots_dir (or a random sample of them),
    find the matching nuclei CSV, mask TIFF, and raw `_n2v_t` TIFF.
    If all exist, call render_nuclei_3d(..., base_id=base).
    """
    qc_out.mkdir(parents=True, exist_ok=True)
    spot_files = list(spots_dir.glob("*_spots.csv"))
    if not spot_files:
        print("No spot CSV files found.")
        return

    # optionally subsample
    if n_samples:
        spot_files = random.sample(spot_files, min(n_samples, len(spot_files)))

    for spot_fp in spot_files:
        base = spot_fp.stem.replace("_spots", "")
        nuclei_fp = nuclei_dir / f"{base}.csv"

        # mask: could be one or more; pick first
        mask_fp_candidates = list(mask_dir.glob(f"{base}.tif"))
        mask_fp = mask_fp_candidates[0] if mask_fp_candidates else None

        # raw image: modify base to include _n2v_ before _t and look for matching files
        base_with_n2v = base.replace("_t", "_n2v_t")
        # Try with _n2v_t first
        raw_fp_candidates = list(raw_img_dir.glob(f"{base_with_n2v}*.tif"))
        if not raw_fp_candidates:
            # Fallback: look for _n2v*.tif without _t suffix
            base_no_t = base.split("_t")[0]
            raw_fp_candidates = list(raw_img_dir.glob(f"{base_no_t}_n2v.tif"))
        raw_fp = raw_fp_candidates[0] if raw_fp_candidates else None

        missing = [name for p, name in [(nuclei_fp, "nuclei CSV"),
                                (mask_fp, "mask TIFF"),
                                (raw_fp, "raw TIFF")] if p is None or not Path(p).exists()]

        if missing:
            print(f"Skipping {base}: missing {', '.join(missing)}")
            continue

        # Everything’s there – call your renderer
        render_nuclei_3d(
            spots_path=str(spot_fp),
            original_img_path=str(raw_fp),
            mask_path=str(mask_fp),
            csv_path=str(nuclei_fp),
            output_path=str(qc_out),
            base_id=base
        )
        print(f" Rendered QC for {base}")

norm = Normalize(vmin=1, vmax=5)
cmap = cm.get_cmap("plasma")  # vibrant colormap for spot contours

def distance_color(d):
    return to_hex(cmap(norm(d)))


def render_spots_z_slider(
    spots_path: str,
    original_img_path: str,
    mask_path: str,
    domains_path: str = None,
    z_range: tuple = None,
    output_path: str = None,
    base_id: str = None
):
    """
    Generates a 2D z-slice explorer with spots, domains, and nuclei outlines.

    This function creates an interactive HTML file with a slider to navigate through
    z-slices, overlaying spots and domains from neighboring slices. It also saves
    static maximum projection and middle-slice images.

    Parameters:
    - spots_path: Path to the spot coordinates CSV file.
    - original_img_path: Path to the raw 3D image.
    - mask_path: Path to the 3D nuclei/shape mask.
    - domains_path: (Optional) Path to the domain coordinates CSV file.
    - z_range: (Optional) A tuple (start, end) to crop the Z-axis.
    - output_path: Directory to save the HTML and PNG outputs.
    - base_id: A unique identifier for the output files.
    """
    print(f"--- Rendering visualization for {base_id} ---")
    # --- Load Data ---
    spots_df = pd.read_csv(spots_path)
    spots_df['axis-0'] = spots_df['axis-0'].round().astype(int)

    raw = AICSImage(original_img_path).get_image_data("ZYX", T=0, C=1)
    raw = np.squeeze(raw).astype(np.float32)
    mask = AICSImage(mask_path).get_image_data("ZYX", T=0, C=0)
    mask = np.squeeze(mask).astype(np.uint8)

    # --- Load Domain Data (if provided) ---
    domains_df = None
    if domains_path and Path(domains_path).exists():
        try:
            domains_df = pd.read_csv(domains_path)
            if not domains_df.empty:
                domains_df['axis-0'] = domains_df['axis-0'].round().astype(int)
                print(f"Loaded {len(domains_df)} domains.")
            else:
                domains_df = None # Treat empty file as no domains
        except Exception as e:
            print(f"Could not load or process domains file {domains_path}: {e}")
            domains_df = None

    # --- Optional z‐range cropping ---
    if z_range is not None:
        z0, z1 = z_range
        raw = raw[z0:z1]
        mask = mask[z0:z1]
        spots_df = spots_df.query("@z0 <= `axis-0` < @z1").assign(**{"axis-0": lambda df: df["axis-0"] - z0})
        if domains_df is not None:
            domains_df = domains_df.query("@z0 <= `axis-0` < @z1").assign(**{"axis-0": lambda df: df["axis-0"] - z0})

    nz, ny, nx = raw.shape

    # --- Helper functions for plotting ---
    def contour_trace(z):
        contours = find_contours(mask[z] > 0, 0.5)
        x_all, y_all = [], []
        for c in contours:
            x_all.extend(c[:, 1].tolist() + [None])
            y_all.extend(c[:, 0].tolist() + [None])
        return go.Scatter(x=x_all, y=y_all, mode="lines", line=dict(color="deepskyblue", width=1.5), name="Nuclei outlines", showlegend=False)

    def create_point_traces(z, z_neighbor_range=5):
        traces = []
        # Plot spots
        for dz in range(-z_neighbor_range, z_neighbor_range + 1):
            current_z = z + dz
            if 0 <= current_z < nz:
                sp = spots_df[spots_df['axis-0'] == current_z]
                if sp.empty: continue
                is_current_slice = (dz == 0)
                opacity = 1.0 if is_current_slice else max(0.1, 1 - abs(dz) / (z_neighbor_range + 1))
                color = f"rgba(255, 0, 0, {opacity:.2f})"
                traces.append(go.Scatter(
                    x=sp['axis-1'], y=sp['axis-2'], mode="markers",
                    marker=dict(size=8, color=color, symbol="circle-open", line=dict(width=2 if is_current_slice else 1)),
                    name=f"Spots (z={current_z})" if not is_current_slice else "Spots (current z)",
                    showlegend=is_current_slice
                ))
        # Plot domains
        if domains_df is not None:
            for dz in range(-z_neighbor_range, z_neighbor_range + 1):
                current_z = z + dz
                if 0 <= current_z < nz:
                    dom = domains_df[domains_df['axis-0'] == current_z]
                    if dom.empty: continue
                    is_current_slice = (dz == 0)
                    opacity = 1.0 if is_current_slice else max(0.1, 1 - abs(dz) / (z_neighbor_range + 1))
                    color = f"rgba(0, 255, 255, {opacity:.2f})"  # Cyan for domains
                    traces.append(go.Scatter(
                        x=dom['axis-1'], y=dom['axis-2'], mode="markers",
                        marker=dict(size=5, color=color, symbol="diamond", line=dict(width=2 if is_current_slice else 1)),
                        name=f"Domains (z={current_z})" if not is_current_slice else "Domains (current z)",
                        showlegend=is_current_slice
                    ))
        return traces

    # --- Build figure frames ---
    frames = [go.Frame(data=[go.Heatmap(z=raw[z], colorscale="gray"), contour_trace(z)] + create_point_traces(z), name=str(z)) for z in range(nz)]
    slider_steps = [dict(method="animate", args=([str(z)], dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0))), label=str(z)) for z in range(nz)]

    # --- Assemble figure ---
    fig = go.Figure(data=[go.Heatmap(z=raw[0], colorscale="gray"), contour_trace(0)] + create_point_traces(0), frames=frames)
    fig.update_layout(
        title=f"Spots, Domains & Nuclei Outlines: {base_id or ''}".strip(),
        xaxis=dict(title="X", autorange="reversed"), yaxis=dict(title="Y", scaleanchor="x"),
        width=900, height=900, legend=dict(x=1.05, y=1),
        sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps)],
        updatemenus=[dict(type="buttons", showactive=False, y=1, x=1.15, xanchor="right", yanchor="top", buttons=[
            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 200}, "fromcurrent": True}]),
            dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])])]
    )

    # --- Save HTML output ---
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_dir / f"{base_id}_z_slider.html", include_plotlyjs="cdn", full_html=True)

    # --- Save Static PNGs ---
    # Maximum projection
    max_proj_fig = go.Figure(go.Heatmap(z=np.max(raw, axis=0), colorscale="gray"))
    all_contours_x, all_contours_y = [], []
    for z_idx in range(raw.shape[0]):
        contours = find_contours(mask[z_idx] > 0, 0.5)
        for c in contours:
            all_contours_x.extend(c[:, 1].tolist() + [None])
            all_contours_y.extend(c[:, 0].tolist() + [None])
    max_proj_fig.add_trace(go.Scatter(x=all_contours_x, y=all_contours_y, mode="lines", line=dict(color="deepskyblue", width=1), name="Nuclei outlines", opacity=0.5))
    max_proj_fig.add_trace(go.Scatter(x=spots_df['axis-1'], y=spots_df['axis-2'], mode="markers", marker=dict(size=2, color="red", opacity=0.6), name="All spots"))
    if domains_df is not None:
        max_proj_fig.add_trace(go.Scatter(x=domains_df['axis-1'], y=domains_df['axis-2'], mode="markers", marker=dict(size=4, color="cyan", opacity=0.8, symbol='diamond'), name="All domains"))
    max_proj_fig.update_layout(title="Maximum projection", xaxis=dict(autorange="reversed"), yaxis=dict(scaleanchor="x"), width=1200, height=1200, plot_bgcolor='white')
    max_proj_fig.write_image(out_dir / f"{base_id}_max_proj.png", scale=2)

    print(f"Successfully generated plots for {base_id}")
    return fig



def render_spots_z_slider(
    spot_mask_dir: Path, # Changed from spots_path to spot_mask_dir
    original_img_path: str,
    mask_path: str,
    domains_path: str = None,
    z_range: tuple = None,
    output_path: str = None,
    base_id: str = None
):
    """
    Generates a 2D z-slice explorer with spot masks, domains, and nuclei outlines.
    This function creates an interactive HTML file with a slider to navigate through
    z-slices, overlaying spot masks and domains from neighboring slices. It also saves
    static maximum projection and middle-slice images.

    Parameters:
    - spot_mask_dir: Directory containing the spot mask TIFF files (e.g., from 'spot_segmentation').
    - original_img_path: Path to the raw 3D image.
    - mask_path: Path to the 3D nuclei/shape mask.
    - domains_path: (Optional) Path to the domain coordinates CSV file.
    - z_range: (Optional) A tuple (start, end) to crop the Z-axis.
    - output_path: Directory to save the HTML and PNG outputs.
    - base_id: A unique identifier for the output files.
    """
    print(f"--- Rendering visualization for {base_id} ---")

    # Construct the full path to the spot mask TIFF file
    spot_mask_path = spot_mask_dir / f"{base_id}_spot_mask.tif"
    if not spot_mask_path.exists():
        print(f"Error: Spot mask file not found at {spot_mask_path}")
        return

    # --- Load Data ---
    raw = AICSImage(original_img_path).get_image_data("ZYX", T=0, C=1)
    raw = np.squeeze(raw).astype(np.float32)

    nuclei_mask = AICSImage(mask_path).get_image_data("ZYX", T=0, C=0)
    nuclei_mask = np.squeeze(nuclei_mask).astype(np.uint8)

    spot_mask = tifffile.imread(str(spot_mask_path)) # Load the spot mask TIFF
    spot_mask = np.squeeze(spot_mask).astype(np.uint16) # Ensure correct dtype

    # --- Load Domain Data (if provided) ---
    domains_df = None
    if domains_path and Path(domains_path).exists():
        try:
            domains_df = pd.read_csv(domains_path)
            if not domains_df.empty:
                domains_df['axis-0'] = domains_df['axis-0'].round().astype(int)
                print(f"Loaded {len(domains_df)} domains.")
            else:
                domains_df = None # Treat empty file as no domains
        except Exception as e:
            print(f"Could not load or process domains file {domains_path}: {e}")
            domains_df = None

    # --- Optional z-range cropping ---
    if z_range is not None:
        z0, z1 = z_range
        raw = raw[z0:z1]
        nuclei_mask = nuclei_mask[z0:z1]
        spot_mask = spot_mask[z0:z1] # Crop spot mask

        if domains_df is not None:
            domains_df = domains_df.query("@z0 <= `axis-0` < @z1").assign(**{"axis-0": lambda df: df["axis-0"] - z0})
    nz, ny, nx = raw.shape

    # --- Helper functions for plotting ---
    def contour_trace_nuclei(z):
        contours = find_contours(nuclei_mask[z] > 0, 0.5)
        x_all, y_all = [], []
        for c in contours:
            x_all.extend(c[:, 1].tolist() + [None])
            y_all.extend(c[:, 0].tolist() + [None])
        return go.Scatter(x=x_all, y=y_all, mode="lines", line=dict(color="deepskyblue", width=1.5), name="Nuclei outlines", showlegend=False)

    def create_spot_mask_traces(z, z_neighbor_range=5):
        traces = []
        for dz in range(-z_neighbor_range, z_neighbor_range + 1):
            current_z = z + dz
            if 0 <= current_z < nz:
                # Extract the current z-slice of the spot mask
                current_spot_slice = spot_mask[current_z]
                unique_spot_labels = np.unique(current_spot_slice)
                
                # Filter out background label (0)
                valid_spot_labels = [lbl for lbl in unique_spot_labels if lbl != 0]

                if not valid_spot_labels:
                    continue

                is_current_slice = (dz == 0)
                opacity = 1.0 if is_current_slice else max(0.1, 1 - abs(dz) / (z_neighbor_range + 1))
                color = f"rgba(255, 0, 0, {opacity:.2f})" # Red for spots

                x_all_spots, y_all_spots = [], []
                for spot_lbl in valid_spot_labels:
                    spot_contours = find_contours(current_spot_slice == spot_lbl, 0.5)
                    for c in spot_contours:
                        x_all_spots.extend(c[:, 1].tolist() + [None])
                        y_all_spots.extend(c[:, 0].tolist() + [None])
                
                if x_all_spots: # Only add trace if there are contours
                    traces.append(go.Scatter(
                        x=x_all_spots, y=y_all_spots, mode="lines",
                        line=dict(color=color, width=2 if is_current_slice else 1),
                        name=f"Spot Mask (z={current_z})" if not is_current_slice else "Spot Mask (current z)",
                        showlegend=is_current_slice
                    ))
        return traces

    def create_domain_traces(z, z_neighbor_range=5):
        traces = []
        # Plot domains
        if domains_df is not None:
            for dz in range(-z_neighbor_range, z_neighbor_range + 1):
                current_z = z + dz
                if 0 <= current_z < nz:
                    dom = domains_df[domains_df['axis-0'] == current_z]
                    if dom.empty: continue
                    is_current_slice = (dz == 0)
                    opacity = 1.0 if is_current_slice else max(0.1, 1 - abs(dz) / (z_neighbor_range + 1))
                    color = f"rgba(0, 255, 255, {opacity:.2f})" # Cyan for domains
                    traces.append(go.Scatter(
                        x=dom['axis-1'], y=dom['axis-2'], mode="markers",
                        marker=dict(size=5, color=color, symbol="diamond", line=dict(width=2 if is_current_slice else 1)),
                        name=f"Domains (z={current_z})" if not is_current_slice else "Domains (current z)",
                        showlegend=is_current_slice
                    ))
        return traces

    # --- Build figure frames ---
    # Combine nuclei contours, spot mask contours, and domain markers
    frames = [go.Frame(data=[
        go.Heatmap(z=raw[z], colorscale="gray"),
        contour_trace_nuclei(z)
    ] + create_spot_mask_traces(z) + create_domain_traces(z), name=str(z)) for z in range(nz)]

    slider_steps = [dict(method="animate", args=([str(z)], dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0))), label=str(z)) for z in range(nz)]

    # --- Assemble figure ---
    fig = go.Figure(data=[
        go.Heatmap(z=raw[0], colorscale="gray"),
        contour_trace_nuclei(0)
    ] + create_spot_mask_traces(0) + create_domain_traces(0), frames=frames)

    fig.update_layout(
        title=f"Spots, Domains & Nuclei Outlines: {base_id or ''}".strip(),
        xaxis=dict(title="X", autorange="reversed"), yaxis=dict(title="Y", scaleanchor="x"),
        width=900, height=900, legend=dict(x=1.05, y=1),
        sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps)],
        updatemenus=[dict(type="buttons", showactive=False, y=1, x=1.15, xanchor="right", yanchor="top", buttons=[
            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 200}, "fromcurrent": True}]),
            dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])])]
    )

    # --- Save HTML output ---
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_dir / f"{base_id}_z_slider.html", include_plotlyjs="cdn", full_html=True)
    print(f"Interactive Z-slider HTML saved to: {out_dir / f'{base_id}_z_slider.html'}")

    # --- Save Static PNGs ---
    # Maximum projection
    max_proj_fig = go.Figure(go.Heatmap(z=np.max(raw, axis=0), colorscale="gray"))
    
    # Add nuclei contours to max projection
    all_nuclei_contours_x, all_nuclei_contours_y = [], []
    for z_idx in range(raw.shape[0]):
        contours = find_contours(nuclei_mask[z_idx] > 0, 0.5)
        for c in contours:
            all_nuclei_contours_x.extend(c[:, 1].tolist() + [None])
            all_nuclei_contours_y.extend(c[:, 0].tolist() + [None])
    max_proj_fig.add_trace(go.Scatter(x=all_nuclei_contours_x, y=all_nuclei_contours_y, mode="lines", line=dict(color="deepskyblue", width=1), name="Nuclei outlines", opacity=0.5))

    # Add spot mask contours to max projection
    all_spot_contours_x, all_spot_contours_y = [], []
    for z_idx in range(raw.shape[0]):
        current_spot_slice = spot_mask[z_idx]
        unique_spot_labels = np.unique(current_spot_slice)
        valid_spot_labels = [lbl for lbl in unique_spot_labels if lbl != 0]
        for spot_lbl in valid_spot_labels:
            spot_contours = find_contours(current_spot_slice == spot_lbl, 0.5)
            for c in spot_contours:
                all_spot_contours_x.extend(c[:, 1].tolist() + [None])
                all_spot_contours_y.extend(c[:, 0].tolist() + [None])
    max_proj_fig.add_trace(go.Scatter(x=all_spot_contours_x, y=all_spot_contours_y, mode="lines", line=dict(color="red", width=1), name="All spot masks", opacity=0.6))

    # Add domains to max projection
    if domains_df is not None:
        max_proj_fig.add_trace(go.Scatter(x=domains_df['axis-1'], y=domains_df['axis-2'], mode="markers", marker=dict(size=4, color="cyan", opacity=0.8, symbol='diamond'), name="All domains"))

    max_proj_fig.update_layout(title="Maximum projection", xaxis=dict(autorange="reversed"), yaxis=dict(scaleanchor="x"), width=1200, height=1200, plot_bgcolor='white')
    max_proj_fig.write_image(out_dir / f"{base_id}_max_proj.png", scale=2)
    print(f"Maximum projection PNG saved to: {out_dir / f'{base_id}_max_proj.png'}")

    print(f"Successfully generated plots for {base_id}")
    return fig

def batch_render_spots_z_slider(
    spot_mask_parent_dir: Path, # Changed to parent directory of spot_segmentation
    mask_dir: Path,
    raw_img_dir: Path,
    qc_out: Path,
    domains_dir: Path = None,
    z_range: tuple = None,
    n_samples: int = None
):
    """
    Batch rendering for spot mask and domain z-slider visualizations.

    Parameters:
    - spot_mask_parent_dir: The parent directory which contains the 'spot_segmentation' folder.
    - mask_dir: Directory containing the corresponding nuclei mask TIFF files.
    - raw_img_dir: Directory containing the corresponding raw image TIFF files.
    - qc_out: The output directory for the generated QC plots.
    - domains_dir: (Optional) Directory containing domain CSV files (e.g., from 'results_domains').
    - z_range: (Optional) A tuple (start, end) to crop the Z-axis.
    - n_samples: (Optional) Number of random samples to process.
    """
    qc_out.mkdir(parents=True, exist_ok=True)
    
    # The spot masks are located in spot_mask_parent_dir / 'spot_segmentation'
    spot_seg_dir = spot_mask_parent_dir / 'spot_segmentation'
    if not spot_seg_dir.exists():
        print(f"Error: Spot segmentation directory not found at {spot_seg_dir}")
        return

    # Find spot mask TIFF files
    spot_mask_files = list(spot_seg_dir.glob("*_spot_mask.tif"))
    if not spot_mask_files:
        print(f"No spot mask TIFF files found in {spot_seg_dir}")
        return

    if n_samples:
        spot_mask_files = random.sample(spot_mask_files, min(n_samples, len(spot_mask_files)))

    for spot_mask_fp in spot_mask_files:
        # Extract base_id from the spot mask filename (e.g., 'my_image_t00_spot_mask.tif' -> 'my_image_t00')
        base = spot_mask_fp.stem.replace("_spot_mask", "")

        # Find corresponding files
        mask_fp = next(mask_dir.glob(f"{base}*.tif"), None)
        raw_fp = next(raw_img_dir.glob(f"{base}*.tif"), None)
        domains_fp = None
        if domains_dir:
            domains_fp_candidate = domains_dir / f"{base}_domains.csv"
            if domains_fp_candidate.exists():
                domains_fp = str(domains_fp_candidate)

        # Validate that all required files exist
        if not mask_fp:
            print(f"Skipping {base}: missing nuclei mask TIFF file.")
            continue
        if not raw_fp:
            print(f"Skipping {base}: missing raw TIFF file.")
            continue

        # Run renderer
        try:
            render_spots_z_slider(
                spot_mask_dir=spot_seg_dir, # Pass the directory containing the spot masks
                mask_path=str(mask_fp),
                original_img_path=str(raw_fp),
                domains_path=domains_fp,
                z_range=z_range,
                output_path=str(qc_out),
                base_id=base
            )
        except Exception as e:
            print(f"!!! Failed to render visualization for {base}: {e}")


# def batch_render_spots_z_slider(
#     spots_dir: Path,
#     mask_dir: Path,
#     raw_img_dir: Path,
#     qc_out: Path,
#     domains_dir: Path = None,
#     z_range: tuple = None,
#     n_samples: int = None
# ):
#     """
#     Batch rendering for spot and domain z-slider visualizations.

#     Parameters:
#     - spots_dir: Directory containing spot CSV files (e.g., from 'results_spots').
#     - mask_dir: Directory containing the corresponding mask TIFF files.
#     - raw_img_dir: Directory containing the corresponding raw image TIFF files.
#     - qc_out: The output directory for the generated QC plots.
#     - domains_dir: (Optional) Directory containing domain CSV files (e.g., from 'results_domains').
#     - z_range: (Optional) A tuple (start, end) to crop the Z-axis.
#     - n_samples: (Optional) Number of random samples to process.
#     """
#     qc_out.mkdir(parents=True, exist_ok=True)
#     spot_files = list(spots_dir.glob("*.csv"))
#     if not spot_files:
#         print(f"No spot CSV files found in {spots_dir}")
#         return

#     if n_samples:
#         spot_files = random.sample(spot_files, min(n_samples, len(spot_files)))

#     for spot_fp in spot_files:
#         base = spot_fp.stem.replace("_spots", "").replace(".csv", "")

#         # Find corresponding files
#         mask_fp = next(mask_dir.glob(f"{base}*.tif"), None)
#         raw_fp = next(raw_img_dir.glob(f"{base}*.tif"), None)
        
#         domains_fp = None
#         if domains_dir:
#             domains_fp_candidate = domains_dir / f"{base}_domains.csv"
#             if domains_fp_candidate.exists():
#                 domains_fp = str(domains_fp_candidate)

#         # Validate that all required files exist
#         if not mask_fp:
#             print(f"Skipping {base}: missing mask TIFF file.")
#             continue
#         if not raw_fp:
#             print(f"Skipping {base}: missing raw TIFF file.")
#             continue

#         # Run renderer
#         try:
#             render_spots_z_slider(
#                 spots_path=str(spot_fp),
#                 mask_path=str(mask_fp),
#                 original_img_path=str(raw_fp),
#                 domains_path=domains_fp,
#                 z_range=z_range,
#                 output_path=str(qc_out),
#                 base_id=base
#             )
#         except Exception as e:
#             print(f"!!! Failed to render visualization for {base}: {e}")





# # Example usage:
# if __name__ == "__main__":
#     spots_path = "/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241108_e_hs/spots/count_spots/20241107_1273_E_30minHS_3h_5min_5um_t13_spots.csv"
#     original_img_path = '/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241108_e_hs/spots/raw_images_timelapse/20241107_1273_E_30minHS_3h_5min_5um_n2v_t13.tif'
#     mask_path = "/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241108_e_hs/segmentation/20241107_1273_E_30minHS_3h_5min_5um_t13.tif"
#     csv_path = "/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241108_e_hs/nuclei/20241107_1273_E_30minHS_3h_5min_5um_t13.csv"
#     output_path = '/mnt/external.data/MeisterLab/mvolosko/image_project/spot-detect-lib/spot_detection/'

#     render_spots_z_slider(
#         spots_path=spots_path,
#         original_img_path=original_img_path,
#         #z_range=(10, 50),          # optional
#         output_path=output_path,
#         base_id="experiment_1",
#         mask_path=mask_path
#     )

