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
    z_range: tuple = None,
    output_path: str = ".",
    base_id: str = None
):
    """
    2D z-slice explorer with spots overlay (including neighboring z-slices), nuclei outlines, and slider.
    """
    # --- Load spots and snap z to integer slices ------------------
    spots_df = pd.read_csv(spots_path)
    spots_df['centroid_z'] = spots_df['centroid_z'].round().astype(int)

    # --- Load raw image and mask stacks ---------------------------
    raw = AICSImage(original_img_path).get_image_data("ZYX", T=0, C=1)
    raw = np.squeeze(raw).astype(np.float32)
    mask = AICSImage(mask_path).get_image_data("ZYX", T=0, C=0)
    mask = np.squeeze(mask).astype(np.uint8)

    # --- Optional z‐range cropping -------------------------------
    if z_range is not None:
        z0, z1 = z_range
        raw = raw[z0:z1]
        mask = mask[z0:z1]
        spots_df = (
            spots_df
            .query("@z0 <= centroid_z < @z1")
            .assign(centroid_z=lambda df: df.centroid_z - z0)
        )

    nz, ny, nx = raw.shape

    def contour_trace(z):
        contours = find_contours(mask[z] > 0, 0.5)
        x_all, y_all = [], []
        for c in contours:
            x_all.extend(c[:, 1].tolist() + [None])
            y_all.extend(c[:, 0].tolist() + [None])
        return go.Scatter(
            x=x_all, y=y_all,
            mode="lines",
            line=dict(color="deepskyblue", width=2),  # more vibrant color for outlines
            name="Nuclei outlines",
            showlegend=False
        )

    def spot_scatter(z, z_max=5):
        traces = []
        for dz in range(-z_max, z_max + 1):
            current_z = z + dz
            if 0 <= current_z < nz:
                sp = spots_df[spots_df.centroid_z == current_z]
                if sp.empty:
                    continue
                opacity = max(0.1, 1 - abs(dz) / (z_max + 1))
                color = f"rgba(255,0,0,{opacity:.2f})"
                traces.append(go.Scatter(
                    x=sp.centroid_x,
                    y=sp.centroid_y,
                    mode="markers",
                    marker=dict(size=8, color=color, symbol="circle-open", line=dict(width=2)),
                    name=f"z={current_z}" if dz != 0 else "z=current",
                    showlegend=(dz == 0)
                ))
        return traces

    # --- Initial traces ------------------------------------------
    init_heatmap = go.Heatmap(z=raw[0], colorscale="gray")  # greyscale for raw image
    init_outlines = contour_trace(0)
    init_spots = spot_scatter(0)

    # --- Build frames --------------------------------------------
    frames = []
    legend_shown = set()

    for z in range(nz):
        z_data = [go.Heatmap(z=raw[z], colorscale="gray"),
                contour_trace(z)]
        
        # Plot current z slice spots
        sp_current = spots_df[spots_df.centroid_z == z]
        z_data.append(go.Scatter(
            x=sp_current.centroid_x,
            y=sp_current.centroid_y,
            mode="markers",
            marker=dict(size=6, color="red", opacity=0.9),
            name="Z = current",
            showlegend=(z == 0)
        ))
        
        # Plot surrounding ±5 slices with different color/intensity
        for d in range(1, 6):
            for direction in [-1, 1]:
                zd = z + direction * d
                if not (0 <= zd < nz):
                    continue
                spz = spots_df[spots_df.centroid_z == zd]
                color = distance_color(d)
                label = f"Z {'+' if direction > 0 else '-'}{d}"
                key = f"{direction}{d}"
                z_data.append(go.Scatter(
                    x=spz.centroid_x,
                    y=spz.centroid_y,
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=color,
                        symbol="circle-open",
                        opacity=max(0.2, 0.9 - d * 0.15),
                        line=dict(width=1)
                    ),
                    name=label,
                    showlegend=(key not in legend_shown)
                ))
                legend_shown.add(key)

        frames.append(go.Frame(data=z_data, name=str(z)))

    # --- Slider steps --------------------------------------------
    slider_steps = [
        dict(
            method="animate",
            args=(
                [str(z)],
                dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0))
            ),
            label=str(z)
        )
        for z in range(nz)
    ]

    # --- Assemble figure -----------------------------------------
    fig = go.Figure(data=[init_heatmap, init_outlines] + init_spots, frames=frames)
    
    fig.update_layout(
        title=f"Spots & nuclei outlines {base_id or ''}".strip(),
        xaxis=dict(title="X", autorange="reversed"),
        yaxis=dict(title="Y", scaleanchor="x"),
        width=900,
        height=900,
        margin=dict(t=60, b=40, l=40, r=40),
        sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps)],
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1,
            x=1.15,
            xanchor="right",
            yanchor="top",
            buttons=[
                dict(label="Play", method="animate",
                    args=[None, {"frame": {"duration": 200}, "fromcurrent": True}]),
                dict(label="Pause", method="animate",
                    args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
            ]
        )]
    )

    # --- Save HTML output ----------------------------------------
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{base_id or 'spots'}_z_dadads.html"
    fig.write_html(out_dir / fname, include_plotlyjs="cdn", full_html=True)

    # Save static plots as PNG at higher resolution
    # Maximum projection
    max_proj = np.max(raw, axis=0)
    static_fig = go.Figure()
    static_fig.add_trace(go.Heatmap(z=max_proj, colorscale="gray"))

    # Create merged outlines from all z-slices
    all_contours_x, all_contours_y = [], []
    for z in range(raw.shape[0]):
        contours = find_contours(mask[z] > 0, 0.5)
        for c in contours:
            all_contours_x.extend(c[:, 1].tolist() + [None])
            all_contours_y.extend(c[:, 0].tolist() + [None])
    
    static_fig.add_trace(go.Scatter(
        x=all_contours_x,
        y=all_contours_y,
        mode="lines",
        line=dict(color="deepskyblue", width=1),
        name="Nuclei outlines",
        opacity=0.5
    ))
    static_fig.add_trace(go.Scatter(
        x=spots_df.centroid_x, 
        y=spots_df.centroid_y,
        mode="markers",
        marker=dict(size=2, color="red", opacity=0.6),
        name="All spots"
    ))
    static_fig.update_layout(
        title="Maximum Projection",
        xaxis=dict(title="X", autorange="reversed"),
        yaxis=dict(title="Y", scaleanchor="x"),
        width=2400, height=2400,  # Increased size
        plot_bgcolor='white'
    )
    static_fig.write_image(
        out_dir / f"{base_id or 'spots'}_max_proj.png",
        scale=4  # Increased resolution scale
    )

    # Middle slice
    mid_z = raw.shape[0]//2
    mid_slice_fig = go.Figure()
    mid_slice_fig.add_trace(go.Heatmap(z=raw[mid_z], colorscale="gray"))
    mid_slice_fig.add_trace(contour_trace(mid_z))
    mid_slice_spots = spots_df[spots_df.centroid_z == mid_z]
    mid_slice_fig.add_trace(go.Scatter(
        x=mid_slice_spots.centroid_x,
        y=mid_slice_spots.centroid_y,
        mode="markers",
        marker=dict(size=2, color="red", opacity=0.9),
        name="Spots in slice"
    ))
    mid_slice_fig.update_layout(
        title=f"Middle slice (Z={mid_z})",
        xaxis=dict(title="X", autorange="reversed"),
        yaxis=dict(title="Y", scaleanchor="x"),
        width=2400, height=2400,  # Increased size
        plot_bgcolor='white'
    )
    mid_slice_fig.write_image(
        out_dir / f"{base_id or 'spots'}_mid_slice.png",
        scale=4  # Increased resolution scale
    )
    return fig

def batch_render_spots_z_slider(
    spots_dir: Path,
    mask_dir: Path,
    raw_img_dir: Path,
    qc_out: Path,
    z_range: tuple = None,
    n_samples: int = None
):
    """
    Batch rendering for spot z-slider visualizations.
    Matches spot files with corresponding masks and raw images.
    """
    qc_out.mkdir(parents=True, exist_ok=True)
    spot_files = list(spots_dir.glob("*_spots.csv"))
    if not spot_files:
        print("No spot CSV files found.")
        return

    if n_samples:
        spot_files = random.sample(spot_files, min(n_samples, len(spot_files)))

    for spot_fp in spot_files:
        base = spot_fp.stem.replace("_spots", "")

        # Mask file
        mask_fp_candidates = list(mask_dir.glob(f"{base}.tif"))
        # if not mask_fp_candidates:
        #     # Try with _t00 ending for mask
        #     base_with_t00 = base
        # if not base.endswith("_t00"):
        #     base_with_t00 = base + "_t00"
        #     mask_fp_candidates = list(mask_dir.glob(f"{base_with_t00}.tif"))
        mask_fp = mask_fp_candidates[0] if mask_fp_candidates else None

        # Raw image
        base_with_n2v = base.replace("_t", "_n2v_t")
        raw_fp_candidates = list(raw_img_dir.glob(f"{base_with_n2v}*.tif"))
        if not raw_fp_candidates:
            base_no_t = base.split("_t")[0]
            raw_fp_candidates = list(raw_img_dir.glob(f"{base_no_t}_n2v.tif"))
        if not raw_fp_candidates:
            raw_fp_candidates = list(raw_img_dir.glob(f"{base}.tif"))
        raw_fp = raw_fp_candidates[0] if raw_fp_candidates else None

        # Validate presence of files
        missing = [name for p, name in [
            (mask_fp, "mask TIFF"),
            (raw_fp, "raw TIFF")
        ] if p is None or not Path(p).exists()]

        if missing:
            print(f"Skipping {base}: missing {', '.join(missing)}")
            continue

        # Run renderer
        render_spots_z_slider(
            spots_path=str(spot_fp),
            mask_path=str(mask_fp),
            original_img_path=str(raw_fp),
            z_range=z_range,
            output_path=str(qc_out),
            base_id=base
        )
        print(f" Rendered Z-slider for {base}")



# Example usage:
if __name__ == "__main__":
    spots_path = "/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241108_e_hs/spots/count_spots/20241107_1273_E_30minHS_3h_5min_5um_t13_spots.csv"
    original_img_path = '/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241108_e_hs/spots/raw_images_timelapse/20241107_1273_E_30minHS_3h_5min_5um_n2v_t13.tif'
    mask_path = "/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241108_e_hs/segmentation/20241107_1273_E_30minHS_3h_5min_5um_t13.tif"
    csv_path = "/mnt/external.data/MeisterLab/mvolosko/image_project/SDC1/1273/20241108_e_hs/nuclei/20241107_1273_E_30minHS_3h_5min_5um_t13.csv"
    output_path = '/mnt/external.data/MeisterLab/mvolosko/image_project/spot-detect-lib/spot_detection/'

    render_spots_z_slider(
        spots_path=spots_path,
        original_img_path=original_img_path,
        #z_range=(10, 50),          # optional
        output_path=output_path,
        base_id="experiment_1",
        mask_path=mask_path
    )

