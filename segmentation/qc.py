# qc.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from matplotlib_scalebar.scalebar import ScaleBar


def plot_cropped_nuclei(
    df_region_props,
    image_5d: np.ndarray,
    out_dir: Path,
    sample_id: str,
    timepoint: int = 0,
    spot_channel: int = 1,
    nuc_channel: int = 0,
    n_nuclei: int = 10,
    seed: int = 1,
    pixel_size_um: float = None,
    display: bool = False
):
    """
    Plot a grid of cropped mid‐Z slices for random nuclei:
      - Top row: spot channel
      - Bottom row: nuclear channel
    Saves to out_dir / qc / cropped_{sample_id}_t{timepoint:02d}.pdf
    """
    out_dir = Path(out_dir) / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    props = df_region_props.reset_index(drop=True)
    n_nuclei = min(n_nuclei, len(props))
    indices = np.random.choice(props.index, size=n_nuclei, replace=False)

    # mid‐Z frame
    mask3d = props.loc[indices[0], "image"]
    zc = mask3d.shape[0] // 2

    fig, axs = plt.subplots(
        2, n_nuclei,
        figsize=(n_nuclei * 1.5, 4),
        dpi=200,
        sharex=False, sharey=False
    )
    fig.suptitle(f"Cropped nuclei: {sample_id} (t={timepoint})", fontsize=12)

    for row, channel in enumerate((spot_channel, nuc_channel)):
        for col, idx in enumerate(indices):
            intensity = props.loc[idx, "intensity_image"][..., channel]
            mask = props.loc[idx, "image"]
            slice_img = np.ma.masked_array(intensity[zc], mask=~mask[zc])

            ax = axs[row, col]
            ax.imshow(slice_img, cmap="gray")
            ax.axis("off")

            # only add scalar bar on last plot of bottom row
            if (
                row == 1
                and col == n_nuclei - 1
                and pixel_size_um is not None
            ):
                scalebar = ScaleBar(
                    pixel_size_um, "µm",
                    length_fraction=0.5,
                    location="lower right",
                    box_alpha=0.5
                )
                ax.add_artist(scalebar)

    out_path = out_dir / f"cropped_{sample_id}_t{timepoint:02d}.pdf"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if display:
        plt.show()
    else:
        fig.savefig(out_path)
        plt.close(fig)


def plot_single_nucleus(
    df_region_props,
    image_5d: np.ndarray,
    nuc_index: int,
    out_dir: Path,
    sample_id: str,
    spot_channel: int = 1,
    nuc_channel: int = 0,
    pixel_size_um: float = None,
    display: bool = False
):
    """
    Plot a single nucleus crop side‐by‐side:
      Left: spot channel
      Right: nuclear channel
    """
    out_dir = Path(out_dir) / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    mask3d = df_region_props.loc[nuc_index, "image"]
    zc = mask3d.shape[0] // 2

    fig, axs = plt.subplots(
        1, 2,
        figsize=(4, 2),
        dpi=200,
        sharey=True
    )
    fig.suptitle(f"{sample_id} nucleus #{nuc_index}", fontsize=10)

    for ax, channel in zip(axs, (spot_channel, nuc_channel)):
        intensity = df_region_props.loc[nuc_index, "intensity_image"][..., channel]
        mask = mask3d
        slice_img = np.ma.masked_array(intensity[zc], mask=~mask[zc])
        ax.imshow(slice_img, cmap="gray")
        ax.axis("off")

        if (
            channel == nuc_channel
            and pixel_size_um is not None
        ):
            scalebar = ScaleBar(
                pixel_size_um, "µm",
                length_fraction=0.5,
                location="lower right",
                box_alpha=0.5
            )
            ax.add_artist(scalebar)

    out_path = out_dir / f"single_nucleus_{sample_id}_{nuc_index}.pdf"
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    if display:
        plt.show()
    else:
        fig.savefig(out_path)
        plt.close(fig)


def plot_segmentation_slices(
    img_5d: np.ndarray,
    masks: np.ndarray,
    out_dir: Path,
    sample_id: str,
    timepoint: int = 0,
    nuc_channel: int = 0,
    display: bool = False,
    draw_contours: bool = False,
):
    """
    Creates a 2×3 grid of XY slices at 30%, 50%, 70% Z,
    overlaying masks (top) and raw signal (bottom),
    and a 4×3 grid of XZ/YZ similarly.
    """
    out_dir = Path(out_dir) / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    C, Z, Y, X = img_5d.shape
    z_slices = [int(img_5d.shape[1] * f) for f in (0.3, 0.5, 0.7)]
    y_slices = [int(img_5d.shape[2] * f) for f in (0.3, 0.5, 0.7)]
    x_slices = [int(img_5d.shape[3] * f) for f in (0.3, 0.5, 0.7)]

    fig = plt.figure(constrained_layout=True, figsize=(10, 12), dpi=200)
    fig.suptitle(f"Segmentation QC: {sample_id} (t={timepoint})", fontsize=12)
    subfigs = fig.subfigures(2, 1, hspace=0.1)

    # XY planes
    axm = subfigs[0].subplots(2, 3, sharex=True, sharey=True)
    for col, z in enumerate(z_slices):
        # mask color overlays
        colored = label2rgb(
            masks[z],
            bg_label=0,
            bg_color=(1, 1, 1),
            colors=np.random.rand(masks.max()+1, 3)
        )
        axm[0, col].imshow(colored)
        axm[1, col].imshow(img_5d[nuc_channel, z], cmap="gray_r")
        if draw_contours:
            axm[1, col].contour(masks[z], levels=[0.5], colors="r", linewidths=0.5)
        for ax in axm[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])

    # XZ and YZ planes
    axs = subfigs[1].subplots(4, 3, sharex=True, sharey=True)

    # First two rows: XZ at each Y slice
    for col, y in enumerate(y_slices):
        mp = masks[:, y, :]                    # mask XZ plane at this Y
        ip = img_5d[nuc_channel, :, y, :]      # image XZ plane
        axs[0, col].imshow(label2rgb(mp, bg_label=0)) 
        axs[1, col].imshow(ip, cmap="gray_r")
        if draw_contours:
            axs[1, col].contour(mp, levels=[0.5], colors="r", linewidths=0.5)
        for ax in (axs[0, col], axs[1, col]):
            ax.set_xticks([]); ax.set_yticks([])

    # Next two rows: YZ at each X slice
    for col, x in enumerate(x_slices):
        mp = masks[:, :, x]                    # mask YZ plane at this X
        ip = img_5d[nuc_channel, :, :, x]      # image YZ plane
        axs[2, col].imshow(label2rgb(mp, bg_label=0))
        axs[3, col].imshow(ip, cmap="gray_r")
        if draw_contours:
            axs[3, col].contour(mp, levels=[0.5], colors="r", linewidths=0.5)
        for ax in (axs[2, col], axs[3, col]):
            ax.set_xticks([]); ax.set_yticks([])


    out_path = out_dir / f"segmentation_{sample_id}_t{timepoint:02d}.png"
    if display:
        plt.show()
    else:
        fig.savefig(out_path)
        plt.close(fig)
