# segment.py

from pathlib import Path
import gc
import numpy as np
from tqdm import tqdm
from cellpose import models
import edt
from skimage import exposure
from aicsimageio import AICSImage
from .io import ImageIO, get_anisotropy
from torch.serialization import DEFAULT_PROTOCOL
import torch
from .utils import timeit
from .logger import logger
from skimage.filters import gaussian 


class Segmenter:
    """
    Wrapper around a Cellpose nucleus model to
    segment 4D (C, Z, Y, X) image volumes per timepoint,
    save masks and their distance‐transforms.
    """
    def __init__(
        self,
        model_path: Path,
        gpu: bool = True,
        diameter: float = 125, #None,
        nuc_channel: int = 0,
        stitch_threshold: float = 0.3,
        cellprob_threshold: float = 0.0,
        invert: bool = False,
        blur_sigma: float = None
    ):
        """
        model_path: path to a pretrained Cellpose model
        gpu: run on GPU if available
        diameter: expected nucleus diameter (pixels)
        nuc_channel: channel index for nuclei
        stitch_threshold, cellprob_threshold: Cellpose params
        blur_sigma: standard deviation for gaussian blur, if provided
        """
        self.model = models.CellposeModel(
            pretrained_model=str(model_path),
            gpu=gpu,
        )
        self.diameter = diameter
        self.nuc_channel = nuc_channel
        self.stitch_threshold = stitch_threshold
        self.cellprob_threshold = cellprob_threshold
        self.invert = invert
        self.blur_sigma = blur_sigma  # <-- store blur sigma

    def segment_image(self, volume: np.ndarray) -> np.ndarray:
        """
        Run Cellpose on a single 3D volume (C, Z, Y, X) or 2D slice.
        Returns a 3D mask (Z, Y, X).
        """
        volume = exposure.rescale_intensity(volume.astype(np.float32), out_range=(0, 1))
        if self.blur_sigma is not None:
            volume = gaussian(volume, sigma=self.blur_sigma, preserve_range=True)
        masks, flows, styles = self.model.eval(
            volume,
            diameter=self.diameter,
            channels=[0, 0],            # single‐channel greyscale
            do_3D=False,
            z_axis=0,          # Explicitly define z-axis (dim 0 for ZYXC data)
            channel_axis=-1,   # Channels last (ZYXC)
            stitch_threshold=self.stitch_threshold,
            cellprob_threshold=self.cellprob_threshold,
            invert=self.invert
        )
        # flows & styles can be GC’d once masks saved
        del flows, styles
        return masks.astype(np.uint16)

    def calc_edt(self, masks: np.ndarray, anisotropy: tuple) -> np.ndarray:
        """
        Compute Euclidean distance transform on 3D mask.
        anisotropy: (z_ratio, 1.0, 1.0) for voxel scaling.
        """
        return edt.edt(masks > 0, anisotropy=anisotropy).astype(np.float32)

    @timeit
    def run_on_directory(
        self,
        file_index: "pd.DataFrame",
        out_seg_dir: Path,
        out_edt_dir: Path,
        qc: bool = False,
        qc_func: callable = None
    ):
        out_seg_dir.mkdir(parents=True, exist_ok=True)
        out_edt_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting segmentation on {len(file_index)} positions")
        for _, row in file_index.iterrows():
            pos_id = row["id"]
            raw_path = Path(row["raw_path"])
            logger.info(f"Segmenting position {pos_id}")
            denoised_path = Path(row["denoised_path"])

            # compute anisotropy from raw ND2 metadata
            anisotropy = get_anisotropy(raw_path)

            # decide which image to load: denoised if exists, else raw
            if denoised_path.exists():
                img5d = AICSImage(str(denoised_path))
            else:
                img5d = AICSImage(str(raw_path))

            for t in range(img5d.dims.T):
                logger.info(f"{t:02d}: loading volume and running Cellpose")
                vol = img5d.get_image_data("CZYX", T=t)
                #print(pos_id, "vol shape:", nuc_vol.shape)  # should be (1, Z, Y, X) with all dims >0
                nuc_vol = vol[self.nuc_channel : self.nuc_channel + 1]

                masks = self.segment_image(nuc_vol)
                edt_map = self.calc_edt(masks, anisotropy)

                seg_path = out_seg_dir / f"{pos_id}_t{t:02d}.tif"
                edt_path = out_edt_dir / f"{pos_id}_t{t:02d}.tif"
                ImageIO.write_tiff(seg_path, masks)
                ImageIO.write_tiff(edt_path, edt_map)
                logger.info(f"t={t:02d}: saving masks and EDT maps")
                if qc and qc_func:
                    qc_func(vol, masks, pos_id, t)

            logger.info("All positions segmented")