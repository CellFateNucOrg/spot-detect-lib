# analysis.py

import os
import pandas as pd
import numpy as np
from skimage.measure import regionprops_table
from scipy.ndimage import distance_transform_edt
from pathlib import Path
from typing import Tuple, List
from bioio import BioImage
import bioio_nd2
import bioio_tifffile
from .utils import timeit
from .logger import logger


class DistanceAnalyzer:
    """
    Compute distance-transformed intensity profiles for each nucleus,
    plus summary statistics and collect data across a whole experiment.
    """
    def __init__(
        self,
        raw_dir: Path,
        seg_dir: Path,
        output_dist_dir: Path,
        output_nuclei_dir: Path,
        spot_channel: int = 1,
        nuc_channel: int = 0
    ):
        self.raw_dir = Path(raw_dir)
        self.seg_dir = Path(seg_dir)
        self.out_dist = Path(output_dist_dir)
        self.out_nuc = Path(output_nuclei_dir)
        self.spot_ch = spot_channel
        self.nuc_ch = nuc_channel
        for d in (self.out_dist, self.out_nuc):
            d.mkdir(parents=True, exist_ok=True)

    @timeit
    def run_for_position(self, position_id: str) -> None:
        """
        - Load raw ND2 (or TIFF) per‐timepoint.
        - Load corresponding segmentation masks.
        - Compute per‐nucleus, per‐distance mean intensities.
        - Save a pickle of the full DataFrame and a CSV for easy viewing.
        """
        df_nuc = []
        raw_path = self.raw_dir / f"{position_id}.nd2"
        img5d = BioImage(raw_path, reader=bioio_nd2.Reader)


        for t in range(img5d.dims.T):
            logger.info(f"    t={t:02d}: computing regionprops")
            img = img5d.get_image_data("ZYXC", T=t)
            mask = BioImage(
                self.seg_dir / f"{position_id}_t{t:02d}.tif",
                reader=bioio_tifffile.Reader
            ).get_image_data("ZYX", T=0, C=0)

            rp = regionprops_table(
                mask,
                intensity_image=img,
                properties=[
                    "label","area","centroid",
                    "MajorAxisLength","solidity",
                    "image","intensity_image"
                ]
            )
            df_rp = pd.DataFrame(rp)
            if df_rp.empty:
                continue

            df_rp["timepoint"] = t
            df_rp["id"] = position_id

            anisotropy = np.round(img5d.physical_pixel_sizes.Z / img5d.physical_pixel_sizes.X, 0)

            # compute distance‐intensity profiles
            profiles = self._compute_profiles(df_rp, anisotropy)
            df_nuc.append(profiles)
            

        if not df_nuc:
            return

        df_all = pd.concat(df_nuc, ignore_index=True)
        # full pickle for downstream plotting/fit
        df_all.to_pickle(self.out_dist / f"{position_id}.pkl")
        # lean CSV without arrays
        df_csv = df_all.drop(
            columns=["intensity_profile_spots","intensity_profile_nuc"]
        )
        df_csv.to_csv(self.out_nuc / f"{position_id}.csv", index=False)
        # After writing the per‐position CSV, split out per‐timepoint tables
        for t, df_t in df_csv.groupby("timepoint"):
            # e.g. nuclei/pos1_t00.csv, nuclei/pos1_t01.csv, …
            df_t.to_csv(self.out_nuc / f"{position_id}_t{t:02d}.csv", index=False)
        logger.info(f"✅ Completed analysis for {position_id}")
        
    def _compute_profiles(self, df_rp: pd.DataFrame, anisotropy: float) -> pd.DataFrame:
        """
        For each row in regionprops DataFrame:
          1. extract central 2D slice from 3D mask
          2. compute distance transform
          3. mask spot and nuc channel
          4. compute mean intensity at each integer distance
        """
        records = []
        for _, row in df_rp.iterrows():
            mask3d = row["image"]  # boolean 3D
            spot3d = row["intensity_image"][..., self.spot_ch]
            nuc3d  = row["intensity_image"][..., self.nuc_ch]

            zc = mask3d.shape[0] // 2
            m2d = mask3d[zc]
            d2d = distance_transform_edt(m2d)

            prof_spot = []
            prof_nuc  = []
            maxd = int(d2d.max())
            for d in range(maxd + 1):
                ann = (d2d.astype(int) == d)
                if not ann.any():
                    prof_spot.append(np.nan)
                    prof_nuc.append(np.nan)
                else:
                    prof_spot.append(spot3d[zc][ann].mean())
                    prof_nuc.append(nuc3d[zc][ann].mean())

            rec = {
                **{k: row[k] for k in ["label","area","centroid-0","centroid-1","centroid-2","MajorAxisLength","solidity","timepoint","id"]},
                "bb_dimZ": mask3d.shape[0],
                "bb_dimY": mask3d.shape[1],
                "bb_dimX": mask3d.shape[2],
                "anisotropy": anisotropy,
                "intensity_profile_spots": prof_spot,
                "intensity_profile_nuc": prof_nuc,
            }
            records.append(rec)
        return pd.DataFrame.from_records(records)

    def collect_all_nuclei_csv(self, df_index: pd.Index) -> pd.DataFrame:
        """
        Read every per-position CSV and concatenate into one table.
        """
        dfs = []
        for pos in df_index:
            path = self.out_nuc / f"{pos}.csv"
            if path.exists():
                dfs.append(pd.read_csv(path))
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(self.out_nuc.parent / "nuclei_analysis.csv", index=False)
        return combined

    def collect_all_dist_profiles(self, df_index: pd.Index) -> pd.DataFrame:
        """
        Read every per-position pickle of profiles and concatenate.
        """
        dfs = []
        for pos in df_index:
            path = self.out_dist / f"{pos}.pkl"
            if path.exists():
                dfs.append(pd.read_pickle(path))
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_pickle(self.out_dist.parent / "dist_profiles_all.pkl")
        return combined
