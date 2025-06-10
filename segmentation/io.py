# io.py

from pathlib import Path
import pandas as pd
import pickle
import imageio.v3 as iio
from aicsimageio import AICSImage
from tifffile import TiffWriter
import numpy as np

class FileIndex:
    """Index raw & (optional) denoised images of multiple formats."""
    def __init__(self, raw_dir: Path, denoised_dir: Path, out_csv: Path):
        self.raw_dir = Path(raw_dir)
        self.denoised_dir = Path(denoised_dir) if denoised_dir else None
        self.out_csv = Path(out_csv)

    def build(self, pattern: str = "*.nd2") -> pd.DataFrame:
        raws = sorted(self.raw_dir.glob(pattern))
        denoised_paths = []
        for raw in raws:
            if self.denoised_dir and self.denoised_dir.exists():
                # find any file starting with the raw stem
                candidates = list(self.denoised_dir.glob(f"{raw.stem}*"))
                # keep only .tif/.tiff and exclude '*_max.tif'
                candidates = [
                    c for c in candidates
                    if c.suffix.lower() in (".tif", ".tiff")
                    and not c.name.endswith("_max.tif")
                ]
                # choose the first match if present, else fallback to raw
                denoised_paths.append(candidates[0] if candidates else raw)
            else:
                denoised_paths.append(raw)

        df = pd.DataFrame({
            "id": [p.stem for p in raws],
            "raw_path": raws,
            "denoised_path": denoised_paths,
        })
        df.to_csv(self.out_csv, index=False)
        return df


class ImageIO:
    """Wraps all image reading and writing."""

    @staticmethod
    def read_nd2(path: Path) -> AICSImage:
        """Use AICSImage to load an ND2 (retains metadata)."""
        return AICSImage(str(path))

    @staticmethod
    def read_tiff(path: Path) -> np.ndarray:
        """Load a multi‐page TIFF into a NumPy array."""
        return iio.imread(str(path))

    @staticmethod
    def write_tiff(path: Path, array):
        """Save an ndarray to a TIFF with OME‐compatible metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with TiffWriter(str(path), ome=True) as tif:
            tif.write(array, metadata={'axes': 'ZYX'})


class TabularIO:
    """CSV / pickle read & write for DataFrame results."""
    @staticmethod
    def save_csv(df: pd.DataFrame, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    @staticmethod
    def load_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def save_pickle(obj, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(path: Path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def get_anisotropy(image_path: Path) -> tuple:
    """
    Extract Z/X pixel size ratio from an ND2 or TIFF file.
    Returns (Zratio, 1, 1) for edt anisotropy.
    """
    if image_path.suffix.lower() == ".nd2":
        img = ImageIO.read_nd2(image_path)
        px = img.physical_pixel_sizes  # in µm
        z_ratio = px.Z / px.X
    elif image_path.suffix.lower() in {".tif", ".tiff"}:
        #img = ImageIO.read_tiff(image_path)
        # Assuming isotropic pixel size for TIFF as metadata may not be available
        z_ratio = 1.0
    else:
        raise ValueError(f"Unsupported file format: {image_path.suffix}")
    
    return (z_ratio, 1.0, 1.0)
