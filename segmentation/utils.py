# utils.py

import os
import random
import time
from pathlib import Path
from functools import wraps
from typing import Callable, Any, Sequence, Tuple

import numpy as np


def ensure_dir(path: Path) -> Path:
    """
    Create a directory (and parents) if it doesn’t exist.
    Returns the resolved Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path




def seed_everything(seed: int = 42):
    """
    Set random seed for reproducibility across `random`, `numpy` and `torch` (if available).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure and print the execution time of functions.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[timeit] {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


def reorder_axes(arr: np.ndarray, src: str, dst: str) -> np.ndarray:
    """
    Reorder a NumPy array’s axes labeled by strings `src` → `dst`.
    Example: reorder_axes(volume, "CZYX", "ZYXC").
    """
    assert arr.ndim == len(src), "Axis labels must match array dimensions"
    axes = [src.index(ax) for ax in dst]
    return np.transpose(arr, axes)


def compute_bbox_dimensions(mask: np.ndarray) -> Tuple[int, int, int]:
    """
    Compute the size (Z, Y, X) of the minimal bounding box that contains all nonzero voxels.
    """
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return tuple((maxs - mins + 1).tolist())


def smooth_profile(profile: Sequence[float], window: int = 3) -> np.ndarray:
    """
    Apply a moving‐average smoothing to a 1D profile.
    """
    arr = np.asarray(profile, dtype=float)
    if window < 2:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def parallel_map(func: Callable, items: Sequence[Any], n_workers: int = None) -> list:
    """
    Parallel version of `map(func, items)` using multiprocessing.Pool.
    """
    import multiprocessing as mp

    with mp.Pool(n_workers) as pool:
        return pool.map(func, items)


def generate_random_colormap(n: int) -> np.ndarray:
    """
    Return an (n, 3) array of random RGB colors in [0, 1],
    useful for label2rgb overlays.
    """
    return np.random.rand(n, 3)


def apply_gmm(image: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Fit a Gaussian Mixture Model to the intensity distribution of `image`
    and return the component labels array of the same shape.
    """
    from sklearn.mixture import GaussianMixture

    flat = image.reshape(-1, 1)
    gm = GaussianMixture(n_components=n_components).fit(flat)
    labels = gm.predict(flat)
    return labels.reshape(image.shape)
