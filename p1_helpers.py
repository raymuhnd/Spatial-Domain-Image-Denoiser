"""
CS 6640 — Project 1 helper utilities
Denoising by Design: From Classical to Learned Spatial Filters

These helpers remove routine software friction so that your effort goes into
assumptions, predictions, experimental control, evidence, and interpretation.

What is supplied here
---------------------
* deterministic image loading and standardization to grayscale float in [0, 1]
* deterministic corruption (additive Gaussian, salt-and-pepper / impulse)
* a PSNR convenience wrapper that fixes the data_range convention
* a common-crop utility so every comparison looks at the same pixels
* display and table-formatting utilities
* the explicit normalized analytical Gaussian kernel used in Parts 2 and 5

What is deliberately NOT supplied here
--------------------------------------
No helper chooses a filter, chooses a parameter, decides whether an experiment
succeeded, or writes an interpretation. Those are the assessed parts of the
project and they belong to you.
"""

from __future__ import annotations

import inspect

import matplotlib.pyplot as plt
import numpy as np
import skimage.data
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import img_as_float, random_noise

# --------------------------------------------------------------------------
# Fixed experimental constants for Project 1.
# These are part of the assignment, not tuning knobs. Do not change them.
# --------------------------------------------------------------------------

SEED = 6640
GAUSSIAN_SIGMA_NOISE = 0.08
IMPULSE_AMOUNT = 0.08
DEFAULT_BOUNDARY = "reflect"
DATA_RANGE = 1.0

# Diagnostic crops used for same-location visual comparison.
# (row0, col0, height, width)
CROP_BOXES = {
    "camera": (180, 180, 96, 96),
    "coins": (80, 120, 96, 96),
    "moon": (200, 200, 96, 96),
    "astronaut": (150, 150, 96, 96),
}


# --------------------------------------------------------------------------
# Loading and representation
# --------------------------------------------------------------------------

def load_standard_image(name: str) -> np.ndarray:
    """Load a built-in scikit-image sample and standardize the representation.

    Returns a 2-D grayscale array of dtype float64 with values in [0, 1].
    No image files need to be downloaded or distributed: every image used by
    this project ships inside scikit-image.

    Supported names: 'camera', 'coins', 'moon', 'astronaut'.
    """
    if not hasattr(skimage.data, name):
        raise ValueError(f"unknown skimage.data image: {name!r}")
    img = getattr(skimage.data, name)()
    img = img_as_float(img)
    if img.ndim == 3:
        img = rgb2gray(img)
    if img.ndim != 2:
        raise ValueError(f"{name!r} did not reduce to a 2-D grayscale image")
    return np.clip(img.astype(np.float64), 0.0, 1.0)


def describe_image(img: np.ndarray, name: str) -> dict:
    """Return the representation facts Part 1 asks you to record."""
    return {
        "name": name,
        "shape": tuple(img.shape),
        "dtype": str(img.dtype),
        "min": float(img.min()),
        "max": float(img.max()),
        "mean": float(img.mean()),
    }


# --------------------------------------------------------------------------
# Deterministic corruption
# --------------------------------------------------------------------------

def _random_noise(img: np.ndarray, *, mode: str, seed: int, **kwargs) -> np.ndarray:
    """Call skimage.util.random_noise deterministically across skimage versions.

    Older releases name the seed argument ``seed``; current releases name it
    ``rng``. This wrapper keeps the project reproducible either way.
    """
    params = inspect.signature(random_noise).parameters
    key = "rng" if "rng" in params else "seed"
    out = random_noise(img, mode=mode, **{key: seed}, clip=True, **kwargs)
    return np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)


def add_gaussian_noise(img: np.ndarray, sigma: float = GAUSSIAN_SIGMA_NOISE,
                       seed: int = SEED) -> np.ndarray:
    """Additive zero-mean Gaussian corruption, clipped back into [0, 1].

    Range convention: the corrupted image is clipped to the valid display range
    [0, 1]. Clipping is applied to the corrupted images only; filter outputs of
    inputs in [0, 1] already lie in [0, 1] and are never re-clipped, so no
    filter is quietly helped or hurt by a range operation.
    """
    return _random_noise(img, mode="gaussian", seed=seed, mean=0.0, var=float(sigma) ** 2)


def add_impulse_noise(img: np.ndarray, amount: float = IMPULSE_AMOUNT,
                      seed: int = SEED) -> np.ndarray:
    """Salt-and-pepper (impulse) corruption of the given fraction of pixels."""
    return _random_noise(img, mode="s&p", seed=seed, amount=float(amount))


def make_corruptions(img: np.ndarray, seed: int = SEED) -> dict:
    """Return {'gaussian': ..., 'impulse': ...} using the fixed project settings."""
    return {
        "gaussian": add_gaussian_noise(img, GAUSSIAN_SIGMA_NOISE, seed),
        "impulse": add_impulse_noise(img, IMPULSE_AMOUNT, seed),
    }


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------

def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """PSNR in dB against a known clean reference, with data_range=1.0.

    The project standardizes images to floating point in [0, 1], so the peak
    signal value is 1.0. Passing data_range explicitly is what makes PSNR
    values comparable across every table in this project.
    """
    return float(peak_signal_noise_ratio(reference, test, data_range=DATA_RANGE))


# --------------------------------------------------------------------------
# Common crops and display
# --------------------------------------------------------------------------

def crop(img: np.ndarray, box=None, image_name: str = "camera") -> np.ndarray:
    """Extract the same diagnostic region from any image of the same size."""
    if box is None:
        box = CROP_BOXES[image_name]
    r0, c0, h, w = box
    return img[r0:r0 + h, c0:c0 + w]


def boundary_crop(img: np.ndarray, size: int = 64, corner: str = "top_left") -> np.ndarray:
    """Extract a crop that touches an image boundary (for the Part 4 experiment)."""
    if corner == "top_left":
        return img[:size, :size]
    if corner == "top_right":
        return img[:size, -size:]
    if corner == "bottom_left":
        return img[-size:, :size]
    if corner == "bottom_right":
        return img[-size:, -size:]
    raise ValueError(f"unknown corner: {corner!r}")


def show_images(images, titles=None, cmap="gray", figsize_per_image=3.2,
                suptitle=None, vmin=0.0, vmax=1.0):
    """Display a row of images on a fixed intensity scale.

    A fixed [0, 1] display scale matters: per-image autoscaling can make two
    very different results look identical.
    """
    images = list(images)
    titles = list(titles) if titles is not None else [""] * len(images)
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_image * n, figsize_per_image + 0.6))
    if n == 1:
        axes = [axes]
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=9)
        ax.set_axis_off()
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig, axes


def show_kernel(kernel: np.ndarray, title: str = "kernel", annotate: bool = True):
    """Display a small kernel as an image with its coefficient values."""
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    im = ax.imshow(kernel, cmap="viridis")
    ax.set_title(f"{title}\nsum = {kernel.sum():.6f}", fontsize=10)
    ax.set_xticks(range(kernel.shape[1]))
    ax.set_yticks(range(kernel.shape[0]))
    if annotate:
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                ax.text(j, i, f"{kernel[i, j]:.3f}", ha="center", va="center",
                        fontsize=7, color="w")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig, ax


def results_table(rows, headers) -> str:
    """Format a list of rows as a Markdown table for the notebook record.

    Print the returned string, or use ``display(Markdown(results_table(...)))``.
    Build tables from computed values; never transcribe numbers by hand.
    """
    def fmt(v):
        if isinstance(v, float):
            if v != 0.0 and abs(v) < 1e-3:
                return f"{v:.3e}"
            return f"{v:.4f}" if abs(v) < 1.0 else f"{v:.3f}"
        return str(v)

    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------
# The explicit analytical Gaussian kernel (used in Part 2 and Part 5)
# --------------------------------------------------------------------------

def analytical_gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Return a normalized (size x size) sampled Gaussian kernel.

    The kernel is the sampled 2-D Gaussian on the discrete grid centered at the
    kernel origin, divided by the sum of its coefficients so that the
    coefficients sum to exactly 1. This is the *explicitly specified* operator
    that Part 5 asks a single learnable convolution layer to reproduce from
    examples.
    """
    if size % 2 == 0:
        raise ValueError("kernel size must be odd so the kernel has a center")
    r = (size - 1) // 2
    ax = np.arange(-r, r + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * float(sigma) ** 2))
    return k / k.sum()


def environment_report() -> str:
    """One-line-per-package versions, for the reproducibility record."""
    import platform
    import matplotlib
    import scipy
    import skimage
    lines = [
        f"python      {platform.python_version()}",
        f"numpy       {np.__version__}",
        f"scipy       {scipy.__version__}",
        f"scikit-image {skimage.__version__}",
        f"matplotlib  {matplotlib.__version__}",
    ]
    try:
        import torch
        lines.append(f"torch       {torch.__version__}")
    except Exception:
        lines.append("torch       (not installed)")
    return "\n".join(lines)
