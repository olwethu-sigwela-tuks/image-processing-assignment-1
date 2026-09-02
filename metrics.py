"""
Shared image I/O and evaluation-metric utilities.

Used by both the single-run driver (run_experiment.py) and the batch
experiment harness (experiment_harness.py) so the two never drift out
of sync on how segmentation / PSNR / SSIM / Uniformity are computed.
"""

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim


def load_grayscale_histogram(image_path):
    """Load an image, convert to 8-bit grayscale, return (gray_array, hist_prob)."""
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    hist_prob = hist / hist.sum()
    return arr, hist_prob


def segment_image(gray_arr, thresholds):
    """
    Map every pixel to the mean intensity of its class, using the
    supplied threshold boundaries. `thresholds` must be sorted ints.
    """
    bounds = np.concatenate(([0], thresholds, [256]))
    out = np.zeros_like(gray_arr, dtype=np.uint8)
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        mask = (gray_arr >= lo) & (gray_arr < hi)
        if not np.any(mask):
            continue
        class_mean = gray_arr[mask].mean()
        out[mask] = np.uint8(round(class_mean))
    return out


def compute_psnr(original, segmented):
    """Peak Signal-to-Noise Ratio between original and reconstructed image (dB)."""
    mse = np.mean((original.astype(float) - segmented.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / mse)


def compute_ssim(original, segmented):
    """Structural Similarity Index between original and reconstructed image."""
    return float(sk_ssim(original, segmented, data_range=255))


def compute_uniformity(gray_arr, thresholds):
    """
    Feature Uniformity Measure (Levine & Nazif, 1985), generalised to
    K thresholds -- widely used in the multilevel thresholding
    literature as a segmentation-quality metric that doesn't require
    ground truth.

        U = 1 - (2 * sum_k sum_{i in C_k} (f_i - mu_k)^2)
                / (M*N * (f_max - f_min)^2)

    Higher U (closer to 1) means each region is internally more
    homogeneous relative to the image's overall dynamic range.
    """
    bounds = np.concatenate(([0], np.asarray(thresholds), [256])).astype(int)
    M_times_N = gray_arr.size
    f_max, f_min = 255.0, 0.0
    denom = M_times_N * (f_max - f_min) ** 2

    ssd_total = 0.0
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        mask = (gray_arr >= lo) & (gray_arr < hi)
        if not np.any(mask):
            continue
        class_pixels = gray_arr[mask].astype(float)
        mu_k = class_pixels.mean()
        ssd_total += np.sum((class_pixels - mu_k) ** 2)

    return 1.0 - (2.0 * ssd_total) / denom