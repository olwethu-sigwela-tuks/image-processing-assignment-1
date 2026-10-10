import os

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_harness import build_algorithm
from metrics import (
    load_grayscale_histogram,
    segment_image,
    compute_psnr,
    compute_ssim,
)
"""Python file to generate comparison images for the CHAOS MRI dataset"""
IMAGE_PATH = "CHAOS_DATA/IMG-0082-00010.png"
OBJECTIVE = "kapur"
K_VALUES = [5, 7]
ALGORITHMS = ["StandardDE", "LADE", "JADE", "SHADE", "LSHADE"]
SEED = 0
OUTDIR = "results/comparison_images"
SAVE_INDIVIDUAL = True
DPI = 150


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    gray_arr, hist_prob = load_grayscale_histogram(IMAGE_PATH)
    base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    n_rows = len(K_VALUES)
    n_cols = len(ALGORITHMS) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.3 * n_rows), squeeze=False)

    for r, K in enumerate(K_VALUES):
        axes[r][0].imshow(gray_arr, cmap="gray", vmin=0, vmax=255)
        axes[r][0].set_title(f"Original\n(K={K})", fontsize=10)
        axes[r][0].axis("off")

        for c, algo_name in enumerate(ALGORITHMS, start=1):
            algo = build_algorithm(algo_name, K, hist_prob, OBJECTIVE, SEED, {})
            best_thresholds, best_fitness, _ = algo.run()

            segmented = segment_image(gray_arr, best_thresholds)
            psnr = compute_psnr(gray_arr, segmented)
            ssim = compute_ssim(gray_arr, segmented)

            axes[r][c].imshow(segmented, cmap="gray", vmin=0, vmax=255)
            axes[r][c].set_title(
                f"{algo_name}\nPSNR={psnr:.2f} dB  SSIM={ssim:.3f}", fontsize=9
            )
            axes[r][c].axis("off")

            if SAVE_INDIVIDUAL:
                tag = f"{base}_{OBJECTIVE}_K{K}_{algo_name}"
                Image.fromarray(segmented).save(os.path.join(OUTDIR, f"{tag}_segmented.png"))

            print(
                f"K={K:2d} {algo_name:10s} PSNR={psnr:6.2f} SSIM={ssim:.4f} "
                f"thresholds={np.asarray(best_thresholds).tolist()}"
            )

    fig.suptitle(f"{base} - {OBJECTIVE} - seed {SEED}", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUTDIR, f"{base}_{OBJECTIVE}_comparison.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
