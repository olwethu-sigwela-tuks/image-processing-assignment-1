"""
End-to-end experiment driver: image -> histogram -> Standard DE -> segmented
image + convergence curve.

Usage (example):
    python run_experiment.py --image img9.png --K 5 --objective kapur \
        --NP 50 --MAX_FES 10000 --seed 0
"""

import argparse
import os

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from standard_de import StandardDE


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
    supplied threshold boundaries. thresholds must be sorted ints.
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


def psnr(original, segmented):
    mse = np.mean((original.astype(float) - segmented.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--objective", choices=["otsu", "kapur", "tsallis"], default="kapur")
    parser.add_argument("--q", type=float, default=0.8, help="Tsallis entropy parameter")
    parser.add_argument("--NP", type=int, default=50)
    parser.add_argument("--MAX_FES", type=int, default=10000)
    parser.add_argument("--F", type=float, default=0.5)
    parser.add_argument("--CR", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    gray_arr, hist_prob = load_grayscale_histogram(args.image)

    objective_kwargs = {"q": args.q} if args.objective == "tsallis" else {}

    de = StandardDE(
        dim=args.K,
        bounds=(1, 254),
        hist_prob=hist_prob,
        objective_name=args.objective,
        NP=args.NP,
        MAX_FES=args.MAX_FES,
        F=args.F,
        CR=args.CR,
        seed=args.seed,
        objective_kwargs=objective_kwargs,
    )

    best_thresholds, best_fitness, history = de.run(verbose=True)
    print(f"\nBest thresholds (K={args.K}, obj={args.objective}): {best_thresholds}")
    print(f"Best fitness ({args.objective}): {best_fitness:.6f}")

    segmented = segment_image(gray_arr, best_thresholds)
    score_psnr = psnr(gray_arr, segmented)
    print(f"PSNR (reconstruction vs original): {score_psnr:.2f} dB")

    base = os.path.splitext(os.path.basename(args.image))[0]
    tag = f"{base}_K{args.K}_{args.objective}"

    # -- save segmented image --
    Image.fromarray(segmented).save(os.path.join(args.outdir, f"{tag}_segmented.png"))

    # -- side-by-side comparison figure --
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(gray_arr, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(segmented, cmap="gray")
    axes[1].set_title(f"Segmented (K={args.K}, {args.objective})\nPSNR={score_psnr:.2f} dB")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"{tag}_comparison.png"), dpi=150)
    plt.close(fig)

    # -- convergence curve --
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(history)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel(f"Best fitness ({args.objective})")
    ax2.set_title(f"Convergence: Standard DE, K={args.K}, {args.objective}")
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, f"{tag}_convergence.png"), dpi=150)
    plt.close(fig2)

    print(f"\nSaved outputs to {args.outdir}/{tag}_*.png")


if __name__ == "__main__":
    main()
