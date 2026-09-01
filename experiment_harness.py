"""
Experiment harness for the COS791 multilevel thresholding assignment.

Runs N independent repetitions of every (algorithm, image, objective, K)
combination under an EQUAL function-evaluation budget (MAX_FES), exactly
as required by the assignment's experimental protocol, and produces:

  1. results/raw_results.csv
       One row per individual run: algorithm, image, objective, K, seed,
       best thresholds, fitness, PSNR, SSIM, Uniformity, wall-clock time,
       FEs used. This is the file Wilcoxon/Friedman tests should be run
       against later (they need per-run, not aggregated, data).

  2. results/summary_table.csv
       Mean +/- std of PSNR/SSIM/Uniformity/time per (algorithm,
       objective, K) -- the same shape as Table 1 in the assignment brief.

  3. results/convergence/*.npy
       Per-run best-fitness-per-generation traces, for convergence-curve
       plots later.

The algorithm registry (ALGORITHMS dict below) is the single point where
JADE / SHADE / L-SHADE / LADE get plugged in later: every algorithm class
is expected to expose the same constructor signature and a `.run()` ->
(best_thresholds, best_fitness, history) contract that StandardDE already
follows, so nothing else in this file needs to change when they're added.

Usage:
    python experiment_harness.py \
        --image_dir sample_images \
        --algorithms StandardDE \
        --objectives otsu kapur tsallis \
        --K_values 3 5 7 \
        --n_runs 30 \
        --NP 50 --MAX_FES 10000 \
        --outdir results
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from standard_de import StandardDE
from metrics import (
    load_grayscale_histogram,
    segment_image,
    compute_psnr,
    compute_ssim,
    compute_uniformity,
)

# --------------------------------------------------------------------------
# Algorithm registry. Add JADE / SHADE / L-SHADE / LADE here as they're
# built -- each must accept the same kwargs as StandardDE and implement
# .run() -> (best_thresholds, best_fitness_natural, history).
# --------------------------------------------------------------------------
ALGORITHMS = {
    "StandardDE": StandardDE,
}

# Default per-objective kwargs (e.g. Tsallis' q parameter).
DEFAULT_OBJECTIVE_KWARGS = {
    "otsu": {},
    "kapur": {},
    "tsallis": {"q": 0.8},
}


def run_single(algo_name, image_path, hist_prob, gray_arr, K, objective,
                NP, MAX_FES, seed, objective_kwargs, bounds=(1, 254),
                F=0.5, CR=0.9):
    """Execute one independent run and return a flat result dict + history."""
    AlgoClass = ALGORITHMS[algo_name]

    algo_kwargs = dict(
        dim=K,
        bounds=bounds,
        hist_prob=hist_prob,
        objective_name=objective,
        NP=NP,
        MAX_FES=MAX_FES,
        seed=seed,
        objective_kwargs=objective_kwargs,
    )
    # F/CR only apply to Standard DE; adaptive variants (JADE/SHADE/...)
    # won't accept them, so only pass if the algorithm supports it.
    if algo_name == "StandardDE":
        algo_kwargs.update(F=F, CR=CR)

    algo = AlgoClass(**algo_kwargs)

    t0 = time.perf_counter()
    best_thresholds, best_fitness, history = algo.run()
    elapsed = time.perf_counter() - t0

    segmented = segment_image(gray_arr, best_thresholds)
    record = {
        "algorithm": algo_name,
        "image": os.path.basename(image_path),
        "objective": objective,
        "K": K,
        "seed": seed,
        "best_thresholds": ";".join(map(str, best_thresholds)),
        "fitness": best_fitness,
        "psnr": compute_psnr(gray_arr, segmented),
        "ssim": compute_ssim(gray_arr, segmented),
        "uniformity": compute_uniformity(gray_arr, best_thresholds),
        "time_sec": elapsed,
        "fes_used": algo.fes_used,
    }
    return record, history


def run_batch(image_paths, algorithms, objectives, K_values, n_runs=30,
              NP=50, MAX_FES=10000, seed_base=0, outdir="results",
              objective_kwargs_map=None, save_convergence=True,
              F=0.5, CR=0.9, verbose=True):
    """
    Full factorial sweep over (algorithm x image x objective x K), with
    `n_runs` independent repetitions of each combination (different seed
    per run, but the SAME seed sequence 0..n_runs-1 reused across every
    combination so comparisons stay paired for Wilcoxon tests later).
    """
    os.makedirs(outdir, exist_ok=True)
    conv_dir = os.path.join(outdir, "convergence")
    if save_convergence:
        os.makedirs(conv_dir, exist_ok=True)

    objective_kwargs_map = objective_kwargs_map or DEFAULT_OBJECTIVE_KWARGS

    # Histograms are expensive-ish and constant across (algo, objective, K,
    # seed) for a given image, so compute them once per image up front.
    image_cache = {}
    for path in image_paths:
        gray_arr, hist_prob = load_grayscale_histogram(path)
        image_cache[path] = (gray_arr, hist_prob)

    total_runs = (len(algorithms) * len(image_paths) * len(objectives)
                  * len(K_values) * n_runs)
    done = 0
    records = []
    t_start = time.time()

    for algo_name in algorithms:
        for image_path in image_paths:
            gray_arr, hist_prob = image_cache[image_path]
            for objective in objectives:
                obj_kwargs = objective_kwargs_map.get(objective, {})
                for K in K_values:
                    for run_idx in range(n_runs):
                        seed = seed_base + run_idx
                        record, history = run_single(
                            algo_name, image_path, hist_prob, gray_arr,
                            K, objective, NP, MAX_FES, seed, obj_kwargs,
                            F=F, CR=CR,
                        )
                        records.append(record)

                        if save_convergence:
                            tag = (f"{algo_name}_{record['image']}_"
                                   f"{objective}_K{K}_run{run_idx}")
                            np.save(os.path.join(conv_dir, f"{tag}.npy"),
                                    np.array(history, dtype=float))

                        done += 1
                        if verbose:
                            elapsed_total = time.time() - t_start
                            eta = (elapsed_total / done) * (total_runs - done)
                            print(
                                f"[{done}/{total_runs}] {algo_name} | "
                                f"{record['image']} | {objective} | K={K} | "
                                f"run={run_idx} -> PSNR={record['psnr']:.2f} "
                                f"SSIM={record['ssim']:.3f} "
                                f"U={record['uniformity']:.3f} "
                                f"t={record['time_sec']:.2f}s | "
                                f"ETA={eta/60:.1f} min"
                            )

    df = pd.DataFrame(records)
    raw_path = os.path.join(outdir, "raw_results.csv")
    df.to_csv(raw_path, index=False)
    if verbose:
        print(f"\nSaved {len(df)} raw run records to {raw_path}")

    summary = summarize(df, outdir=outdir, verbose=verbose)
    summary_by_image = summarize_by_image(df, outdir=outdir, verbose=verbose)
    return df, summary, summary_by_image


def summarize(df, outdir="results", verbose=True):
    """Aggregate raw per-run results into mean +/- std per (algo, objective, K)."""
    summary = (
        df.groupby(["algorithm", "objective", "K"])
        .agg(
            psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"),
            ssim_mean=("ssim", "mean"), ssim_std=("ssim", "std"),
            uniformity_mean=("uniformity", "mean"), uniformity_std=("uniformity", "std"),
            time_mean=("time_sec", "mean"), time_std=("time_sec", "std"),
            n_runs=("seed", "count"),
        )
        .reset_index()
        .sort_values(["algorithm", "objective", "K"])
    )
    summary_path = os.path.join(outdir, "summary_table.csv")
    summary.to_csv(summary_path, index=False)
    if verbose:
        print(f"Saved summary table to {summary_path}")
    return summary


def summarize_by_image(df, outdir="results", verbose=True):
    """
    Same aggregation as summarize(), but keeping `image` as a group key --
    useful for spotting any single image that behaves as an outlier before
    you aggregate it away in the Table-1-style summary.
    """
    summary = (
        df.groupby(["algorithm", "image", "objective", "K"])
        .agg(
            psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"),
            ssim_mean=("ssim", "mean"), ssim_std=("ssim", "std"),
            uniformity_mean=("uniformity", "mean"), uniformity_std=("uniformity", "std"),
            time_mean=("time_sec", "mean"),
            n_runs=("seed", "count"),
        )
        .reset_index()
        .sort_values(["algorithm", "image", "objective", "K"])
    )
    path = os.path.join(outdir, "summary_by_image.csv")
    summary.to_csv(path, index=False)
    if verbose:
        print(f"Saved per-image summary table to {path}")
    return summary


def discover_images(image_dir, extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(image_dir, f"*{ext.upper()}")))
    # exclude ground-truth mask files (common naming convention: *_gt.*)
    paths = [p for p in paths if "_gt" not in os.path.basename(p).lower()]
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True,
                         help="Folder of images (GT masks named *_gt.* are auto-excluded).")
    parser.add_argument("--algorithms", nargs="+", default=["StandardDE"],
                         choices=list(ALGORITHMS.keys()))
    parser.add_argument("--objectives", nargs="+", default=["otsu", "kapur", "tsallis"],
                         choices=["otsu", "kapur", "tsallis"])
    parser.add_argument("--K_values", nargs="+", type=int, default=[3, 5, 7, 9, 11, 12])
    parser.add_argument("--n_runs", type=int, default=30)
    parser.add_argument("--NP", type=int, default=50)
    parser.add_argument("--MAX_FES", type=int, default=10000)
    parser.add_argument("--F", type=float, default=0.5)
    parser.add_argument("--CR", type=float, default=0.9)
    parser.add_argument("--q", type=float, default=0.8, help="Tsallis entropy parameter")
    parser.add_argument("--seed_base", type=int, default=0)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--no_convergence", action="store_true",
                         help="Skip saving per-run convergence traces (saves disk/time).")
    args = parser.parse_args()

    image_paths = discover_images(args.image_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {args.image_dir}")
    print(f"Found {len(image_paths)} image(s) in {args.image_dir}: "
          f"{[os.path.basename(p) for p in image_paths]}")

    objective_kwargs_map = dict(DEFAULT_OBJECTIVE_KWARGS)
    objective_kwargs_map["tsallis"] = {"q": args.q}

    run_batch(
        image_paths=image_paths,
        algorithms=args.algorithms,
        objectives=args.objectives,
        K_values=args.K_values,
        n_runs=args.n_runs,
        NP=args.NP,
        MAX_FES=args.MAX_FES,
        seed_base=args.seed_base,
        outdir=args.outdir,
        objective_kwargs_map=objective_kwargs_map,
        save_convergence=not args.no_convergence,
        F=args.F,
        CR=args.CR,
    )


if __name__ == "__main__":
    main()
