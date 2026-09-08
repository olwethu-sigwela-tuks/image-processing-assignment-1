"""
Experiment harness for the COS791 multilevel thresholding assignment.

This is the PLUG-IN SKELETON: every DE variant (Standard DE, JADE, SHADE,
L-SHADE, LADE) is registered once in ALGORITHMS below, and everything else
-- the 30-independent-run sweep, equal-FEs comparability, CSV logging,
resuming a crashed/partial run, and summary-table generation -- is
generic and never needs to change when a new algorithm is added.

All five algorithms are implemented as of this version:
  - StandardDE, LADE, JADE were built directly against the shared
    contract (de_base.py) and needed no changes to integrate.
  - SHADE, L-SHADE were originally written as standalone notebook
    scripts with their own objective functions, their own metrics, and
    generation-count stopping instead of an FES budget -- they were
    ported into this contract (mutation/archive/memory-adaptation logic
    preserved from the original) so every algorithm's fitness values,
    metrics, and evaluation budgets are computed identically and are
    therefore actually comparable. See the top of shade.py / lshade.py
    for the specifics of what changed and why.

See de_base.py for the exact contract a class must satisfy to be
registered here.

WHAT THIS PRODUCES (all under --outdir):
  1. raw_results.csv
       One row per individual run: algorithm, image, objective, K, seed,
       best thresholds, fitness, PSNR, SSIM, Uniformity, wall-clock time,
       FEs used. This is the file Wilcoxon/Friedman tests should be run
       against later (they need per-run, not aggregated, data). Written
       INCREMENTALLY (one row appended per completed run), so a crash or
       Ctrl-C partway through a long sweep does not lose completed work.
  2. summary_table.csv
       Mean +/- std of PSNR/SSIM/Uniformity/time per (algorithm,
       objective, K) -- the same shape as Table 1 in the assignment brief.
  3. summary_by_image.csv
       Same aggregation but keeping `image` as a group key, to help spot
       an outlier image before it's averaged away.
  4. convergence/*.npy
       Per-run best-fitness-per-generation traces, for convergence plots.

RESUMING: if raw_results.csv already exists in --outdir, the harness
loads it and SKIPS any (algorithm, image, objective, K, seed) combination
already present. This means you can: (a) re-run the same command after a
crash and only the missing runs execute, and (b) add a newly-implemented
algorithm to --algorithms and re-run without repeating everything you
already have for the others.

Usage examples:
    # Quick sanity check of what a full sweep would run, without running it
    python experiment_harness.py --phase phase1 --algorithms StandardDE LADE \\
        --dry_run

    # Full Phase 1 (BSD500) protocol run for Standard DE + LADE
    python experiment_harness.py --phase phase1 \\
        --algorithms StandardDE LADE \\
        --objectives otsu kapur tsallis \\
        --K_values 3 5 7 9 11 12 \\
        --n_runs 30 --NP 50 --MAX_FES 10000

    # Same, but for Phase 2 (CHAOS MRI)
    python experiment_harness.py --phase phase2 --algorithms StandardDE LADE

    # Regenerate summary tables only, from an existing raw_results.csv
    python experiment_harness.py --outdir results/phase1_bsd500 --summarize_only
"""

import argparse
import glob
import inspect
import os
import time

import numpy as np
import pandas as pd

from standard_de import StandardDE
from lade import LADE
from jade import JADE
from shade import SHADE
from lshade import LSHADE
from metrics import (
    load_grayscale_histogram,
    segment_image,
    compute_psnr,
    compute_ssim,
    compute_uniformity,
)

# --------------------------------------------------------------------------
# ALGORITHM REGISTRY -- this is the one dict you touch to plug in a new
# variant. Every entry must satisfy the contract in de_base.py.
# --------------------------------------------------------------------------
ALGORITHMS = {
    "StandardDE": StandardDE,
    "LADE": LADE,
    "JADE": JADE,
    "SHADE": SHADE,
    "LSHADE": LSHADE,
}

# Convenience mapping for --phase, matching the actual folder names in the
# assignment repo.
PHASE_DIRS = {
    "phase1": "BDS500",
    "phase2": "CHAOS_DATA",
}

DEFAULT_OBJECTIVE_KWARGS = {
    "otsu": {},
    "kapur": {},
    "tsallis": {"q": 0.8},
}

RAW_RESULTS_COLUMNS = [
    "algorithm", "image", "objective", "K", "seed",
    "best_thresholds", "fitness", "psnr", "ssim", "uniformity",
    "time_sec", "fes_used",
]


# --------------------------------------------------------------------------
# Generic kwargs forwarding -- build_algo_kwargs() inspects the target
# class's constructor and only forwards the extra kwargs it actually
# declares, so a new algorithm's own hyperparameters never require edits
# to run_single()/run_batch(). extra_kwargs itself is a dict keyed by
# algorithm name (see main()'s ALGO_EXTRA_KWARGS below) rather than one
# flat dict shared by everyone -- this matters because different
# algorithms reuse the same parameter NAME for different things (e.g.
# both JADE and SHADE/L-SHADE have a "p" = pbest-fraction parameter, but
# with different CLI defaults), so keeping them namespaced by algorithm
# avoids one silently overwriting the other.
# --------------------------------------------------------------------------
def build_algo_kwargs(AlgoClass, base_kwargs, extra_kwargs):
    sig = inspect.signature(AlgoClass.__init__)
    accepted = set(sig.parameters.keys())
    kwargs = dict(base_kwargs)
    for k, v in extra_kwargs.items():
        if k in accepted:
            kwargs[k] = v
    return kwargs


def run_single(algo_name, image_path, hist_prob, gray_arr, K, objective,
                NP, MAX_FES, seed, objective_kwargs, bounds, extra_kwargs):
    """Execute one independent run and return a flat result dict + history."""
    AlgoClass = ALGORITHMS[algo_name]

    base_kwargs = dict(
        dim=K,
        bounds=bounds,
        hist_prob=hist_prob,
        objective_name=objective,
        NP=NP,
        MAX_FES=MAX_FES,
        seed=seed,
        objective_kwargs=objective_kwargs,
    )
    algo_kwargs = build_algo_kwargs(AlgoClass, base_kwargs, extra_kwargs.get(algo_name, {}))
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


def load_completed_runs(raw_path):
    """
    Read an existing raw_results.csv (if any) and return the set of
    (algorithm, image, objective, K, seed) tuples already completed, so
    run_batch() can skip them. Returns an empty set if the file doesn't
    exist yet.
    """
    if not os.path.exists(raw_path):
        return set()
    existing = pd.read_csv(raw_path)
    keys = set(
        zip(
            existing["algorithm"], existing["image"], existing["objective"],
            existing["K"], existing["seed"],
        )
    )
    return keys


def run_batch(image_paths, algorithms, objectives, K_values, n_runs=30,
              NP=50, MAX_FES=10000, seed_base=0, outdir="results",
              objective_kwargs_map=None, save_convergence=True,
              bounds=(1, 254), extra_kwargs=None, verbose=True):
    """
    Full factorial sweep over (algorithm x image x objective x K), with
    `n_runs` independent repetitions of each combination. Resumable: any
    combination already present in outdir/raw_results.csv is skipped.
    Results are appended to the CSV as each run completes (crash-safe).

    extra_kwargs : dict[str, dict] or None
        Per-algorithm hyperparameters, e.g.
        {"StandardDE": {"F": 0.5, "CR": 0.9}, "LADE": {"F": 0.5, "CR": 0.9, "L": 20}, ...}
        Only the keys present in a given algorithm's own dict (and only
        the ones that class's __init__ actually declares) are forwarded
        to it -- see build_algo_kwargs().
    """
    os.makedirs(outdir, exist_ok=True)
    conv_dir = os.path.join(outdir, "convergence")
    if save_convergence:
        os.makedirs(conv_dir, exist_ok=True)

    objective_kwargs_map = objective_kwargs_map or DEFAULT_OBJECTIVE_KWARGS
    extra_kwargs = extra_kwargs or {}

    raw_path = os.path.join(outdir, "raw_results.csv")
    completed = load_completed_runs(raw_path)
    if completed and verbose:
        print(f"Resuming: found {len(completed)} previously-completed runs "
              f"in {raw_path}, these will be skipped.")

    # Histograms are constant across (algo, objective, K, seed) for a given
    # image, so compute them once per image up front.
    image_cache = {}
    for path in image_paths:
        gray_arr, hist_prob = load_grayscale_histogram(path)
        image_cache[path] = (gray_arr, hist_prob)

    planned = []
    for algo_name in algorithms:
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            for objective in objectives:
                for K in K_values:
                    for run_idx in range(n_runs):
                        seed = seed_base + run_idx
                        key = (algo_name, image_name, objective, K, seed)
                        if key not in completed:
                            planned.append((algo_name, image_path, objective, K, seed))

    total_runs = len(planned)
    if verbose:
        skipped = (len(algorithms) * len(image_paths) * len(objectives)
                   * len(K_values) * n_runs) - total_runs
        print(f"Planned: {total_runs} runs to execute "
              f"({skipped} already completed and skipped).")

    file_exists = os.path.exists(raw_path)
    skipped_algorithms = set()
    done = 0
    t_start = time.time()

    for algo_name, image_path, objective, K, seed in planned:
        if algo_name in skipped_algorithms:
            continue

        gray_arr, hist_prob = image_cache[image_path]
        obj_kwargs = objective_kwargs_map.get(objective, {})

        try:
            record, history = run_single(
                algo_name, image_path, hist_prob, gray_arr, K, objective,
                NP, MAX_FES, seed, obj_kwargs, bounds, extra_kwargs,
            )
        except NotImplementedError as e:
            print(f"\n[SKIPPING '{algo_name}'] not yet implemented: {e}\n"
                  f"All remaining planned runs for '{algo_name}' will be skipped.\n")
            skipped_algorithms.add(algo_name)
            continue
        except Exception as e:
            print(f"[ERROR] {algo_name} | {os.path.basename(image_path)} | "
                  f"{objective} | K={K} | seed={seed} -> {type(e).__name__}: {e}. "
                  f"Skipping this run.")
            continue

        # Append this single run to the CSV immediately -- crash-safe.
        row_df = pd.DataFrame([record], columns=RAW_RESULTS_COLUMNS)
        row_df.to_csv(raw_path, mode="a", header=not file_exists, index=False)
        file_exists = True

        if save_convergence:
            tag = f"{algo_name}_{record['image']}_{objective}_K{K}_run{seed}"
            np.save(os.path.join(conv_dir, f"{tag}.npy"), np.array(history, dtype=float))

        done += 1
        if verbose:
            elapsed_total = time.time() - t_start
            eta = (elapsed_total / done) * (total_runs - done) if done else 0
            print(
                f"[{done}/{total_runs}] {algo_name} | {record['image']} | "
                f"{objective} | K={K} | seed={seed} -> "
                f"PSNR={record['psnr']:.2f} SSIM={record['ssim']:.3f} "
                f"U={record['uniformity']:.3f} t={record['time_sec']:.2f}s | "
                f"ETA={eta/60:.1f} min"
            )

    if not os.path.exists(raw_path):
        print("No runs were completed (nothing planned, or every algorithm "
              "requested is unimplemented). Nothing to summarise.")
        return None, None, None

    df = pd.read_csv(raw_path)
    if verbose:
        print(f"\n{raw_path} now has {len(df)} total run records.")
        if skipped_algorithms:
            print(f"Algorithms skipped this run (not yet implemented): "
                  f"{sorted(skipped_algorithms)}")

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
    # exclude ground-truth mask files (naming convention: *_gt.*)
    paths = [p for p in paths if "_gt" not in os.path.basename(p).lower()]
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default=None,
                         help="Folder of images (GT masks named *_gt.* are auto-excluded). "
                              "Overrides --phase if both are given.")
    parser.add_argument("--phase", choices=list(PHASE_DIRS.keys()), default=None,
                         help=f"Shortcut for the assignment's dataset folders: "
                              f"{PHASE_DIRS}. Ignored if --image_dir is given.")
    parser.add_argument("--algorithms", nargs="+", default=["StandardDE"],
                         choices=list(ALGORITHMS.keys()))
    parser.add_argument("--objectives", nargs="+", default=["otsu", "kapur", "tsallis"],
                         choices=["otsu", "kapur", "tsallis"])
    parser.add_argument("--K_values", nargs="+", type=int, default=[3, 5, 7, 9, 11, 12])
    parser.add_argument("--n_runs", type=int, default=30)
    parser.add_argument("--NP", type=int, default=50)
    parser.add_argument("--MAX_FES", type=int, default=10000)
    parser.add_argument("--seed_base", type=int, default=0)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--no_convergence", action="store_true",
                         help="Skip saving per-run convergence traces (saves disk/time).")
    parser.add_argument("--dry_run", action="store_true",
                         help="Print the planned run count and exit without executing anything.")
    parser.add_argument("--summarize_only", action="store_true",
                         help="Skip running anything; just regenerate summary tables from "
                              "an existing outdir/raw_results.csv.")

    # Standard DE / LADE hyperparameters (silently ignored by algorithms
    # that don't declare a matching constructor parameter).
    parser.add_argument("--F", type=float, default=0.5, help="Standard DE / LADE mutation scale factor")
    parser.add_argument("--CR", type=float, default=0.9, help="Standard DE / LADE crossover probability")
    parser.add_argument("--q", type=float, default=0.8, help="Tsallis entropy parameter")
    parser.add_argument("--late_acceptance_L", type=int, default=20,
                         help="LADE fitness-history buffer length (maps to LateAcceptanceDE's L kwarg)")
    parser.add_argument("--jade_c", type=float, default=0.1, help="JADE adaptation rate c")
    parser.add_argument("--jade_p", type=float, default=0.05, help="JADE pbest fraction p")
    parser.add_argument("--shade_H", type=int, default=10, help="SHADE / L-SHADE memory size H")
    parser.add_argument("--shade_p", type=float, default=0.1, help="SHADE / L-SHADE pbest fraction p")
    parser.add_argument("--lshade_Nmin", type=int, default=4, help="L-SHADE minimum population size")

    args = parser.parse_args()

    if args.summarize_only:
        raw_path = os.path.join(args.outdir, "raw_results.csv")
        if not os.path.exists(raw_path):
            raise SystemExit(f"--summarize_only given but {raw_path} does not exist.")
        df = pd.read_csv(raw_path)
        summarize(df, outdir=args.outdir)
        summarize_by_image(df, outdir=args.outdir)
        return

    if args.image_dir:
        image_dir = args.image_dir
    elif args.phase:
        image_dir = PHASE_DIRS[args.phase]
    else:
        raise SystemExit("Provide either --image_dir or --phase.")

    image_paths = discover_images(image_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {image_dir}")

    objective_kwargs_map = dict(DEFAULT_OBJECTIVE_KWARGS)
    objective_kwargs_map["tsallis"] = {"q": args.q}

    extra_kwargs = dict(
        StandardDE=dict(F=args.F, CR=args.CR),
        LADE=dict(F=args.F, CR=args.CR, L_a=args.late_acceptance_L),
        JADE=dict(c=args.jade_c, p=args.jade_p),
        SHADE=dict(H=args.shade_H, p_max=args.shade_p),
        LSHADE=dict(H=args.shade_H, p_max=args.shade_p, N_min=args.lshade_Nmin),
    )

    if args.dry_run:
        n_combos = (len(args.algorithms) * len(image_paths) * len(args.objectives)
                    * len(args.K_values) * args.n_runs)
        print(f"Images found ({len(image_paths)}): "
              f"{[os.path.basename(p) for p in image_paths]}")
        print(f"Algorithms: {args.algorithms}")
        print(f"Objectives: {args.objectives}")
        print(f"K values:   {args.K_values}")
        print(f"n_runs:     {args.n_runs}")
        print(f"\nTotal planned runs: {n_combos}")
        print("(Dry run -- nothing executed. Remove --dry_run to run it for real.)")
        return

    print(f"Found {len(image_paths)} image(s) in {image_dir}: "
          f"{[os.path.basename(p) for p in image_paths]}")

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
        extra_kwargs=extra_kwargs,
    )


if __name__ == "__main__":
    main()
