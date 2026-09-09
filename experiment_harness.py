"""
Experiment harness for the COS791 multilevel thresholding assignment.

WHAT THIS PRODUCES (under each phase's own --outdir):
  1. raw_results.csv
       One row per individual run: algorithm, image, objective, K, seed,
       best thresholds, fitness, PSNR, SSIM, Uniformity, wall-clock time,
       FEs used. This is the file Wilcoxon/Friedman tests should run
       against later (they need per-run, not aggregated, data).
  2. summary_table.csv
       Mean +/- std of PSNR/SSIM/Uniformity/time per (algorithm,
       objective, K)
  3. summary_by_image.csv
       Same aggregation but keeping `image` as a group key, to help spot
       an outlier image before it's averaged away.
  4. convergence/*.npy
       Per-run best-fitness-per-generation traces, for convergence plots.

"""

import functools
import glob
import os
import time

import numpy as np
import pandas as pd

from standard_de import StandardDE
from lade import LADE
from jade import JADE
import shade_imageprocessing as _shade_mod
import l_shade_imageprocessing as _lshade_mod
from metrics import (
    load_grayscale_histogram,
    segment_image,
    compute_psnr,
    compute_ssim,
    compute_uniformity,
)


# ==========================================================================
# CONFIG
# ==========================================================================
ALGORITHMS_TO_RUN = ["StandardDE", "LADE", "JADE", "SHADE", "LSHADE"]
OBJECTIVES = ["otsu", "kapur", "tsallis"]
K_VALUES = [3, 5, 7, 9, 11, 12]
N_RUNS = 30
NP = 50
MAX_FES = 10000
SEED_BASE = 0
BOUNDS = (1, 254)
SAVE_CONVERGENCE = True

# Algorithm-specific hyperparameters
F = 0.5                    # StandardDE / LADE mutation scale factor
CR = 0.9                   # StandardDE / LADE crossover probability
LATE_ACCEPTANCE_L = 20     # LADE fitness-history buffer length
JADE_C = 0.1                # JADE adaptation rate
JADE_P = 0.05                # JADE pbest fraction
TSALLIS_Q = 0.8             # Tsallis entropy parameter

# Dataset folders
PHASE1_DIR = "BDS500"
PHASE1_OUTDIR = "results/phase1_bsd500"
PHASE2_DIR = "CHAOS_DATA"
PHASE2_OUTDIR = "results/phase2_chaos"


# --------------------------------------------------------------------------
# Adapters for shade_imageprocessing.shade() and l_shade_imageprocessing
# --------------------------------------------------------------------------
class _FESBudgetExhausted(Exception):
    """Internal signal used to interrupt shade()/lshade() once MAX_FES
    evaluations have been spent."""


class _CountingObjective:
    """
    Wraps one of shade_imageprocessing.py's / l_shade_imageprocessing.py's
    own objective functions (otsu_objective / kapur_objective /
    tsallis_objective)
    """

    def __init__(self, base_fn, max_fes):
        self.base_fn = base_fn
        self.max_fes = max_fes
        self.fes_used = 0
        self.best_val = np.inf
        self.best_vec = None
        self.history = []

    def __call__(self, vec, hist):
        if self.fes_used >= self.max_fes:
            raise _FESBudgetExhausted()
        self.fes_used += 1
        val = self.base_fn(vec, hist)
        if val < self.best_val:
            self.best_val = val
            self.best_vec = np.array(vec, dtype=float, copy=True)
        self.history.append(-self.best_val)
        return val


_OBJ_NAME_MAP = {"otsu": "Otsu", "kapur": "Kapur", "tsallis": "Tsallis"}


class SHADEAdapter:
    """Wraps shade_imageprocessing.shade()"""

    def __init__(self, dim, bounds, hist_prob, objective_name, NP, MAX_FES,
                 seed=None, objective_kwargs=None):
        self.dim = dim
        self.hist_prob = hist_prob
        self.objective_name = objective_name
        self.NP = NP
        self.MAX_FES = MAX_FES
        self.seed = seed
        self.objective_kwargs = objective_kwargs or {}
        self.fes_used = 0
        self.history = []

    def run(self, verbose=False):
        if self.seed is not None:
            np.random.seed(self.seed)

        base_fn = _shade_mod.objective_functions[_OBJ_NAME_MAP[self.objective_name]]
        if self.objective_kwargs:
            base_fn = functools.partial(base_fn, **self.objective_kwargs)
        tracker = _CountingObjective(base_fn, self.MAX_FES)

        max_gen = max(1, self.MAX_FES // max(1, self.NP))

        try:
            _shade_mod.shade(self.hist_prob, self.dim, tracker,
                              pop_size=self.NP, max_gen=max_gen)
        except _FESBudgetExhausted:
            pass

        self.fes_used = tracker.fes_used
        self.history = tracker.history or [-tracker.best_val]

        best_thresholds = np.sort(np.round(np.clip(tracker.best_vec, 1, 255)).astype(int))
        return best_thresholds, -tracker.best_val, self.history


class LSHADEAdapter:
    """Wraps l_shade_imageprocessing.lshade()"""

    def __init__(self, dim, bounds, hist_prob, objective_name, NP, MAX_FES,
                 seed=None, objective_kwargs=None):
        self.dim = dim
        self.hist_prob = hist_prob
        self.objective_name = objective_name
        self.NP = NP
        self.MAX_FES = MAX_FES
        self.seed = seed
        self.objective_kwargs = objective_kwargs or {}
        self.fes_used = 0
        self.history = []

    def run(self, verbose=False):
        if self.seed is not None:
            np.random.seed(self.seed)

        base_fn = _lshade_mod.objective_functions[_OBJ_NAME_MAP[self.objective_name]]
        if self.objective_kwargs:
            base_fn = functools.partial(base_fn, **self.objective_kwargs)
        tracker = _CountingObjective(base_fn, self.MAX_FES)

        N_min = 4
        avg_pop = (self.NP + N_min) / 2.0
        max_gen = max(1, int(self.MAX_FES / avg_pop))

        try:
            _lshade_mod.lshade(self.hist_prob, self.dim, tracker,
                                pop_size=self.NP, max_gen=max_gen)
        except _FESBudgetExhausted:
            pass

        self.fes_used = tracker.fes_used
        self.history = tracker.history or [-tracker.best_val]

        best_thresholds = np.sort(np.round(np.clip(tracker.best_vec, 1, 255)).astype(int))
        return best_thresholds, -tracker.best_val, self.history


# --------------------------------------------------------------------------
# Per-algorithm construction
# --------------------------------------------------------------------------
def build_algorithm(algo_name, dim, hist_prob, objective_name, seed, objective_kwargs):
    if algo_name == "StandardDE":
        return StandardDE(
            dim=dim, bounds=BOUNDS, hist_prob=hist_prob, objective_name=objective_name,
            NP=NP, MAX_FES=MAX_FES, F=F, CR=CR, seed=seed, objective_kwargs=objective_kwargs,
        )
    elif algo_name == "LADE":
        return LADE(
            dim=dim, bounds=BOUNDS, hist_prob=hist_prob, objective_name=objective_name,
            NP=NP, MAX_FES=MAX_FES, F=F, CR=CR, seed=seed, objective_kwargs=objective_kwargs,
            L_a=LATE_ACCEPTANCE_L,
        )
    elif algo_name == "JADE":
        return JADE(
            dim=dim, bounds=BOUNDS, hist_prob=hist_prob, objective_name=objective_name,
            NP=NP, MAX_FES=MAX_FES, seed=seed, objective_kwargs=objective_kwargs,
            c=JADE_C, p=JADE_P,
        )
    elif algo_name == "SHADE":
        return SHADEAdapter(
            dim=dim, bounds=BOUNDS, hist_prob=hist_prob, objective_name=objective_name,
            NP=NP, MAX_FES=MAX_FES, seed=seed, objective_kwargs=objective_kwargs,
        )
    elif algo_name == "LSHADE":
        return LSHADEAdapter(
            dim=dim, bounds=BOUNDS, hist_prob=hist_prob, objective_name=objective_name,
            NP=NP, MAX_FES=MAX_FES, seed=seed, objective_kwargs=objective_kwargs,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


RAW_RESULTS_COLUMNS = [
    "algorithm", "image", "objective", "K", "seed",
    "best_thresholds", "fitness", "psnr", "ssim", "uniformity",
    "time_sec", "fes_used",
]


def run_single(algo_name, image_path, hist_prob, gray_arr, K, objective, seed):
    """Execute one independent run and return a flat result dict + history."""
    objective_kwargs = {"q": TSALLIS_Q} if objective == "tsallis" else {}
    algo = build_algorithm(algo_name, K, hist_prob, objective, seed, objective_kwargs)

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
    run_phase() can skip them. Returns an empty set if the file doesn't
    exist yet.
    """
    if not os.path.exists(raw_path):
        return set()
    existing = pd.read_csv(raw_path)
    return set(zip(
        existing["algorithm"], existing["image"], existing["objective"],
        existing["K"], existing["seed"],
    ))


def discover_images(image_dir, extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(image_dir, f"*{ext.upper()}")))
    # exclude ground-truth mask files (naming convention: *_gt.*)
    paths = [p for p in paths if "_gt" not in os.path.basename(p).lower()]
    return sorted(set(paths))


def run_phase(image_dir, outdir):
    """
    Full factorial sweep over (algorithm x image x objective x K)
    """
    image_paths = discover_images(image_dir)
    if not image_paths:
        print(f"No images found in {image_dir}, skipping this phase.")
        return None

    print(f"Found {len(image_paths)} image(s) in {image_dir}: "
          f"{[os.path.basename(p) for p in image_paths]}")

    os.makedirs(outdir, exist_ok=True)
    conv_dir = os.path.join(outdir, "convergence")
    if SAVE_CONVERGENCE:
        os.makedirs(conv_dir, exist_ok=True)

    raw_path = os.path.join(outdir, "raw_results.csv")
    completed = load_completed_runs(raw_path)
    if completed:
        print(f"Resuming: found {len(completed)} previously-completed runs "
              f"in {raw_path}, these will be skipped.")

    # Histograms are constant across (algo, objective, K, seed) for a given image, so compute them once per image up front.
    image_cache = {p: load_grayscale_histogram(p) for p in image_paths}

    planned = []
    for algo_name in ALGORITHMS_TO_RUN:
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            for objective in OBJECTIVES:
                for K in K_VALUES:
                    for run_idx in range(N_RUNS):
                        seed = SEED_BASE + run_idx
                        key = (algo_name, image_name, objective, K, seed)
                        if key not in completed:
                            planned.append((algo_name, image_path, objective, K, seed))

    total_runs = len(planned)
    total_possible = (len(ALGORITHMS_TO_RUN) * len(image_paths) * len(OBJECTIVES)
                       * len(K_VALUES) * N_RUNS)
    print(f"Planned: {total_runs} runs to execute "
          f"({total_possible - total_runs} already completed and skipped).")

    file_exists = os.path.exists(raw_path)
    skipped_algorithms = set()
    done = 0
    t_start = time.time()

    for algo_name, image_path, objective, K, seed in planned:
        if algo_name in skipped_algorithms:
            continue

        gray_arr, hist_prob = image_cache[image_path]

        try:
            record, history = run_single(algo_name, image_path, hist_prob, gray_arr, K, objective, seed)
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

        if SAVE_CONVERGENCE:
            tag = f"{algo_name}_{record['image']}_{objective}_K{K}_run{seed}"
            np.save(os.path.join(conv_dir, f"{tag}.npy"), np.array(history, dtype=float))

        done += 1
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
        print("No runs were completed. Nothing to summarise.")
        return None

    df = pd.read_csv(raw_path)
    print(f"\n{raw_path} now has {len(df)} total run records.")
    if skipped_algorithms:
        print(f"Algorithms skipped this run (not yet implemented): {sorted(skipped_algorithms)}")

    summarize(df, outdir)
    summarize_by_image(df, outdir)
    return df


def summarize(df, outdir):
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
    path = os.path.join(outdir, "summary_table.csv")
    summary.to_csv(path, index=False)
    print(f"Saved summary table to {path}")
    return summary


def summarize_by_image(df, outdir):
    """
    Same aggregation as summarize(), but keeping `image` as a group key
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
    print(f"Saved per-image summary table to {path}")
    return summary


def main():
    print("=== Phase 1: BSD500 (proof-of-concept) ===")
    run_phase(PHASE1_DIR, PHASE1_OUTDIR)

    print("\n=== Phase 2: CHAOS MRI (domain application) ===")
    run_phase(PHASE2_DIR, PHASE2_OUTDIR)


if __name__ == "__main__":
    main()
