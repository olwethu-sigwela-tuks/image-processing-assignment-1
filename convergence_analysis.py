"""
Convergence-curve plotting: compares DE, JADE, SHADE, L-SHADE, LADE's best-fitness-so-far trajectories
on a common function-evaluations (FES) x-axis.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONV_DIR = "results/phase1_bsd500/convergence"
RAW_RESULTS = "results/phase1_bsd500/raw_results.csv"
IMAGE = "img1.png"
OBJECTIVE = "kapur"
K = 5
ALGORITHMS = ["StandardDE", "LADE", "JADE", "SHADE", "LSHADE"]
OUTDIR = "results/phase1_bsd500"


def _reconstruct_fes_axis(history_len, fes_used):
    """Approximate the FES value at each point in a history array."""
    if history_len <= 1:
        return np.array([fes_used])
    return np.linspace(fes_used / history_len, fes_used, history_len)


def load_convergence_curves(conv_dir, raw_results_csv, algorithm, image, objective, K):
    """
    Load every matching convergence trace for one (algorithm, image, objective, K) combination (one file per seed/run), each resampled
    onto a common FES grid, ready to be averaged.
    Returns (fes_grid, curves) where curves is a 2D array
    [n_runs x len(fes_grid)], or (None, None) if nothing matched.
    """
    df = pd.read_csv(raw_results_csv)
    df = df[(df.algorithm == algorithm) & (df.image == image)
            & (df.objective == objective) & (df.K == K)]
    if df.empty:
        return None, None

    max_fes = int(df["fes_used"].max())
    fes_grid = np.linspace(1, max_fes, 200)

    curves = []
    for _, row in df.iterrows():
        fname = f"{algorithm}_{image}_{objective}_K{K}_run{row['seed']}.npy"
        fpath = os.path.join(conv_dir, fname)
        if not os.path.exists(fpath):
            continue
        history = np.load(fpath)
        fes_axis = _reconstruct_fes_axis(len(history), row["fes_used"])
        resampled = np.interp(fes_grid, fes_axis, history,
                               left=history[0], right=history[-1])
        curves.append(resampled)

    if not curves:
        return None, None
    return fes_grid, np.array(curves)


def plot_algorithm_comparison(conv_dir, raw_results_csv, image, objective, K, algorithms, outpath):
    """One figure: mean +/- std band convergence curve per algorithm, for a fixed (image, objective, K)"""
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted_any = False

    for algo in algorithms:
        fes_grid, curves = load_convergence_curves(
            conv_dir, raw_results_csv, algo, image, objective, K)
        if fes_grid is None:
            print(f"  (skipping {algo}: no convergence data found for this combination)")
            continue

        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        ax.plot(fes_grid, mean_curve, label=f"{algo} (n={len(curves)})")
        ax.fill_between(fes_grid, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("Nothing to plot - no convergence data found for any requested algorithm.")
        return

    ax.set_xlabel("Function Evaluations (FES)")
    ax.set_ylabel(f"Best fitness ({objective})")
    ax.set_title(f"Convergence comparison -- {image}, {objective}, K={K}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    outpath = os.path.join(OUTDIR, f"convergence_{IMAGE}_{OBJECTIVE}_K{K}.png")
    plot_algorithm_comparison(CONV_DIR, RAW_RESULTS, IMAGE, OBJECTIVE, K, ALGORITHMS, outpath)


if __name__ == "__main__":
    main()
