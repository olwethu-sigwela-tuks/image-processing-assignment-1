"""
Computational time vs K scalability analysis.
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_RESULTS = "results/phase1_bsd500/raw_results.csv"
OUTDIR = "results/phase1_bsd500"


def scalability_table(df):
    """Mean +/- std time per (algorithm, K), averaged over all images/objectives/seeds"""
    table = (
        df.groupby(["algorithm", "K"])
        .agg(time_mean=("time_sec", "mean"), time_std=("time_sec", "std"),
             fes_used_mean=("fes_used", "mean"), n_runs=("seed", "count"))
        .reset_index()
        .sort_values(["algorithm", "K"])
    )
    return table


def plot_scalability(df, outpath):
    """One line per algorithm: mean time vs K, with error bars."""
    table = scalability_table(df)
    fig, ax = plt.subplots(figsize=(7, 5))

    for algo, group in table.groupby("algorithm"):
        group = group.sort_values("K")
        ax.errorbar(group["K"], group["time_mean"], yerr=group["time_std"],
                    marker="o", capsize=3, label=algo)

    ax.set_xlabel("K (number of thresholds)")
    ax.set_ylabel("Wall-clock time per run (s)")
    ax.set_title("Computational time vs K (scalability)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")
    return table


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(RAW_RESULTS)
    table = plot_scalability(df, os.path.join(OUTDIR, "scalability_time_vs_K.png"))
    table_path = os.path.join(OUTDIR, "scalability_table.csv")
    table.to_csv(table_path, index=False)
    print(f"Saved {table_path} ({len(table)} rows)")


if __name__ == "__main__":
    main()
