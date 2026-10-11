"""
Computational time vs K scalability analysis.
"""

import os

import pandas as pd
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_RESULTS = "results/phase2_chaos/raw_results.csv"
OUTDIR = "results/phase2_chaos"


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


def linear_fit_table(df):
    """
    Scalability summary matching the report table (Table 16): for each algorithm, mean time at K=3 and K=12, growth ratio t(12)/t(3),
    and the least-squares line time = a + slope*K fitted to the mean time at each K (averaged over all images/objectives/seeds).
    r2 is the coefficient of determination of that fit (1 - SS_res/SS_tot).
    """
    rows = []
    for algo, g in df.groupby("algorithm"):
        m = g.groupby("K")["time_sec"].mean()
        fit = linregress(m.index.values, m.values)
        rows.append({
            "algorithm": algo,
            "t_K3": m.loc[3], "t_K12": m.loc[12],
            "growth_ratio": m.loc[12] / m.loc[3],
            "slope_s_per_K": fit.slope,
            "r2": fit.rvalue ** 2,
        })
    return pd.DataFrame(rows)


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

    fit = linear_fit_table(df)
    fit_path = os.path.join(OUTDIR, "scalability_fit.csv")
    fit.to_csv(fit_path, index=False)
    print(f"Saved {fit_path}")
    print("\nAlgorithm   t(K=3)  t(K=12)  Growth  Slope(s/K)  r^2")
    for _, r in fit.iterrows():
        print(f"{r['algorithm']:<11} {r['t_K3']:.3f} s {r['t_K12']:.3f} s  {r['growth_ratio']:.2f}x  "
              f"{r['slope_s_per_K']:.4f}      {r['r2']:.4f}")


if __name__ == "__main__":
    main()
