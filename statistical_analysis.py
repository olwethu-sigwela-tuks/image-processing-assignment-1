"""Statistical significance testing: pairwise Wilcoxon signed-rank tests and a Friedman test across all algorithms"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

RAW_RESULTS = "results/phase1_bsd500/raw_results.csv"
METRIC = "uniformity"
ALPHA = 0.05
OUTDIR = "results/phase1_bsd500"


def _pivot_for_group(df, metric, algorithms):
    """
    Pivot one (objective, K) slice of raw_results.csv into a wide table:
    rows = (image, seed), columns = algorithm, values = metric.
    Only rows where EVERY requested algorithm has a value are kept, since paired tests require complete cases.
    """
    sub = df[df["algorithm"].isin(algorithms)]
    wide = sub.pivot_table(index=["image", "seed"], columns="algorithm", values=metric)
    present = [a for a in algorithms if a in wide.columns]
    wide = wide.dropna(subset=present)
    return wide


def wilcoxon_pairwise(df, metric="psnr", algorithms=None, alpha=0.05):
    """
    Pairwise Wilcoxon signed-rank test between every pair of algorithms, separately for each (objective, K) combination present in df.
    Returns a DataFrame: objective, K, algo_a, algo_b, n_pairs, statistic, p_value, significant (p < alpha),
    better (which of the two has the higher mean `metric`
    """
    algorithms = algorithms or sorted(df["algorithm"].unique())
    rows = []

    for (objective, K), group in df.groupby(["objective", "K"]):
        wide = _pivot_for_group(group, metric, algorithms)
        if wide.empty:
            continue

        for algo_a, algo_b in itertools.combinations(algorithms, 2):
            if algo_a not in wide.columns or algo_b not in wide.columns:
                continue
            a = wide[algo_a].values
            b = wide[algo_b].values
            if len(a) < 1 or np.allclose(a, b):
                continue

            try:
                stat, p = wilcoxon(a, b)
            except ValueError:
                continue

            rows.append({
                "objective": objective, "K": K,
                "algo_a": algo_a, "algo_b": algo_b,
                "n_pairs": len(a),
                "statistic": stat, "p_value": p,
                "significant": p < alpha,
                "better": algo_a if a.mean() > b.mean() else algo_b,
            })

    return pd.DataFrame(rows)


def friedman_test(df, metric="psnr", algorithms=None, alpha=0.05):
    """
    Friedman test across ALL algorithms simultaneously, separately for each (objective, K). Returns one row per (objective, K) with the
    Friedman statistic, p-value, significance flag, and each algorithm's mean rank (rank 1 = best, i.e. highest metric value for that run).
    """
    algorithms = algorithms or sorted(df["algorithm"].unique())
    rows = []

    for (objective, K), group in df.groupby(["objective", "K"]):
        wide = _pivot_for_group(group, metric, algorithms)
        present = [a for a in algorithms if a in wide.columns]
        wide = wide[present]
        if wide.shape[1] < 3 or wide.empty:
            continue

        try:
            stat, p = friedmanchisquare(*[wide[a].values for a in wide.columns])
        except ValueError:
            continue

        ranks = wide.rank(axis=1, ascending=False)
        mean_ranks = ranks.mean(axis=0).to_dict()

        row = {"objective": objective, "K": K, "n_pairs": len(wide),
               "statistic": stat, "p_value": p, "significant": p < alpha}
        for algo in algorithms:
            row[f"mean_rank_{algo}"] = mean_ranks.get(algo, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(RAW_RESULTS)

    wpw = wilcoxon_pairwise(df, metric=METRIC, alpha=ALPHA)
    wpw_path = os.path.join(OUTDIR, f"wilcoxon_{METRIC}.csv")
    wpw.to_csv(wpw_path, index=False)
    print(f"Saved pairwise Wilcoxon results to {wpw_path} ({len(wpw)} comparisons)")

    fr = friedman_test(df, metric=METRIC, alpha=ALPHA)
    fr_path = os.path.join(OUTDIR, f"friedman_{METRIC}.csv")
    fr.to_csv(fr_path, index=False)
    print(f"Saved Friedman test results to {fr_path} ({len(fr)} (objective,K) groups)")


if __name__ == "__main__":
    main()
