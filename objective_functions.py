"""
Objective functions for multilevel image thresholding.

All functions share the signature:
    f(thresholds, hist_prob) -> float (fitness to be MAXIMISED)

`thresholds` is a 1D array-like of K integer/float threshold levels in
(0, L-1), NOT necessarily sorted or unique when it arrives from a raw DE
vector -- callers should sanitise (sort + dedupe + clip) before calling,
which is exactly what standard_de.py does at evaluation time.

`hist_prob` is the normalised histogram p_i = h(i) / (M*N), a 1D array of
length L (256 for 8-bit images).

Since DE as implemented here performs greedy MINIMISATION (f(trial) <=
f(target)), every objective below returns its value negated so that
"better" (higher entropy / higher between-class variance) corresponds to
a LOWER (more negative) number. This lets a single DE engine be reused
for all three objectives without an explicit maximise/minimise flag
scattered through the optimiser code. Wrapper `evaluate()` handles the
sign flip in one place -- see bottom of file.
"""

import numpy as np

EPS = 1e-12


def _class_bounds(thresholds, L):
    """
    Turn a raw threshold vector into sorted, deduplicated integer class
    boundaries covering the full range [0, L-1].

    Returns bounds = [0, t1, t2, ..., tK, L-1] as ints, with t_i clipped
    into (0, L-1) and strictly increasing (duplicates nudged apart).
    """
    t = np.asarray(thresholds, dtype=float)
    t = np.clip(t, 1, L - 2)
    t = np.sort(t)

    # de-duplicate / enforce strict ordering by nudging ties upward
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = min(t[i - 1] + 1, L - 2)
    t = np.round(t).astype(int)
    # re-check after rounding (rounding can re-collide neighbours)
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = min(t[i - 1] + 1, L - 2)

    bounds = np.concatenate(([0], t, [L - 1]))
    return bounds


def otsu(thresholds, hist_prob):
    """
    Otsu's between-class variance, generalised to K thresholds.

    Maximise: sigma_B^2 = sum_k omega_k * (mu_k - mu_T)^2
    where omega_k is the class probability and mu_k the class mean.
    """
    L = len(hist_prob)
    bounds = _class_bounds(thresholds, L)
    levels = np.arange(L)

    mu_T = np.sum(levels * hist_prob)

    variance = 0.0
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        # class k covers levels [lo, hi) except final class is [lo, hi] inclusive
        if k == len(bounds) - 2:
            idx = slice(lo, hi + 1)
        else:
            idx = slice(lo, hi)
        omega_k = np.sum(hist_prob[idx])
        if omega_k < EPS:
            continue
        mu_k = np.sum(levels[idx] * hist_prob[idx]) / omega_k
        variance += omega_k * (mu_k - mu_T) ** 2

    return variance


def kapur(thresholds, hist_prob):
    """
    Kapur's (maximum) entropy criterion, generalised to K thresholds.

    Maximise: H = sum_k H_k, where H_k is the Shannon entropy of the
    pixel intensity distribution within class k, normalised by the
    class's own total probability omega_k.
    """
    L = len(hist_prob)
    bounds = _class_bounds(thresholds, L)

    total_entropy = 0.0
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        if k == len(bounds) - 2:
            idx = slice(lo, hi + 1)
        else:
            idx = slice(lo, hi)
        p_class = hist_prob[idx]
        omega_k = np.sum(p_class)
        if omega_k < EPS:
            continue
        p_norm = p_class[p_class > EPS] / omega_k
        H_k = -np.sum(p_norm * np.log(p_norm))
        total_entropy += H_k

    return total_entropy


def tsallis(thresholds, hist_prob, q=0.8):
    """
    Tsallis non-extensive entropy criterion, generalised to K thresholds.

    For each class k with entropy parameter q != 1:
        S_k^q = (1 - sum_i (p_i / omega_k)^q) / (q - 1)

    Maximise: sum_k S_k^q + (1 - q) * prod_k S_k^q
    (the pseudo-additivity correction term used in the standard
    multilevel-Tsallis thresholding formulation, e.g. de Albuquerque
    et al. 2004).
    """
    if abs(q - 1.0) < 1e-8:
        # degenerates to Shannon/Kapur entropy in the limit q -> 1
        return kapur(thresholds, hist_prob)

    L = len(hist_prob)
    bounds = _class_bounds(thresholds, L)

    S = []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        if k == len(bounds) - 2:
            idx = slice(lo, hi + 1)
        else:
            idx = slice(lo, hi)
        p_class = hist_prob[idx]
        omega_k = np.sum(p_class)
        if omega_k < EPS:
            S.append(0.0)
            continue
        p_norm = p_class / omega_k
        S_k = (1.0 - np.sum(p_norm ** q)) / (q - 1.0)
        S.append(S_k)

    S = np.array(S)
    total = np.sum(S) + (1.0 - q) * np.prod(S)
    return total


# --------------------------------------------------------------------------
# Registry + sign-flipped wrapper so the DE engine can always MINIMISE.
# --------------------------------------------------------------------------

_OBJECTIVES = {
    "otsu": otsu,
    "kapur": kapur,
    "tsallis": tsallis,
}


def evaluate(name, thresholds, hist_prob, **kwargs):
    """
    Evaluate the named objective and return the NEGATED value, i.e. the
    quantity to be MINIMISED by the DE engine. `name` in {"otsu",
    "kapur", "tsallis"}. Extra kwargs (e.g. q=0.8 for tsallis) are
    forwarded to the underlying objective.
    """
    if name not in _OBJECTIVES:
        raise ValueError(f"Unknown objective '{name}', choose from {list(_OBJECTIVES)}")
    raw_value = _OBJECTIVES[name](thresholds, hist_prob, **kwargs)
    return -raw_value


def raw_evaluate(name, thresholds, hist_prob, **kwargs):
    """Same as evaluate() but returns the natural (to-be-maximised) value."""
    if name not in _OBJECTIVES:
        raise ValueError(f"Unknown objective '{name}', choose from {list(_OBJECTIVES)}")
    return _OBJECTIVES[name](thresholds, hist_prob, **kwargs)
