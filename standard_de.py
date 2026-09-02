"""
Standard Differential Evolution: DE/rand/1/bin with greedy selection.

Reference (see Background_Differential_Evolution.pdf, Eq. 1):
    x_i^{G+1} = u_i^G  if f(u_i^G) <= f(x_i^G)
              = x_i^G  otherwise
"""

import numpy as np

from objective_functions import evaluate as obj_evaluate


class StandardDE:
    """
    DE/rand/1/bin optimiser for multilevel image thresholding.

    Parameters
    ----------
    dim : int
        Number of thresholds K to search for.
    bounds : tuple(float, float)
        (lower, upper) bound applied to every dimension, e.g. (1, 254)
        for 8-bit images (thresholds must lie strictly inside [0, L-1]).
    hist_prob : np.ndarray
        Normalised image histogram, passed straight to the objective
        function every evaluation.
    objective_name : str
        One of "otsu", "kapur", "tsallis" (see objective_functions.py).
    NP : int
        Population size.
    MAX_FES (Maximum Function Evaluations) : int
        Maximum number of objective-function evaluations. This is the
        stopping criterion
    F : float
        Differential mutation scale factor.
    CR : float
        Crossover probability.
    seed : int or None
        RNG seed for reproducibility across independent runs.
    objective_kwargs : dict
        Extra kwargs forwarded to the objective (e.g. {"q": 0.8} for
        tsallis).
    """

    def __init__(
        self,
        dim,
        bounds,
        hist_prob,
        objective_name,
        NP=50,
        MAX_FES=10000,
        F=0.5,
        CR=0.9,
        seed=None,
        objective_kwargs=None,
    ):
        self.dim = dim
        self.lb, self.ub = bounds
        self.hist_prob = hist_prob
        self.objective_name = objective_name
        self.NP = NP
        self.MAX_FES = MAX_FES
        self.F = F
        self.CR = CR
        self.objective_kwargs = objective_kwargs or {}
        self.rng = np.random.default_rng(seed)

        self.fes_used = 0
        # convergence history: best-so-far fitness recorded once per
        # generation (natural/maximised units, i.e. sign-flipped back from the internal minimisation objective) for plotting later.
        self.history = []

    # ----------------------------------------------------------------
    def _evaluate(self, vec):
        self.fes_used += 1
        return obj_evaluate(
            self.objective_name, vec, self.hist_prob, **self.objective_kwargs
        )

    def _init_population(self):
        pop = self.rng.uniform(self.lb, self.ub, size=(self.NP, self.dim))
        fitness = np.array([self._evaluate(ind) for ind in pop])
        return pop, fitness

    def _mutate(self, pop, target_idx):
        idxs = [i for i in range(self.NP) if i != target_idx]
        r1, r2, r3 = self.rng.choice(idxs, size=3, replace=False)
        mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
        return np.clip(mutant, self.lb, self.ub)

    def _crossover(self, target, mutant):
        trial = target.copy()
        j_rand = self.rng.integers(self.dim)
        cross_mask = self.rng.random(self.dim) < self.CR
        cross_mask[j_rand] = True  # guarantee at least one mutant gene
        trial[cross_mask] = mutant[cross_mask]
        return trial

    # ----------------------------------------------------------------
    def run(self, verbose=False):
        """
        Execute DE/rand/1/bin until MAX_FES is exhausted.

        Returns
        -------
        best_thresholds : np.ndarray, sorted, rounded to int
        best_fitness_natural : float
            The objective value in its natural (to-be-maximised) sense.
        history : list of float
            Best-so-far fitness (natural sense) recorded once per
            generation -- use for convergence-curve plots.
        """
        pop, fitness = self._init_population()

        best_idx = np.argmin(fitness)  # internal fitness is minimised
        best_vec = pop[best_idx].copy()
        best_fit = fitness[best_idx]
        self.history.append(-best_fit)  # store in natural (maximise) units

        generation = 0
        while self.fes_used < self.MAX_FES:
            generation += 1
            for i in range(self.NP):
                if self.fes_used >= self.MAX_FES:
                    break

                mutant = self._mutate(pop, i)
                trial = self._crossover(pop[i], mutant)
                trial_fit = self._evaluate(trial)

                # Greedy selection (Eq. 1 in background doc)
                if trial_fit <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_fit:
                        best_fit = trial_fit
                        best_vec = trial.copy()

            self.history.append(-best_fit)
            if verbose:
                print(
                    f"[gen {generation}] FEs={self.fes_used}/{self.MAX_FES} "
                    f"best={-best_fit:.6f}"
                )

        best_thresholds = np.sort(np.round(np.clip(best_vec, self.lb, self.ub)).astype(int))
        return best_thresholds, -best_fit, self.history
