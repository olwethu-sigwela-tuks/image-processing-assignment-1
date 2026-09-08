"""
JADE
"""

import numpy as np

from objective_functions import evaluate as obj_evaluate

import random


class JADE:
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
        c = 0.1,
        p=0.05
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
        self.c = c
        self.p = p

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

    def sort_lists_in_parallel(self, list1, list2):
        sorted1, sorted2 = zip(*sorted(zip(list1, list2), key=lambda x: x[0]))

        return list(sorted1), list(sorted2)

    def _init_population(self):
        pop = list(self.rng.uniform(self.lb, self.ub, size=(self.NP, self.dim)))
        fitness = [self._evaluate(ind) for ind in pop]

        fitness, pop = self.sort_lists_in_parallel(fitness, pop)
        return np.array(pop), np.array(fitness)

    def _mutate(self, pop, target_idx, archive, F_i):
        
        idxs = [i for i in range(self.NP) if i != target_idx]
        
        

        p_num = max(1, int(round(self.p * self.NP)))
        p_best = pop[:p_num]

        threshold_i = pop[target_idx]
        threshold_best = random.choice(p_best)

        r1_idx = self.rng.choice(idxs)
        threshold_r1 = pop[r1_idx]

        union = [pop[i] for i in idxs if i != r1_idx] + archive
        
        threshold_r2 = random.choice(union)

        mutant = threshold_i + (F_i * (threshold_best - threshold_i)) + (F_i * (threshold_r1 - threshold_r2))
        return np.clip(mutant, self.lb, self.ub)

    def lehmer_mean(self, lst):
        return sum(i**2 for i in lst)/sum(i for i in lst)


    def _crossover(self, target, mutant, CR_i):
        trial = target.copy()
        j_rand = self.rng.integers(self.dim)
        cross_mask = self.rng.random(self.dim) <= CR_i
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

        s_CR = [] #set of all successful CR's
        s_F = [] #set of all successful F's
        archive = []
        
        
        CR_mean = 0.5
        F_mean = 0.5
        
        generation = 0
        while self.fes_used < self.MAX_FES:
            generation += 1
            new_pop, new_fitness = pop.copy(), fitness.copy()

            s_CR = []
            s_F = []

            for i in range(self.NP):
                if self.fes_used >= self.MAX_FES:
                    break

                CR_i = self.rng.normal(loc=CR_mean, scale=0.1)
                if CR_i < 0:
                    CR_i = 0
    
                if CR_i > 1:
                    CR_i = 1
    
                F_i = F_mean + (0.1*self.rng.standard_cauchy())
    
                if F_i > 1:
                    F_i = 1
    
                if F_i <= 0:
                    while F_i <= 0:
                        F_i = F_mean + (0.1*self.rng.standard_cauchy())
    
                
                
                mutant = self._mutate(pop, i, archive, F_i)
                trial = self._crossover(pop[i], mutant, CR_i)
                trial_fit = self._evaluate(trial)


                # Greedy selection (Eq. 1 in background doc)
                if trial_fit <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = trial_fit
                    s_CR.append(CR_i)
                    s_F.append(F_i)
                    archive.append(pop[i])

                    if len(archive) > self.NP:
                        random_removal_idx = random.randint(0, len(archive) - 1)
                        archive.pop(random_removal_idx)

                    if trial_fit < best_fit:
                        best_fit = trial_fit
                        best_vec = trial.copy()

            self.history.append(-best_fit)
            if verbose:
                print(
                    f"[gen {generation}] FEs={self.fes_used}/{self.MAX_FES} "
                    f"best={-best_fit:.6f}"
                )

            new_fitness, new_pop = self.sort_lists_in_parallel(list(new_fitness), list(new_pop))
            pop = np.array(new_pop)
            fitness = np.array(new_fitness)
            if len(s_CR) > 0 and len(s_F) > 0:
                CR_mean = ((1 - self.c) * CR_mean) + (self.c * np.mean(s_CR))
                F_mean = ((1 - self.c) * F_mean) + (self.c * self.lehmer_mean(s_F))

        best_thresholds = np.sort(np.round(np.clip(best_vec, self.lb, self.ub)).astype(int))
        return best_thresholds, -best_fit, self.history
