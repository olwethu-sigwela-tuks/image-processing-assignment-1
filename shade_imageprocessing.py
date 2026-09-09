# -*- coding: utf-8 -*-
"""SHADE Implementation"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from objective_functions import evaluate as _obj_evaluate

"""**Population Generation**"""

def initialize_population(pop_size, K, L=256):
    population = []
    for _ in range(pop_size):
        candidate = np.sort(np.random.randint(1, L-1, size=K))
        population.append(candidate)
    return np.array(population)

"""**Objective Functions**"""
def otsu_objective(thresholds, hist):
    return _obj_evaluate("otsu", thresholds, hist)

def kapur_objective(thresholds, hist):
    return _obj_evaluate("kapur", thresholds, hist)

def tsallis_objective(thresholds, hist, q=0.8):
    return _obj_evaluate("tsallis", thresholds, hist, q=q)

# def otsu_objective(thresholds, hist):
#     thresholds = np.sort(thresholds.astype(int))
#     bins = np.arange(len(hist))
#     classes = np.split(hist, thresholds)
#     means = [np.mean(c) for c in classes]
#     weights = [c.sum() for c in classes]
#     overall_mean = np.sum(bins * hist)
#     variance = sum([w * (m - overall_mean)**2 for w, m in zip(weights, means)])
#     return -variance
#
# def kapur_objective(thresholds, hist):
#     thresholds = np.sort(thresholds.astype(int))
#     classes = np.split(hist, thresholds)
#     entropy = sum([-np.sum(c * np.log(c + 1e-12)) for c in classes])
#     return -entropy
#
# def tsallis_objective(thresholds, hist, q=0.8):
#     thresholds = np.sort(thresholds.astype(int))
#     classes = np.split(hist, thresholds)
#     tsallis = sum([(1 - np.sum(c**q)) / (q-1) for c in classes])
#     return -tsallis

objective_functions = {
    "Otsu": otsu_objective,
    "Kapur": kapur_objective,
    "Tsallis": tsallis_objective
}

"""**Mutation & Crossover (DE)**"""

def mutation(current, pop, fitness, archive, F, p=0.2):
    # Select p-best individual by FITNESS RANK, not array position
    p_best_size = max(2, int(p * len(pop)))
    best_idx = np.argsort(fitness)[:p_best_size]
    p_best = pop[best_idx[np.random.randint(0, len(best_idx))]]

    # Random individuals
    r1 = pop[np.random.randint(len(pop))]
    if len(archive) > 0:
        r2 = archive[np.random.randint(len(archive))]
    else:
        r2 = pop[np.random.randint(len(pop))]

    # Current-to-pbest mutation
    mutant = current + F * (p_best - current) + F * (r1 - r2)
    
    # Clip thresholds to valid range [1, 255]
    mutant = np.clip(mutant, 1, 255)
    return mutant


def crossover(target, mutant, CR=0.9):
    mask = np.random.rand(len(target)) < CR
    j_rand = np.random.randint(len(target))
    mask[j_rand] = True
    trial = np.where(mask, mutant, target)
    return np.sort(trial)

"""**Selection**"""

def selection(target, trial, hist, objective_fn):
    if objective_fn(trial, hist) <= objective_fn(target, hist):
        return trial
    else:
        return target

"""**Adaptive Parameter Sampling & Memory Update**"""

def sample_parameters(memory_F, memory_CR, mem_index):
    F = np.random.standard_cauchy() * 0.1 + memory_F[mem_index]
    while F <= 0:
        F = np.random.standard_cauchy() * 0.1 + memory_F[mem_index]
    F = min(F, 1.0)

    CR = np.random.normal(memory_CR[mem_index], 0.1)
    CR = np.clip(CR, 0, 1)

    return F, CR

def update_memory(memory_F, memory_CR, mem_index, success_F, success_CR, success_deltas):
    if success_F:
        weights = np.array(success_deltas) / np.sum(success_deltas)
        mean_F = np.sum(weights * np.array(success_F)**2) / np.sum(weights * np.array(success_F))
        mean_CR = np.sum(weights * np.array(success_CR))
        memory_F[mem_index] = mean_F
        memory_CR[mem_index] = mean_CR
        mem_index = (mem_index + 1) % len(memory_F)
    return memory_F, memory_CR, mem_index

"""**SHADE Implementation (Fixed Population Size)**"""

def shade(hist, K, objective_fn, pop_size=30, max_gen=100):
    pop = initialize_population(pop_size, K)
    fitness = np.array([objective_fn(ind, hist) for ind in pop])
    best = None
    archive = []

    # --- Adaptive parameter memory initialization ---
    memory_size = 5
    memory_F = [0.5] * memory_size
    memory_CR = [0.9] * memory_size
    mem_index = 0

    for gen in range(max_gen):
        success_F, success_CR, success_deltas = [], [], []
        new_pop = []
        new_fitness = []

        for i, target in enumerate(pop):
            F, CR = sample_parameters(memory_F, memory_CR, mem_index)
            p = np.random.uniform(2/len(pop), 0.2)
            mutant = mutation(target, pop, fitness, archive, F, p=p)
            trial = crossover(target, mutant, CR=CR)
            trial_fit = objective_fn(trial, hist)
            target_fit = fitness[i]

            if trial_fit <= target_fit:
                selected, selected_fit = trial, trial_fit
                # Track successful parameters for memory update
                success_F.append(F)
                success_CR.append(CR)
                success_deltas.append(abs(target_fit - trial_fit) + 1e-12)

                archive.append(target)
                if len(archive) > len(pop):
                  # Keep archive size ≤ population size
                  archive.pop(np.random.randint(len(archive)))
            else:
                selected, selected_fit = target, target_fit

            new_pop.append(selected)
            new_fitness.append(selected_fit)

        pop = np.array(new_pop)
        fitness = np.array(new_fitness)
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_score = fitness[best_idx]

        # --- Update memory pools ---
        memory_F, memory_CR, mem_index = update_memory(memory_F, memory_CR, mem_index,
                                                       success_F, success_CR, success_deltas)

    return best, best_score, -best_score

"""**Performance Evaluation Metrics**"""

def compute_metrics(original, segmented, thresholds):
    psnr = peak_signal_noise_ratio(original, segmented)
    ssim = structural_similarity(original, segmented)

    regions = np.digitize(original, thresholds)
    U = 0
    for k in range(len(thresholds)+1):
        region_pixels = original[regions == k]
        if len(region_pixels) > 0 and np.mean(region_pixels) != 0:
            mu = np.mean(region_pixels)
            sigma = np.std(region_pixels)
            U += sigma / mu
    U /= (len(thresholds)+1)

    return psnr, ssim, U

"""**Reconstruct Segmented Image**"""

def apply_thresholds(img, thresholds):
    thresholds = np.sort(np.round(thresholds).astype(int))
    segmented = np.zeros_like(img)
    regions = np.digitize(img, thresholds)

    for k in range(len(thresholds)+1):
        region_pixels = img[regions == k]
        if len(region_pixels) > 0:
            mean_val = np.mean(region_pixels)
            segmented[regions == k] = mean_val

    return segmented

"""**Experiment Loop**"""

def run_experiments(image_folder, K_values=[3,5,7,9,11,12], trials=30):
    for img_file in os.listdir(image_folder):
        if not (img_file.endswith(".png") or img_file.endswith(".jpg")):
            continue

        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Skipping (not an image):", img_path)
            continue

        print("Loaded:", img_path)
        hist = cv2.calcHist([img], [0], None, [256], [0,256])
        hist_norm = hist.flatten() / hist.sum()

        for K in K_values:
            for obj_name, obj_fn in objective_functions.items():
                for trial in range(trials):
                    best_thresholds, raw_score, minimized_score = shade(hist_norm, K, obj_fn)

                    segmented = apply_thresholds(img, best_thresholds)
                    psnr, ssim, U = compute_metrics(img, segmented, best_thresholds)

                    print(f"Image: {img_file}, Trial: {trial+1}, K={K}, Objective={obj_name}, "
                          f"Best Thresholds={best_thresholds}, Raw Score={raw_score:.4f}, "
                          f"PSNR={psnr:.4f}, SSIM={ssim:.4f}, Uniformity={U:.4f}")

if __name__ == "__main__":
    run_experiments("content")
