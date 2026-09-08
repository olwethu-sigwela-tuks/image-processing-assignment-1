# -*- coding: utf-8 -*-
"""SHADE Implementation"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

"""**Population Generation**"""

def initialize_population(pop_size, K, L=256):
    population = []
    for _ in range(pop_size):
        candidate = np.sort(np.random.randint(1, L-1, size=K))
        population.append(candidate)
    return np.array(population)

"""**Objective Functions**"""

def otsu_objective(thresholds, hist):
    thresholds = np.sort(thresholds.astype(int))
    bins = np.arange(len(hist))
    classes = np.split(hist, thresholds)
    means = [np.mean(c) for c in classes]
    weights = [c.sum() for c in classes]
    overall_mean = np.sum(bins * hist)
    variance = sum([w * (m - overall_mean)**2 for w, m in zip(weights, means)])
    return -variance

def kapur_objective(thresholds, hist):
    thresholds = np.sort(thresholds.astype(int))
    classes = np.split(hist, thresholds)
    entropy = sum([-np.sum(c * np.log(c + 1e-12)) for c in classes])
    return -entropy

def tsallis_objective(thresholds, hist, q=0.8):
    thresholds = np.sort(thresholds.astype(int))
    classes = np.split(hist, thresholds)
    tsallis = sum([(1 - np.sum(c**q)) / (q-1) for c in classes])
    return -tsallis

objective_functions = {
    "Otsu": otsu_objective,
    "Kapur": kapur_objective,
    "Tsallis": tsallis_objective
}

"""**Mutation & Crossover (DE)**"""

def mutation(pop, F=0.5):
    idxs = np.random.choice(len(pop), 3, replace=False)
    r1, r2, r3 = pop[idxs]
    return r1 + F * (r2 - r3)

def crossover(target, mutant, CR=0.9):
    mask = np.random.rand(len(target)) < CR
    trial = np.where(mask, mutant, target)
    return np.sort(trial)

"""**Selection**"""

def selection(target, trial, hist, objective_fn):
    if objective_fn(trial, hist) < objective_fn(target, hist):
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
    best = None

    # --- Adaptive parameter memory initialization ---
    memory_size = 5
    memory_F = [0.5] * memory_size
    memory_CR = [0.9] * memory_size
    mem_index = 0

    for gen in range(max_gen):
        success_F, success_CR, success_deltas = [], [], []
        new_pop = []

        for target in pop:
            F, CR = sample_parameters(memory_F, memory_CR, mem_index)
            mutant = mutation(pop, F=F)
            trial = crossover(target, mutant, CR=CR)
            selected = selection(target, trial, hist, objective_fn)

            if not np.array_equal(selected, target):
                success_F.append(F)
                success_CR.append(CR)
                success_deltas.append(abs(objective_fn(target, hist) - objective_fn(selected, hist)))

            new_pop.append(selected)

        pop = np.array(new_pop)
        scores = [objective_fn(ind, hist) for ind in pop]
        best_idx = np.argmin(scores)
        best = pop[best_idx]
        best_score = scores[best_idx]

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
