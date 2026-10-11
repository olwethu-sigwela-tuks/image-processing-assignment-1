# image-processing-assignment-1

# Multilevel Image Thresholding with Differential Evolution

## 📘 Assignment Overview
Multilevel image thresholding is a core technique in digital image processing used to segment an image into multiple regions based on pixel intensity distributions. As the number of thresholds **K** increases, exhaustive search becomes computationally infeasible. Unlike deep learning approaches that require costly annotated ground truth masks, multilevel thresholding is **unsupervised** — requiring zero training data, zero manual labels, and zero offline training.

This project implements and evaluates **Differential Evolution (DE)** and its variants to solve the multilevel thresholding optimization problem using three objective functions:
- **Otsu’s Variance**
- **Kapur’s Entropy**
- **Tsallis Non-Extensive Entropy**

---

## 📂 Datasets
1. **BSD500 (10 standard images)**  
   Proof-of-concept benchmarking and baseline convergence analysis.
2. **CHAOS MRI scans (15 medical images)**  
   Real-world medical application to assess robustness and stability.

---

## ⚙️ Algorithms Implemented
- Standard DE (DE/rand/1/bin)
- JADE (adaptive DE with p-best mutation)
- SHADE (success-history based adaptive DE)
- L-SHADE (SHADE + linear population size reduction)
- LADE (Late Acceptance DE)

---

## 🎯 Threshold Levels
Algorithms are evaluated across six threshold levels:
K ∈ {3, 5, 7, 9, 11, 12}

## 📊 Evaluation Metrics
Image reconstruction quality is measured using:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **Uniformity (U)** — region homogeneity

For medical images, additional metrics:
- **Class Separability**

## 🗂️ Project Structure
| File | Purpose |
|---|---|
| `experiment_harness.py` | Main entry point. Runs all algorithms × objectives × K × seeds on both datasets and writes the result CSVs |
| `standard_de.py`, `lade.py`, `jade.py`, `shade_imageprocessing.py`, `l_shade_imageprocessing.py` | The five DE variants |
| `objective_functions.py` | Otsu, Kapur and Tsallis objectives |
| `metrics.py` | PSNR, SSIM, Uniformity, Class Separability and segmentation helpers |
| `statistical_analysis.py` | Pairwise Wilcoxon signed-rank + Friedman tests |
| `convergence_analysis.py` | Convergence curves (mean ± std over seeds) |
| `scalability_analysis.py` | Runtime vs K analysis |
| `comparison_images.py` | Side-by-side segmented images for all 5 algorithms (Kapur, K = 5 and 7) |
| `BDS500/`, `CHAOS_DATA/` | Input images (BSD500 and CHAOS MRI) |
| `results/` | All outputs (`phase1_bsd500/`, `phase2_chaos/`) |

---

## 🚀 How to Run

### 1. Requirements
Python 3.9+ and the following packages:
```bash
pip install -r requirements.txt
```

### 2. Run the experiments
```bash
python3 experiment_harness.py
```
This runs **Phase 1 (BSD500)** and then **Phase 2 (CHAOS MRI)** with the settings in the `CONFIG` block at the top of `experiment_harness.py`:

| Setting | Default |
|---|---|
| Algorithms | StandardDE, LADE, JADE, SHADE, LSHADE |
| Objectives | otsu, kapur, tsallis (q = 0.8) |
| K values | 3, 5, 7, 9, 11, 12 |
| Runs (seeds) | 30 per configuration |
| Population size / budget | NP = 50, MAX_FES = 10,000 |

Results are saved per dataset to `results/phase1_bsd500/` and `results/phase2_chaos/`:
- `raw_results.csv` – one row per run (used by the statistical tests)
- `summary_table.csv` – mean ± std per (algorithm, objective, K)
- `summary_by_image.csv` – same, per image
- `convergence/*.npy` – per-run best-fitness traces

The harness is resumable, meaning if it is interrupted, run the same command again and already-completed runs are skipped. The full experiment is long (about 67,500 runs), so leave it running. To run only one dataset, comment out the other `run_phase(...)` call in `main()`.

### 3. Run the analyses
Run these from the project root after the experiments have finished. Each script has constants at the top (`RAW_RESULTS`, `OUTDIR`, etc.) that point to the dataset it analyses; edit them to switch between `phase1_bsd500` and `phase2_chaos`.

```bash
python3 statistical_analysis.py    # Wilcoxon + Friedman  -> wilcoxon_<metric>.csv, friedman_<metric>.csv
python3 convergence_analysis.py    # convergence plot     -> convergence_<image>_<objective>_K<k>.png
python3 scalability_analysis.py    # runtime vs K table + plot
```

Settings you may want to change:
- `statistical_analysis.py`: `METRIC` (`psnr`, `ssim`, `uniformity`, `class_separability`) and `ALPHA` (0.05).
- `convergence_analysis.py`: `IMAGE`, `OBJECTIVE`, `K`, `ALGORITHMS`. Needs the `convergence/` folder from the experiment run.

### 4. Generate comparison images
```bash
python3 comparison_images.py
```
Segments one image with all five algorithms on Kapur for K = 5 and K = 7 and saves a comparison grid plus the individual segmented images to `results/comparison_images/`. Change `IMAGE_PATH`, `OBJECTIVE` or `K_VALUES` at the top of the file to vary it.

---

## 📖 Reference
ESWA journal article template: [Overleaf link](https://www.overleaf.com/latex/templates/eswajournal-articletemplate/xryvqrgpxdvx)
