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
- **Jaccard Index**
- **Dice Coefficient**

## 📖 Reference
ESWA journal article template: [Overleaf link](https://www.overleaf.com/latex/templates/eswajournal-articletemplate/xryvqrgpxdvx)
