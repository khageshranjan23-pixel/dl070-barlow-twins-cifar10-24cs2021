# DL-070 Deep Learning Laboratory: Reduced-Scale Reproduction and Ablation Study of Barlow Twins on CIFAR-10

**Coursework Project Submission -- October 2026**  
**Author:** Khagesh Ranjan  
**Roll Number:** 24cs2021  
**Institution:** Rajiv Gandhi Institute of Petroleum Technology (RGIPT)  
**Laboratory / Course:** DL-070 Deep Learning Laboratory  
**Assignment ID:** DL-070  
**Date:** October 2026  
**GitHub Repository:** [https://github.com/khageshranjan23-pixel/dl070-barlow-twins-cifar10-24cs2021](https://github.com/khageshranjan23-pixel/dl070-barlow-twins-cifar10-24cs2021)  

---

## Executive Summary

This repository contains the complete assignment deliverables for the DL-070 Deep Learning Laboratory paper reproduction and ablation study. We investigate **Barlow Twins: Self-Supervised Learning via Redundancy Reduction** (Zbontar et al., ICML 2021) adapted to consumer hardware (an NVIDIA GeForce RTX 5050 Laptop GPU) on CIFAR-10 under a constrained 50-epoch budget ($N=64, D=512$).

### Verified Experimental Results Matrix

All values are directly supported by completed run manifests and evaluation artifacts on disk:

| Method / Experimental Role | Val Probe Acc | Test Probe Acc | 20-NN Test Acc | Feature Rank | $\Delta$ vs Baseline | Total Pretraining Time | Peak VRAM | Raw Artifact Source |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`bt_s0` (Standard BT Baseline)** | **75.68%** | **74.84%** | **72.41%** | **506 / 512** | Baseline | **120.2 min** (7211.15s) | **537 MB** | `runs/main/bt_s0/manifest.json` (`pretrain_seconds`) |
| **`no_rr_s0` (Invariance Only)** | **57.42%** | **57.12%** | **51.98%** | **508 / 512** | **-17.72%** | **93.1 min** (5587.64s)* | **602 MB** | `runs/main/no_rr_s0/loss_log.json` (sum of 50 epoch_seconds) |
| **`fixed_s0` (Static Hinge Control)**| **10.28%** | **10.00%** | N/A | **0 / 512** | **-64.84%** | **76.4 min** (4581.24s) | **537 MB** | `runs/main/fixed_s0/manifest.json` (`pretrain_seconds`) |
| **`scalar_s0` (Weak $\lambda=0.001$)** | 72.34% | 71.15% | 67.50% | 505 / 512 | -3.69% | **87.0 min** (5218.23s) | **537 MB** | `runs/main/scalar_s0/manifest.json` (`pretrain_seconds`) |
| **`noise_s0` (Noise Pilot)** | 73.40% | 72.69% | 70.75% | 508 / 512 | -2.15% | **88.3 min** (5297.05s) | **537 MB** | `runs/main/noise_s0/manifest.json` (`pretrain_seconds`) |

*\*Runtime Detail for `no_rr_s0`:* The 50-epoch pretraining was completed with a measured epoch-time sum of 5,587.64 seconds (93.13 minutes) logged across all 50 epochs in `loss_log.json`. The final resumed session (epochs 38–49) logged 1,260.28 seconds (21.00 minutes) in `manifest.json`. Peak VRAM was 537 MB during initial epochs and reached 602 MB during the resumed session.

### Key Scientific Findings

1. **Reduced-Scale CIFAR-10 Reproduction:** Standard Barlow Twins reproduces robustly on CIFAR-10 with batch size 64, achieving **74.84%** linear probe test accuracy and **72.41%** 20-NN cosine accuracy with **120.2 minutes** of pretraining on a consumer RTX 5050 Laptop GPU (537 MB peak VRAM).
2. **The Role of Continuous Redundancy Reduction:** Completely removing the redundancy reduction penalty ($\lambda=0$, `no_rr`) drops linear probe accuracy by **17.72 percentage points** (to 57.12%) and $k$-NN accuracy by 20.43 percentage points. Removing redundancy reduction substantially reduced downstream discriminability despite preserving high numerical feature rank (508/512). This shows that full rank alone does not guarantee semantically useful or sufficiently decorrelated representations.
3. **Static-Threshold Collapse:** The static-threshold configuration produced rank-0 collapse in this particular implementation and training setup. Applying a constant hinge threshold $T=0.05^2$ (`fixed`) to zero out weak off-diagonal penalties caused the representation to collapse into a 1-dimensional singularity (rank 0/512, 10.00% chance accuracy), demonstrating that continuous gradient pressure across coordinate pairs is critical in this setup.
4. **Controls:** The noise-aware pilot underperformed standard Barlow Twins in this reduced-scale setup. Its adaptive threshold weakened some off-diagonal penalties; however, it remained stronger than globally reducing lambda (-2.15% vs -3.69%).

---

## Directory Structure

```text
DL070_Barlow_Twins_Submission/
├── .gitignore                          # Clean git ignore configuration
├── README.md                           # This complete documentation file
├── SUBMISSION_CHECKLIST.md             # Formal deliverable audit & compliance checklist
├── requirements.txt                    # Clean, platform-agnostic dependencies
├── LICENSE_OR_CITATION_NOTES.md        # MIT license and academic BibTeX citations
├── src/
│   └── barlow_local.py                 # Self-contained, modular Barlow Twins PyTorch engine
├── tests/
│   └── test_core.py                    # Comprehensive unit tests (8/8 passing)
├── notebooks/
│   └── DL070_BarlowTwins_Colab.ipynb   # Standalone Google Colab reproduction notebook
├── report/
│   ├── DL070_BarlowTwins_Report.pdf    # IEEE-formatted assignment report (exact 4 pages)
│   ├── DL070_BarlowTwins_Report.tex    # Full LaTeX source in IEEEtran format
│   └── references.bib                  # BibTeX bibliography with 10 formal citations
├── presentation/
│   ├── DL070_BarlowTwins_Presentation.pptx # 12-slide modern 16:9 widescreen presentation
│   └── PRESENTATION_SCRIPT.md          # 10-minute slide transcript and speaking notes
├── results/
│   ├── results_summary.json            # Machine-readable verified primary metrics
│   ├── RESULTS_SUMMARY.md              # Detailed scientific results documentation
│   └── figures/
│       ├── loss_curves.png             # Pretraining convergence curves (total, inv, rr)
│       └── ablation_comparison.png     # Quantitative accuracy and rank bar chart
└── viva/
    ├── CODE_WALKTHROUGH.md             # Code architecture walkthrough for examination
    └── VIVA_QUESTIONS_AND_ANSWERS.md   # 15+ rigorous viva defense questions and board proofs
```

---

## Quickstart & Installation

### Local Environment Setup

1. **Navigate to the Submission Package:**
   ```bash
   cd DL070_Barlow_Twins_Submission
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv bt_env
   # On Linux / macOS:
   source bt_env/bin/activate
   # On Windows:
   .\bt_env\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running Automated Unit Tests

To verify minibatch feature standardization, loss formulation, autograd mechanics, invariance-only ablations, and numerical stability:
```bash
python -m unittest tests/test_core.py
```
*Expected output: `Ran 8 tests in ... OK` (100% pass rate).*

---

## Reproducing Pretraining and Evaluation

### 1. Pretraining Standard Barlow Twins (Baseline `bt_s0`)
```bash
python src/barlow_local.py train \
    --method bt \
    --data data \
    --out runs/bt_s0 \
    --epochs 50 \
    --batch-size 64 \
    --dim 512 \
    --lambd 0.0051 \
    --lr 0.05 \
    --amp \
    --seed 0
```

### 2. Evaluating Downstream Accuracy (Linear Probe & $k$-NN)
```bash
python src/barlow_local.py evaluate \
    --run runs/bt_s0 \
    --data data \
    --probe-epochs 50 \
    --seed 0
```

### 3. Running the Invariance-Only Main Ablation (`no_rr`)
```bash
python src/barlow_local.py train \
    --method no_rr \
    --data data \
    --out runs/no_rr_s0 \
    --epochs 50 \
    --batch-size 64 \
    --dim 512 \
    --seed 0
```

---

## Google Colab Reproduction

For zero-setup evaluation on Google Cloud hardware:
1. Open Google Colab: [https://colab.research.google.com](https://colab.research.google.com)
2. Upload `notebooks/DL070_BarlowTwins_Colab.ipynb`.
3. Select a GPU runtime (`Runtime -> Change runtime type -> T4 GPU`).
4. Click `Runtime -> Run all`.
Structured and statically validated for execution in a clean Colab GPU session. The notebook configures dependencies, sets up CIFAR-10 data loaders, and runs pretraining and linear evaluation with clean relative paths. All paths inside the notebook are clean POSIX relative paths (`./data`, `./runs`).

---

## Limitations and Future Work

Key directions for future study:
1. **Multi-Seed Evaluation:** Conducting extended multi-seed evaluations to compute standard deviations and confidence intervals.
2. **Longer Pretraining Schedules:** Scaling pretraining from 50 epochs to 200–1000 epochs to assess further representation refinement.
3. **Additional Benchmark Datasets:** Validating representation transfer and decorrelation behavior on more complex visual benchmarks such as CIFAR-100.
4. **Reliable Correlation Weighting:** Investigating adaptive correlation weighting mechanisms that can statistically distinguish between sampling noise and persistent redundancy in small minibatches without causing representation collapse.

---

## Licensing and Attribution

- **Original Student Code:** Licensed under the MIT License - see [`LICENSE_OR_CITATION_NOTES.md`](LICENSE_OR_CITATION_NOTES.md).
- **Adapted Code & Architectural References:**
  - The Barlow Twins objective formulation is adapted from the official research repository by Zbontar et al.: [facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins) (CC-BY-NC 4.0 / MIT compatible research codebase).
  - The ResNet-18 backbone is adapted from PyTorch's `torchvision.models.resnet18` (BSD 3-Clause License), with standard modifications for 32x32 CIFAR resolution (3x3 conv1, stride 1, maxpool removed).
  - Complete academic BibTeX citations for Barlow Twins, ResNet, SimCLR, BYOL, and CIFAR-10 are provided in `report/references.bib` and `LICENSE_OR_CITATION_NOTES.md`.
