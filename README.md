# Reduced-Scale Reproduction and Ablation Study of Barlow Twins on CIFAR-10

- **Author:** Khagesh Ranjan
- **Course:** DL-070 Deep Learning Laboratory
- **Institution:** Rajiv Gandhi Institute of Petroleum Technology (RGIPT)
- **Repository:** [https://github.com/khageshranjan23-pixel/dl070-barlow-twins-cifar10-24cs2021](https://github.com/khageshranjan23-pixel/dl070-barlow-twins-cifar10-24cs2021)
- **License:** [MIT License](LICENSE)

---

## Overview

This repository provides a lightweight, self-contained implementation and empirical ablation study of **Barlow Twins: Self-Supervised Learning via Redundancy Reduction** (Zbontar et al., ICML 2021) on CIFAR-10, adapted for execution on consumer hardware (NVIDIA GeForce RTX 5050 Laptop GPU, 8.55 GB VRAM).

The codebase reproduces the core cross-correlation optimization objective under a 50-epoch budget ($N=64, D=512$), evaluates representations via frozen linear probing and non-parametric $k$-NN classification ($k=20$), and tests the necessity of continuous off-diagonal cross-correlation decorrelation.

---

## Verified Experimental Results

All reported values are directly verified from the execution manifests and evaluation logs preserved in [`results/evidence/`](results/evidence/):

| Method / Experimental Role | Val Probe Acc | Test Probe Acc | 20-NN Test Acc | Feature Rank | $\Delta$ vs Baseline | Total Pretraining Time | Peak VRAM | Evidence Source |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`bt_s0` (Standard BT Baseline)** | **75.68%** | **74.84%** | **72.41%** | **506 / 512** | Baseline | **120.2 min** (7,211.15s) | **537 MB** | [`results/evidence/bt_s0/manifest.json`](results/evidence/bt_s0/manifest.json) |
| **`no_rr_s0` (Invariance Only)** | **57.42%** | **57.12%** | **51.98%** | **508 / 512** | **-17.72%** | **93.1 min** (5,587.64s)* | **602 MB** | [`results/evidence/no_rr_s0/manifest.json`](results/evidence/no_rr_s0/manifest.json) |
| **`fixed_s0` (Static Hinge Control)**| **10.28%** | **10.00%** | N/A | **0 / 512** | **-64.84%** | **76.4 min** (4,581.24s) | **537 MB** | [`results/evidence/fixed_s0/manifest.json`](results/evidence/fixed_s0/manifest.json) |
| **`scalar_s0` (Weak $\lambda=0.001$)** | 72.34% | 71.15% | 67.50% | 505 / 512 | -3.69% | **87.0 min** (5,218.23s) | **537 MB** | [`results/evidence/scalar_s0/manifest.json`](results/evidence/scalar_s0/manifest.json) |
| **`noise_s0` (Noise Pilot)** | 73.40% | 72.69% | 70.75% | 508 / 512 | -2.15% | **88.3 min** (5,297.05s) | **537 MB** | [`results/evidence/noise_s0/manifest.json`](results/evidence/noise_s0/manifest.json) |

*\*Runtime Detail for `no_rr_s0`:* The 50-epoch pretraining was completed with a measured epoch-time sum of 5,587.64 seconds (93.13 minutes) logged across all 50 epochs. The final resumed session (epochs 38–49) logged 1,260.28 seconds (21.00 minutes) in `manifest.json`. Peak VRAM was 537 MB initially and reached 602 MB during the resumed session.

### Key Scientific Findings

1. **Reduced-Scale CIFAR-10 Reproduction:** Standard Barlow Twins achieved **74.84%** linear probe test accuracy and **72.41%** 20-NN cosine accuracy in this reduced-scale seed-0 run with batch size 64 and **120.2 minutes** of pretraining on a consumer RTX 5050 Laptop GPU (537 MB peak VRAM).
2. **The Role of Continuous Redundancy Reduction:** Completely removing the redundancy reduction penalty ($\lambda=0$, `no_rr`) drops linear probe accuracy by **17.72 percentage points** (to 57.12%) and $k$-NN accuracy by 20.43 percentage points. Removing redundancy reduction substantially reduced downstream discriminability despite preserving high numerical feature rank (508/512). This shows that full rank alone does not guarantee semantically useful or sufficiently decorrelated representations.
3. **Static-Threshold Collapse:** The tested static-threshold configuration produced rank-0 collapse in this setup. This single experiment does not establish that every thresholded formulation will collapse. Applying a constant hinge threshold $T=0.05^2$ (`fixed`) to zero out weak off-diagonal penalties caused the representation to completely collapse (rank 0/512, 10.00% chance accuracy), demonstrating that continuous gradient pressure across coordinate pairs is critical in this setup.
4. **Controls:** The noise-aware pilot underperformed standard Barlow Twins in this reduced-scale setup. Its adaptive threshold weakened some off-diagonal penalties; however, it remained stronger than globally reducing lambda (-2.15% vs -3.69%).

---

## Repository Structure

```text
.
├── .gitignore
├── LICENSE
├── ATTRIBUTION.md
├── README.md
├── requirements.txt
├── src/
│   └── barlow_local.py
├── tests/
│   └── test_core.py
└── results/
    ├── results_summary.json
    ├── evidence/
    └── figures/
        ├── loss_curves.png
        └── ablation_comparison.png
```

- **`src/barlow_local.py`**: Modular implementation of Barlow Twins, ResNet-18 CIFAR-10 adaptation, cross-correlation calculation, loss variants, linear probe training, $k$-NN evaluation, and SVD effective rank computation.
- **`tests/test_core.py`**: Unit test suite verifying feature standardization, cross-correlation shape and values, loss terms, symmetry, gradient flow, checkpoint roundtrips, multi-format feature extraction, linear probing, and rank calculation.
- **`results/`**: Machine-readable primary summary (`results_summary.json`), pre-generated loss curves and ablation bar charts (`figures/`), and experimental run manifests and downstream evaluation records (`evidence/`).

---

## Quickstart & Installation

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/khageshranjan23-pixel/dl070-barlow-twins-cifar10-24cs2021.git
   cd dl070-barlow-twins-cifar10-24cs2021
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv bt_env
   # Linux / macOS:
   source bt_env/bin/activate
   # Windows:
   .\bt_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running Unit Tests

Run the complete 11-test suite to verify core mechanics, loss formulations, gradient flows, and evaluation utilities:
```bash
python -m unittest tests/test_core.py -v
```
*Expected: 11 tests pass (`OK`).*

### Hardware & Architecture Preflight

To verify CUDA availability and execute a single forward/backward pass through the backbone and projector:
```bash
python src/barlow_local.py selftest --device cuda
```

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
*Note: Pretraining automatically constructs the 45,000-image training split and 5,000-image validation split, and does not download or construct the official test dataset.*

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

## Limitations and Future Work

1. **Multi-Seed Evaluation:** Conducting extended multi-seed evaluations to compute standard deviations and confidence intervals.
2. **Longer Pretraining Schedules:** Scaling pretraining from 50 epochs to 200–1000 epochs to assess further representation refinement.
3. **Additional Benchmark Datasets:** Validating representation transfer and decorrelation behavior on more complex visual benchmarks such as CIFAR-100.
4. **Adaptive Correlation Weighting:** Investigating adaptive correlation weighting mechanisms that can statistically distinguish between sampling noise and persistent redundancy in small minibatches without causing representation collapse.

---

## Licensing and Attribution

- **Code License:** [MIT License](LICENSE) (Copyright (c) 2026 Khagesh Ranjan).
- **Attribution & Citations:** See [ATTRIBUTION.md](ATTRIBUTION.md) for academic citations for Barlow Twins, torchvision ResNet-18, and CIFAR-10.
