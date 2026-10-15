# DL-070 Deep Learning Laboratory: Scientific Results Summary

**Course:** DL-070 Deep Learning Laboratory  
**Assignment:** Reduced-Scale Reproduction and Ablation Study of Barlow Twins on CIFAR-10  
**Student:** Khagesh Ranjan (Roll: 24cs2021)  
**Institution:** Rajiv Gandhi Institute of Petroleum Technology (RGIPT)  
**Date:** October 2026  
**Hardware Platform:** NVIDIA GeForce RTX 5050 Laptop GPU (8.55 GB VRAM, CUDA 12.8, PyTorch 2.11.0+cu128)  
**Dataset:** CIFAR-10 (45,000 SSL pretrain / 5,000 validation / 10,000 held-out test)  
**Budget Protocol:** 50 epochs pretraining, batch size 64, projector width 512, SGD ($\text{lr}=0.05$, Cosine Annealing, momentum 0.9, weight decay $10^{-4}$, gradient clipping 5.0).  

---

## 1. Verified Experimental Results Matrix

All values reported below are directly extracted from completed run manifests and evaluation artifacts on disk:

| Run Identifier | Method / Experimental Role | Val Probe Acc | Test Probe Acc | 20-NN Test Acc | Feature Rank | $\Delta$ vs Baseline | Total Pretrain Time | Peak VRAM | Raw Artifact Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`bt_s0`** | Standard Barlow Twins (Baseline) | **75.68%** | **74.84%** | **72.41%** | **506 / 512** | Baseline | **120.2 min** (7211.15s) | **537 MB** | `runs/main/bt_s0/manifest.json` (`pretrain_seconds`) |
| **`no_rr_s0`** | Invariance-Only Ablation ($\lambda=0$) | **57.42%** | **57.12%** | **51.98%** | **508 / 512** | **-17.72%** | **93.1 min** (5587.64s)* | **602 MB** | `runs/main/no_rr_s0/loss_log.json` (sum of 50 epoch_seconds) |
| **`fixed_s0`** | Static Hinge Control ($T=0.05^2$) | **10.28%** | **10.00%** | N/A | **0 / 512** | **-64.84%** | **76.4 min** (4581.24s) | **537 MB** | `runs/main/fixed_s0/manifest.json` (`pretrain_seconds`) |
| **`scalar_s0`** | Scalar Attenuation ($\lambda=0.001$) | **72.34%** | **71.15%** | **67.50%** | **505 / 512** | **-3.69%** | **87.0 min** (5218.23s) | **537 MB** | `runs/main/scalar_s0/manifest.json` (`pretrain_seconds`) |
| **`noise_s0`** | Noise Pilot (Adaptive Threshold) | **73.40%** | **72.69%** | **70.75%** | **508 / 512** | **-2.15%** | **88.3 min** (5297.05s) | **537 MB** | `runs/main/noise_s0/manifest.json` (`pretrain_seconds`) |

*\*Runtime Detail for `no_rr_s0`:* The 50-epoch pretraining was completed with a measured epoch-time sum of 5,587.64 seconds (93.13 minutes) logged across all 50 epochs in `loss_log.json`. The final resumed session (epochs 38–49) logged 1,260.28 seconds (21.00 minutes) in `manifest.json`. Peak VRAM was 537 MB during initial epochs and reached 602 MB during the resumed session.

*Note on Runtime:* The reported 120.2 minutes for `bt_s0` represents the verified **total pretraining time** (7,211.15 seconds) logged in `runs/main/bt_s0/manifest.json`. Downstream linear probe evaluation required an additional 106.8 seconds (18.8s feature extraction + 88.0s probe training across 4 learning rates).

---

## 2. Scientific Analysis of Findings

### A. The Necessity of Continuous Redundancy Reduction (`no_rr_s0`)
- Removing the cross-correlation decorrelation penalty entirely ($\lambda = 0$) results in a **17.72 percentage point drop** in linear probe test accuracy (from 74.84% down to 57.12%) and a **20.43 percentage point drop** in 20-NN accuracy (51.98%).
- Removing redundancy reduction substantially reduced downstream discriminability despite preserving high numerical feature rank (508/512). This shows that full rank alone does not guarantee semantically useful or sufficiently decorrelated representations.
- Without the off-diagonal penalty, different feature dimensions can copy identical information while still satisfying the diagonal invariance objective.

### B. Static-Threshold Collapse (`fixed_s0`)
- The tested static-threshold configuration produced rank-0 collapse, showing that this particular thresholding strategy failed to maintain a useful representation.
- When applying a constant hinge threshold $T = 0.05^2$ to zero out gradients for all small cross-correlations ($|C_{ij}| \le 0.05$), the representation collapsed into a 1-dimensional singularity.
- Effective feature rank dropped to **0 / 512**, and classification accuracy fell to **10.00%** (exact chance level on 10 balanced CIFAR-10 classes).
- In this reduced-scale setup, removing redundancy reduction substantially reduced downstream discriminability. The tested static-threshold configuration produced rank-0 collapse, showing that this particular thresholding strategy failed to maintain a useful representation. Continuous redundancy reduction across off-diagonal coordinates proved essential in this implementation.

### C. Controls: Scalar Attenuation & Noise Pilot
- **`scalar_s0` ($\lambda=0.001$):** Weakening the decorrelation penalty by 5-fold resulted in a **-3.69%** degradation in test accuracy (71.15%), showing that adequate penalty weight is required to separate feature channels.
- **`noise_s0` (Noise-Aware Pilot):** The noise-aware pilot underperformed standard Barlow Twins in this reduced-scale setup. Its adaptive threshold weakened some off-diagonal penalties; however, it remained stronger than globally reducing lambda (-2.15% vs -3.69%).

### D. Computational Feasibility on Consumer Hardware
- Under the local resource constraints of an NVIDIA GeForce RTX 5050 Laptop GPU, standard Barlow Twins completes 50 epochs of pretraining in **120.2 minutes** with peak VRAM utilization of **537 MB**.
- Linear probe evaluation reaches **74.84%** test accuracy, establishing a functional reduced-scale CIFAR-10 reproduction of the Barlow Twins self-supervised representation learning mechanism.

---

## 3. Cryptographic Provenance Hashes

| Run Identifier | Checkpoint SHA-256 | Encoder SHA-256 | Code Hash | Selection Metric |
| :--- | :--- | :--- | :---: | :---: |
| `runs/main/bt_s0` | `a1017b4cbbb05a61cbb74a3b477cd1c0ad753c11a4d3da01764ca97083647808` | `28594d984a00544352390242a0b2dc76d6d28516d0f28fc1cab430a9389b0c9c` | `8c273e7afb33` | Best Val LR = 0.01 (75.68%) |
| `runs/main/no_rr_s0` | `83c847fcbaea7ef0ebca5e39aa305609825b7a1772eb7ba30432655bf5644917` | `f3ea172fae58d4a94ff951e73a7027c95a0600021b0dc561fe7842217743ea41` | `43fe535a9032` | Best Val LR = 0.10 (57.42%) |
| `runs/main/fixed_s0` | `c5123d9b1aaebc0e862c6dcf6ef3eb10aa808799464ca4930b8b512032608e49` | `26d1d4f23b2c28646b10eb82fcb025fcbe827ae6cba8ec17d6ee3cb2b7be58a2` | `43fe535a9032` | Best Val LR = 0.01 (10.28%) |
| `runs/main/scalar_s0` | `6c10eb363be3dbce249ad83d95cf19e9177a9be3ae717df37119ff3eb2ca5bf5` | `371804f5eecbcfd2c18ba6a50b86a32cb2aa99066440db7fecf205c48bdfd750` | `43fe535a9032` | Best Val LR = 0.10 (71.15%) |
| `runs/main/noise_s0` | `c82305a498c199583b48f95c10fa25e64ae54f3b25916ca1ba034f89fbff27d2` | `0c6dfec61ae5bb6197b1ca7b823e20e8b15d909ddb172a08dc325bc12e2f385c` | `43fe535a9032` | Best Val LR = 0.10 (73.40%) |

---

## 4. Limitations and Future Work

Key directions for future investigation include:
1. **Multi-Seed Evaluation:** Conducting extended multi-seed evaluations to compute standard deviations and establish confidence intervals across runs.
2. **Longer Pretraining Schedules:** Scaling pretraining from 50 epochs to 200–1000 epochs to assess whether feature discrimination continues to improve toward large-scale ImageNet performance levels.
3. **Additional Benchmark Datasets:** Validating representation transfer and decorrelation behavior on more complex visual benchmarks such as CIFAR-100 and TinyImageNet.
4. **Reliable Correlation Weighting:** Investigating adaptive correlation weighting mechanisms that can statistically distinguish between sampling noise and persistent redundancy in small minibatches without causing representation collapse.
