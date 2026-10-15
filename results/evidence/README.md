# Experimental Evidence & Cryptographic Provenance

This directory stores the sanitized execution manifests, evaluation records, and cryptographic provenance metadata corresponding to the reported Barlow Twins CIFAR-10 experiments.

---

## 1. Cryptographic Provenance Matrix

Every checkpoint and encoder hash below was calculated directly from the original binary artifacts on disk using SHA-256. The 12-character prefixes match the contemporaneous training manifests generated at pretraining time on October 13–14, 2026.

| Run ID | Method & Role | Code Hash | Checkpoint SHA-256 (Full 64-char) | Encoder SHA-256 (Full 64-char) | Resumed? | Manifest Source |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **`bt_s0`** | Standard BT (Baseline) | `8c273e7afb33` | `a1017b4cbbb05a61cbb74a3b477cd1c0ad753c11a4d3da01764ca97083647808` | `28594d984a00544352390242a0b2dc76d6d28516d0f28fc1cab430a9389b0c9c` | No | [`bt_s0/manifest.json`](bt_s0/manifest.json) |
| **`no_rr_s0`** | Invariance Only ($\lambda=0$) | `8c273e7afb33` | `a2b40775cb7f77edcbf418fa915867d41be843d57599a102e06a0cbbf886e9a6` | `89a588ee73e4c3e62226e399796ce08dc4cef0753b137d2d66371525f34820b5` | Yes* | [`no_rr_s0/manifest.json`](no_rr_s0/manifest.json) |
| **`fixed_s0`** | Static Hinge ($T=0.05^2$) | `8c273e7afb33` | `eb8bdc524efa260273a984536723cd45b41d934d07f2f7ed6b6c7bd744e2a49b` | `73d84e32efe38f9ff776ade1c066f0cdfcb96179cbf058f365dc4499338a09f5` | No | [`fixed_s0/manifest.json`](fixed_s0/manifest.json) |
| **`scalar_s0`** | Weak $\lambda=0.001$ | `8c273e7afb33` | `9ebe04ba9d34e2b26d1c3db420b203ce9c29c37fac6eae92036112b73092b12c` | `ffe478171e8b4abbb567f73b6fb2123869d0e9d85564d02a7a35d4016b62eee6` | No | [`scalar_s0/manifest.json`](scalar_s0/manifest.json) |
| **`noise_s0`** | Noise-Aware Pilot | `8c273e7afb33` | `76d1125e8193dad4ca1fc58a6315ac328b2c2d98c19e944a4f266a09a1eee603` | `8d1536ebc672b182da3b8451f02338fe57ae48f26a91f968b321854a39490812` | No | [`noise_s0/manifest.json`](noise_s0/manifest.json) |

*\*Note on `no_rr_s0` Resumption:* The `no_rr_s0` experiment was resumed from checkpoint at epoch 38 and completed through epoch 49. The measured runtime across all 50 individual epochs in `loss_log.json` totals 5,587.64 seconds (93.13 minutes), while the final resumed session recorded 1,260.28 seconds (21.00 minutes) in `manifest.json`.

---

## 2. Resolution of Hash Discrepancy

Earlier intermediate drafts (such as the results ledger in commit `9fba0db`) listed provisional placeholder hashes (`43fe535a9032` for code hash, and entries like `83c847fcbaea...` for checkpoints). Forensic verification of the original run artifacts revealed:

1. **Authentic Code Hash:** All five experiments (`bt_s0`, `no_rr_s0`, `fixed_s0`, `scalar_s0`, and `noise_s0`) were trained sequentially between October 13, 2026 (21:28 IST) and October 14, 2026 (07:08 IST) using the identical training script version (`code_hash: 8c273e7afb33`).
2. **Authentic Binary Checkpoint Hashes:** The 12-character prefix recorded by `barlow_local.py` in every contemporaneous `manifest.json` (`a1017b4cbbb0`, `a2b40775cb7f`, `eb8bdc524efa`, `9ebe04ba9d34`, `76d1125e8193`) perfectly matches the fresh, full 64-character SHA-256 calculation of the original physical checkpoint files on disk. The values in earlier documentation tables were unverified draft placeholders and have been corrected to reflect the true historical artifacts.

---

## 3. Provenance Limitations

- **Binary Checkpoints Excluded:** Full PyTorch checkpoint files (`checkpoint.pth`, ~150–300 MB each) and backbone weights (`encoder.pth`, ~45 MB each) are archived in local laboratory storage and excluded from Git tracking to maintain a lightweight repository.
- **Verification via SHA-256:** Independent integrity verification is supported via the 64-character SHA-256 hashes recorded above and the complete execution manifests and evaluation logs tracked in this directory.
