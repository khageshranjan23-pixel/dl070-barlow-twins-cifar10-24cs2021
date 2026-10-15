# Experimental Evidence & Provenance

This directory stores the sanitized execution manifests and downstream evaluation records corresponding to the reported results.

## Model Checkpoints & Large Artifacts
Large model checkpoints (`checkpoint.pth`, `encoder.pth`, each ~150–300 MB) and raw CIFAR-10 data batches are intentionally excluded from Git tracking to maintain a lightweight, easily cloneable public repository. The tracked files here preserve full experimental provenance:

- **`provenance.json`**: Top-level index mapping each reported experiment configuration to its method, seed, source code hash, manifest, and evaluation summary.
- **`bt_s0/`**: Baseline Barlow Twins ($D=512$, batch size 64, 50 epochs, $\lambda=0.0051$).
- **`no_rr_s0/`**: Invariance-only ablation ($\lambda=0.0$).
- **`fixed_s0/`**: Static hinge threshold control ($T=0.05^2$).
- **`scalar_s0/`**: Weakened scalar regularizer ($\lambda=0.001$).
- **`noise_s0/`**: Noise-aware adaptive threshold pilot.

Each subfolder contains:
- `manifest.json`: Configuration parameters, hardware specifications, pretraining runtime in seconds, peak VRAM, and checkpoint checksums.
- `eval_results.json`: Downstream linear probe accuracy (over learning rate grid search), $k$-NN cosine accuracy ($k=20$), and effective feature rank computed via SVD.
