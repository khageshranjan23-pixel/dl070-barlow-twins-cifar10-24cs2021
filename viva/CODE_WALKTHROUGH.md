# Code Walkthrough — Barlow Twins CIFAR-10

**Student Viva & Code Walkthrough Reference Document**  
**Course:** DL-070 Deep Learning Laboratory  
**Candidate:** Khagesh Ranjan (Roll: 24cs2021)  
**Institution:** Rajiv Gandhi Institute of Petroleum Technology (RGIPT)  
**Submission:** October 2026  

---

## Big Picture: What Does the Code Do?

This project trains a deep neural network to learn visual representations **without human labels** using self-supervised learning (SSL). Specifically, it implements the Barlow Twins algorithm (Zbontar et al., ICML 2021) adapted for CIFAR-10 on consumer hardware (NVIDIA RTX 5050 Laptop GPU).

The self-supervised pipeline consists of three sequential steps:
1. **Pretrain:** Train the feature encoder on 45,000 unlabelled CIFAR-10 images using the Barlow Twins redundancy-reduction objective.
2. **Evaluate:** Freeze the encoder, extract 512-dimensional representations, train a linear classification probe on the frozen features, and evaluate cosine 20-NN classification.
3. **Summarize & Plot:** Aggregate empirical accuracy, effective representation rank, and convergence loss curves.

---

## Five Functions to Understand Before the Viva

If you only have five minutes before the examination, master these five functions:

1. **`barlow_twins_loss(z1, z2, lambd)`**: Computes the composite loss ($L = L_{\text{inv}} + \lambda L_{\text{rr}}$) by driving diagonal correlations to 1 and off-diagonal correlations to 0 without requiring negative pairs.
2. **`_standardize(z)`**: Centers and normalizes embeddings across the minibatch so cross-correlations act as genuine Pearson correlation coefficients between $[-1, 1]$.
3. **`barlow_loss_no_rr(z_a, z_b)`**: Main ablation setting $\lambda = 0$, evaluating whether invariance alone is sufficient (proves a -17.72% drop without decorrelation).
4. **`TwoCropsTransform.__call__(x)`**: Generates two distinct, stochastically distorted views of the same image to create positive pairs.
5. **`train_linear_probe(...)`**: Trains a linear classifier on frozen representations, sweeping learning rates exclusively on the validation split.

---

## Structured Code Walkthrough: 10 Core Modules

### 1. Data Augmentation
- **Class / Function Name:** `TwoCropsTransform`
- **Purpose:** Applies stochastic transformations to an input image to create two augmented views ($x_A, x_B$) sharing the same semantic identity.
- **Inputs:** `x` (PIL Image or Tensor)
- **Outputs:** Tuple `(view_a, view_b)`
- **Key Statements:**
  ```python
  self.transform_a = T.Compose(base_ops_a + norm_ops)
  self.transform_b = T.Compose(base_ops_b + norm_ops)
  return self.transform_a(x), self.transform_b(x)
  ```
- **Simple Explanation for the Professor:** "Sir, self-supervised learning requires two distorted perspectives of each image. `TwoCropsTransform` applies random cropping, horizontal flip, color jitter, blur, and solarization so the network learns features that are invariant to distortions."
- **Likely Viva Question:** *Why do view A and view B have slightly different augmentations (like solarization only on view B)?*  
  *Answer:* Following Zbontar et al., asymmetrical augmentation strengths prevent the twin networks from exploiting trivial shortcut correlations across identical distortion channels.

---

### 2. Encoder Backbone
- **Class / Function Name:** `_adapt_resnet18_for_cifar` / `BarlowEncoder.encode`
- **Purpose:** Extracts a 512-dimensional representation vector $h$ from $32 \times 32$ images.
- **Inputs:** `x` (Tensor of shape `[N, 3, 32, 32]`)
- **Outputs:** `h` (Tensor of shape `[N, 512]`)
- **Key Statements:**
  ```python
  model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
  model.maxpool = nn.Identity()
  model.fc = nn.Identity()
  ```
- **Simple Explanation for the Professor:** "Standard ResNet-18 was created for ImageNet ($224 \times 224$). To prevent losing all spatial resolution on $32 \times 32$ CIFAR images, we replaced the $7 \times 7$ stride-2 convolution with a $3 \times 3$ stride-1 convolution and removed max-pooling."
- **Likely Viva Question:** *Why discard the final fully connected classification layer?*  
  *Answer:* Because in self-supervised learning there are no class labels during pretraining; we want the raw 512-dimensional penultimate representations.

---

### 3. Projector MLP
- **Class / Function Name:** `BarlowEncoder` (`self.projector`)
- **Purpose:** Maps representation $h$ to projection embedding $z \in \mathbb{R}^{512}$ where the cross-correlation objective is applied.
- **Inputs:** `h` (Tensor `[N, 512]`)
- **Outputs:** `z` (Tensor `[N, 512]`)
- **Key Statements:**
  ```python
  self.projector = nn.Sequential(
      nn.Linear(enc_dim, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
      nn.Linear(hidden, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
      nn.Linear(hidden, dim, bias=False)
  )
  ```
- **Simple Explanation for the Professor:** "The 3-layer MLP acts as an information buffer. It absorbs the strict decorrelation constraints, allowing the backbone representation $h$ to retain general visual information for downstream classification."
- **Likely Viva Question:** *Do we use the projector $z$ during downstream testing?*  
  *Answer:* No, the projector is strictly discarded after pretraining; downstream evaluation uses only the frozen backbone representation $h$.

---

### 4. Cross-Correlation Matrix Computation
- **Class / Function Name:** `compute_cross_correlation` (and `_standardize`)
- **Purpose:** Centers and normalizes embeddings across the minibatch and computes the empirical $D \times D$ cross-correlation matrix.
- **Inputs:** `z1, z2` (Tensors `[N, D]`)
- **Outputs:** `C` (Tensor `[D, D]`)
- **Key Statements:**
  ```python
  A = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + eps)
  B = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + eps)
  C = torch.matmul(A.T, B) / N
  ```
- **Simple Explanation for the Professor:** "First, we standardize each coordinate across all $N$ samples in the batch to have zero mean and unit variance. Then matrix multiplying $A^T B / N$ gives the empirical cross-correlation between every coordinate in view A and view B."
- **Likely Viva Question:** *What happens if you omit the standardization step?*  
  *Answer:* Without standardization, the magnitude of embeddings could scale arbitrarily, destroying the Pearson correlation interpretation and causing numeric instability.

---

### 5. Diagonal Invariance Loss
- **Class / Function Name:** `barlow_twins_loss` (invariance term)
- **Purpose:** Forces diagonal elements $C_{ii} \to 1$, ensuring invariance across augmented views.
- **Inputs:** Cross-correlation matrix `C`
- **Outputs:** Scalar tensor `on_diag`
- **Key Statements:**
  ```python
  on_diag = (C.diagonal().add(-1.0)).pow(2).sum()
  ```
- **Simple Explanation for the Professor:** "The diagonal elements represent the correlation of dimension $i$ with itself across the two augmented views. Pushing $C_{ii}$ towards 1 forces the representations to be invariant to data augmentations."
- **Likely Viva Question:** *What is the gradient of $(C_{ii} - 1)^2$ with respect to $C_{ii}$?*  
  *Answer:* $\frac{\partial}{\partial C_{ii}} (C_{ii} - 1)^2 = 2(C_{ii} - 1)$, which continuously pulls $C_{ii}$ toward 1 until unit correlation is reached.

---

### 6. Off-Diagonal Redundancy Reduction Loss
- **Class / Function Name:** `_off_diag` / `barlow_twins_loss` (redundancy term)
- **Purpose:** Forces off-diagonal elements $C_{ij} \to 0$ for $i \ne j$, decorrelating feature channels.
- **Inputs:** Cross-correlation matrix `C`
- **Outputs:** Scalar tensor `off_diag`
- **Key Statements:**
  ```python
  mask = ~torch.eye(n, dtype=torch.bool, device=C.device)
  off_diag = (C[mask]).pow(2).sum()
  ```
- **Simple Explanation for the Professor:** "The off-diagonal elements represent correlations between different feature dimensions. Pushing $C_{ij}$ towards 0 ensures that individual coordinates capture statistically independent, non-redundant visual properties."
- **Likely Viva Question:** *How many off-diagonal elements are there in a $512 \times 512$ matrix?*  
  *Answer:* Exactly $D \times (D - 1) = 512 \times 511 = 261,632$ off-diagonal elements.

---

### 7. Complete Barlow Twins Objective
- **Class / Function Name:** `barlow_twins_loss`
- **Purpose:** Combines invariance and redundancy reduction into a single scalar objective.
- **Inputs:** `z1, z2` (`[N, D]`), `lambd` (float, default 0.0051)
- **Outputs:** Composite scalar loss tensor
- **Key Statements:**
  ```python
  return on_diag + lambd * off_diag
  ```
- **Simple Explanation for the Professor:** "The full loss balances agreement on the diagonal against decorrelation on the off-diagonal. Parameter $\lambda = 0.0051$ scales down the off-diagonal sum because there are $D-1$ times more off-diagonal terms than diagonal terms."
- **Likely Viva Question:** *Why does Barlow Twins not collapse without negative sample pairs?*  
  *Answer:* Because if representations collapsed to a constant or 1D vector, off-diagonal correlations would equal 1.0, causing the redundancy reduction penalty to explode to $\approx 1334$, algebraically repelling the features outward.

---

### 8. The `no_rr` Main Ablation
- **Class / Function Name:** `barlow_loss_no_rr`
- **Purpose:** Evaluates self-supervised learning dynamics when the off-diagonal penalty is completely removed ($\lambda = 0$).
- **Inputs:** `z_a, z_b` (`[N, D]`)
- **Outputs:** Invariance-only loss
- **Key Statements:**
  ```python
  return (C.diagonal().add(-1.0)).pow(2).sum()
  ```
- **Simple Explanation for the Professor:** "In `no_rr`, we train solely on invariance. Accuracy plummets from 74.84% down to 57.12% (-17.72%). While numerical rank remains 508, coordinates copy redundant information, proving redundancy reduction is indispensable."
- **Likely Viva Question:** *Did `no_rr` suffer from dimensional collapse?*  
  *Answer:* No, `no_rr` retained high algebraic rank (508/512) due to batch standardization, but suffered severe loss in semantic discriminability. Full dimensional collapse (rank 0) occurred only in the static hinge control `fixed_s0`.

---

### 9. Pretraining Loop & AMP Stability
- **Class / Function Name:** `pretrain`
- **Purpose:** Orchestrates 50-epoch training with SGD, cosine annealing, PyTorch GradScaler, and gradient clipping.
- **Inputs:** Parsed CLI arguments (`args`)
- **Outputs:** Checkpoints (`checkpoint.pth`), encoder weights (`encoder.pth`), loss logs (`loss_log.json`)
- **Key Statements:**
  ```python
  with torch.amp.autocast("cuda", enabled=args.amp):
      _, za = model(xa)
      _, zb = model(xb)
  with torch.amp.autocast("cuda", enabled=False):
      loss, _ = compute_loss(args.method, za.float(), zb.float(), lambd=args.lambd)
  scaler.scale(loss).backward()
  scaler.unscale_(optimizer)
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
  scaler.step(optimizer)
  scaler.update()
  ```
- **Simple Explanation for the Professor:** "We forward-pass in FP16 to maximize RTX 5050 throughput, but compute cross-correlations in FP32 for numerical stability. We unscale gradients and clip norms at 5.0 to prevent gradient explosions."
- **Likely Viva Question:** *When does GradScaler skip an optimizer update?*  
  *Answer:* When gradients contain non-finite values (Inf/NaN). `scaler.step(optimizer)` detects this, skips parameter updates, and reduces the scaling factor (`scale_after < scale_before`).

---

### 10. Downstream Linear Probe & k-NN Evaluation
- **Class / Function Name:** `evaluate` (`train_linear_probe`, `evaluate_knn`, `effective_rank`)
- **Purpose:** Measures representation quality on frozen features.
- **Inputs:** Pretrained run directory (`--run`)
- **Outputs:** Validation & test probe accuracy, 20-NN cosine accuracy, effective rank (`eval_results.json`)
- **Key Statements:**
  ```python
  X_train, y_train = extract_features(model, train_loader, device)
  best_probe, best_lr, val_acc, _ = train_linear_probe(X_train, y_train, X_val, y_val, candidate_lrs=[0.01, 0.05, 0.1, 0.3])
  knn_test = evaluate_knn(X_train, y_train, X_test, y_test, k=20)
  val_rank = effective_rank(X_val)
  ```
- **Simple Explanation for the Professor:** "We freeze the encoder, extract 512-dim features, tune the linear probe learning rate on the 5k validation split, and evaluate on the 10k test split. We also run 20-NN cosine classification and SVD rank to measure dimensional spanning."
- **Likely Viva Question:** *Why is hyperparameter tuning performed on the validation split rather than the test split?*  
  *Answer:* To strictly prevent information leakage from the test split, adhering to rigorous empirical machine learning standards.
