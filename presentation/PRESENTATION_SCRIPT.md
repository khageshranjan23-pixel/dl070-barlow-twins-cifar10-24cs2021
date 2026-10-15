# DL-070 Barlow Twins Presentation Script

**Presentation Duration:** ~10 Minutes  
**Course:** DL-070 Deep Learning Laboratory  
**Topic:** Reduced-Scale Reproduction and Ablation Study of Barlow Twins on CIFAR-10  
**Slide Deck:** `DL070_BarlowTwins_Presentation.pptx` (12 Slides)

---

## Slide 1: Title & Introduction (0:00 – 0:45)
**Visual:** Title slide with DL-070 metadata, candidate attribution Khagesh Ranjan (24cs2021), Rajiv Gandhi Institute of Petroleum Technology (RGIPT), and local NVIDIA RTX 5050 hardware specification.

**Spoken Script:**
> "Good morning, Professor and members of the panel. Today, I am presenting my DL-070 laboratory project: a reduced-scale reproduction and comprehensive ablation study of the Barlow Twins self-supervised learning algorithm on CIFAR-10.
> 
> In this project, we reproduced the core self-supervised learning dynamics of Barlow Twins on a local consumer GPU—an NVIDIA GeForce RTX 5050 Laptop GPU. We investigated the mathematical mechanism that prevents representation collapse without negative sample pairs, conducted rigorous ablation experiments isolating the redundancy reduction objective, and uncovered critical failure modes that lead to dimensional collapse.
> 
> Let us begin by examining the self-supervised learning landscape and the core challenge that Barlow Twins addresses."

---

## Slide 2: Context & Motivation: The Collapse Challenge (0:45 – 1:40)
**Visual:** Three structured cards contrasting Contrastive SSL (SimCLR), Asymmetric SSL (BYOL), and Barlow Twins.

**Spoken Script:**
> "In self-supervised representation learning from images, the primary technical hurdle is preventing representation collapse—the degenerate trivial state where an encoder maps all inputs to a constant representation.
> 
> As shown on this slide, existing paradigms approached this in two main ways:
> First, contrastive methods like SimCLR prevent collapse by comparing instance pairs across a minibatch. However, to get diverse negative samples, SimCLR requires massive batch sizes—typically 4,096 images—making it inaccessible on consumer hardware.
> Second, asymmetric methods like BYOL and SimSiam eliminate negative samples but rely on architectural tricks: momentum encoder target networks or heuristic stop-gradient operations.
> 
> Barlow Twins, proposed by Zbontar et al. at ICML 2021, introduces a fundamentally different paradigm. It eliminates both negative pairs and architectural asymmetries. Instead, it operates directly on the cross-correlation matrix between feature dimensions, drawing its foundation from Horace Barlow's 1961 biological hypothesis of sensory redundancy reduction."

---

## Slide 3: The Barlow Twins Principle & Objective (1:40 – 2:40)
**Visual:** The Barlow Twins loss equation with dual callouts for the Invariance Term ($C_{ii} \to 1$) and the Redundancy Reduction Term ($C_{ij} \to 0$).

**Spoken Script:**
> "Here is the core objective function. We pass two distorted views of a batch through twin weight-sharing encoders and projectors, producing standardized embeddings $\hat{Z}^A$ and $\hat{Z}^B$. We then compute their $D \times D$ cross-correlation matrix $C$.
> 
> The loss consists of two complementary components:
> 1. The **Invariance Term** on the diagonal: $\sum_i (1 - C_{ii})^2$. This forces corresponding feature dimensions across the two augmented views to have unit correlation, ensuring that representations are invariant to data augmentations.
> 2. The **Redundancy Reduction Term** on the off-diagonals: $\lambda \sum_{i \ne j} C_{ij}^2$, scaled by $\lambda = 0.0051$. This penalizes correlations between different feature coordinates, forcing individual dimensions to capture statistically independent concepts.
> 
> Together, these terms push the cross-correlation matrix toward the identity matrix, algebraically preventing dimensional collapse without requiring negative pairs."

---

## Slide 4: Architecture & Pipeline Adaptation (2:40 – 3:35)
**Visual:** Four horizontal pipeline blocks showing Two-Crop Transforms, ResNet-18 adaptation, 3-layer Projector MLP, and Standardization.

**Spoken Script:**
> "To train Barlow Twins effectively on CIFAR-10 at $32 \times 32$ resolution, we adapted the standard ResNet-18 architecture:
> 1. The original ImageNet $7 \times 7$ stride-2 convolution was replaced with a $3 \times 3$ stride-1 convolution, and the initial max-pooling layer was removed to prevent premature spatial downsampling.
> 2. The backbone outputs a 512-dimensional representation vector $h$.
> 3. During pretraining, $h$ passes through a 3-layer non-linear projector MLP with Batch Normalization and ReLU, producing a 512-dimensional embedding $z$.
> 4. Minibatch feature standardization centers and scales each coordinate across the batch before correlation computation.
> Crucially, downstream evaluation uses only the frozen backbone representation $h$; the projector is discarded after pretraining."

---

## Slide 5: Experimental Setup & Hardware Protocol (3:35 – 4:25)
**Visual:** Side-by-side cards detailing the RTX 5050 training parameters and the frozen scientific evaluation protocol.

**Spoken Script:**
> "Our experimental setup was designed for reproducibility on consumer hardware. We trained for 50 epochs on the 45,000-image training split using SGD with momentum 0.9, weight decay $10^{-4}$, and a cosine annealing learning rate schedule starting at 0.05.
> 
> We utilized PyTorch Automatic Mixed Precision with careful gradient scaling and gradient norm clipping at 5.0. On our RTX 5050 GPU, each 50-epoch baseline pretraining run completed in 120.2 minutes (2.00 hours) total pretraining time, consuming just 537 megabytes of VRAM. Evaluation required an additional 1.8 minutes.
> 
> For evaluation, we followed a strict protocol: the encoder was frozen, and a linear probe was trained on the frozen representations. Hyperparameter tuning was performed exclusively on the 5,000-sample validation split, and final performance was measured on the held-out 10,000-image test set. We also evaluated non-parametric cosine $k$-NN and representation rank."

---

## Slide 6: Primary Results: Reduced-Scale Reproduction (4:25 – 5:25)
**Visual:** Big stat callouts (74.84% probe, 72.41% k-NN, 506/512 rank) and complete empirical results table.

**Spoken Script:**
> "Turning to our primary results:
> Standard Barlow Twins (`bt_s0`) achieved **74.84% test accuracy** on the linear probe and **72.41% accuracy** on 20-NN cosine classification within the 50-epoch budget.
> The effective feature representation rank reached **506 out of 512**, demonstrating that the representations successfully span nearly the entire embedding space without dimensional collapse.
> 
> This confirms that Barlow Twins functions effectively in a reduced-scale setting on consumer hardware. The core mechanisms documented in the ImageNet paper reproduce reliably on CIFAR-10."

---

## Slide 7: Main Ablation Study: Isolating Redundancy Reduction (5:25 – 6:30)
**Visual:** Side-by-side comparison of the `no_rr` ablation formulation and quantitative impact (-17.72% drop).

**Spoken Script:**
> "To understand what truly drives representation quality, we performed our primary ablation: `no_rr`. We set $\lambda = 0$, completely removing the off-diagonal redundancy reduction penalty. The network was trained solely on the diagonal invariance term.
> 
> The results were dramatic:
> Linear probe test accuracy plummeted from **74.84% down to 57.12%**—a massive **-17.72 percentage point drop**.
> The $k$-NN test accuracy fell by **-20.43 percentage points** to 51.98%.
> 
> Removing redundancy reduction substantially reduced downstream discriminability despite preserving high numerical feature rank. This shows that full rank alone does not guarantee semantically useful or sufficiently decorrelated representations. Without the off-diagonal redundancy penalty, different feature channels become co-linear and redundant, severely degrading downstream linear separability."

---

## Slide 8: Negative Control: Static Hinge Collapse (6:30 – 7:30)
**Visual:** The static hinge formulation and the catastrophic collapse findings (Rank 0, 10.00% accuracy).

**Spoken Script:**
> "We also evaluated a static hinge threshold control (`fixed_s0`). The intuition was: if small cross-correlations are noisy, could we zero out penalties below a threshold of $T = 0.05^2$?
> 
> The outcome was catastrophic:
> Effective feature rank collapsed completely from 506 to **0 / 512**.
> Linear probe accuracy dropped to exactly **10.00%**—pure random guessing on 10 CIFAR-10 classes.
> 
> Why did this happen? In this reduced-scale setup, removing redundancy reduction substantially reduced downstream discriminability. The tested static-threshold configuration produced rank-0 collapse, showing that this particular thresholding strategy failed to maintain a useful representation. When correlations drop below 0.05, the hinge loss produces exactly zero gradient, allowing features to collapse into a 1-dimensional singularity. Continuous redundancy reduction proved critical in this setup."

---

## Slide 9: Empirical Training Dynamics (7:30 – 8:15)
**Visual:** Multi-panel plot showing loss curves, off-diagonal penalty convergence, and effective rank comparisons.

**Spoken Script:**
> "Examining the training dynamics on Slide 9:
> In standard Barlow Twins, the total loss decreases steadily from over 200 down to approximately 60. The diagonal invariance loss rapidly converges as positive view features align.
> Notice the off-diagonal redundancy penalty: it starts high and is progressively driven downward as coordinates decorrelate.
> In contrast, the `no_rr` ablation achieves lower numerical loss because it ignores off-diagonal penalties, but its representations are functionally degraded. The static hinge run collapses early in training as its feature rank drops to zero."

---

## Slide 10: Local Minibatch Limitations & The Noise Dilemma (8:15 – 9:00)
**Visual:** Mathematical analysis of minibatch sample size effects ($N=64$ vs $N=2048$).

**Spoken Script:**
> "This brings us to a key theoretical insight regarding local GPU constraints.
> In the original ImageNet paper, a batch size of 2,048 was used across 64 GPUs. The sample correlation estimation error scales as $1/\sqrt{N}$, which is approximately 0.022.
> However, on our local RTX 5050 with batch size 64, the standard error is roughly 0.125—nearly six times higher.
> 
> In small minibatches, many off-diagonal correlations are transient sampling noise rather than persistent feature redundancies. Yet, as our negative control proved, naive thresholding causes catastrophic collapse.
> Therefore, any adaptation for small batches must strictly preserve continuous gradient flow and total off-diagonal coefficient mass."

---

## Slide 11: Limitations & Future Work (9:00 – 9:40)
**Visual:** Four core future research directions: Multi-Seed Ensembles, Extended Training Schedules (100–200 epochs), Cross-Dataset Transferability (CIFAR-100 / TinyImageNet), and Noise-Resilient Reliability Weighting.

**Spoken Script:**
> "Looking ahead, our findings open several compelling avenues for future research:
> 1. Multi-Seed Statistical Validation: While our individual training runs converged stably, expanding to multi-seed trials with formal hypothesis testing will provide even tighter statistical confidence intervals.
> 2. Extended Training Schedules: Standard Barlow Twins thrives on long schedules. Evaluating 100- to 200-epoch budgets on CIFAR-10 will help quantify how close reduced-scale setups can get to full-scale asymptotic bounds.
> 3. Generalization Across Benchmarks: Scaling this evaluation to CIFAR-100 and TinyImageNet will shed light on how well decorrelated features transfer across fine-grained semantic distributions.
> 4. Noise-Resilient Decorrelation: Developing continuous, mass-conserving reliability weighting schemes to adaptively attenuate small-batch correlation estimation noise without causing threshold-induced collapse remains an exciting theoretical frontier."

---

## Slide 12: Conclusions & Key Takeaways (9:40 – 10:15)
**Visual:** Five core conclusion pillars and Q&A invitation.

**Spoken Script:**
> "To conclude our findings:
> 1. We achieved a successful, reduced-scale reproduction of Barlow Twins on genuine CIFAR-10 using consumer hardware, reaching 74.84% linear probe test accuracy and 72.41% $k$-NN accuracy.
> 2. We demonstrated that redundancy reduction is indispensable: removing it drops accuracy by 17.72 percentage points.
> 3. The tested static-threshold configuration produced rank-0 collapse, showing that this particular thresholding strategy failed to maintain a useful representation, and continuous redundancy reduction proved critical in this setup.
> 4. We verified that Barlow Twins is feasible for local academic research, completing 50 pretraining epochs in 120.2 minutes with 537 MB peak VRAM.
> 5. All code, unit tests, Colab notebooks, and IEEE report deliverables are complete, verified, and publicly available at https://github.com/khageshranjan23-pixel/dl070-barlow-twins-cifar10-24cs2021.
> 
> Thank you for your time. I welcome any questions from the panel."
