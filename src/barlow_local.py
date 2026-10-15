"""
barlow_local.py — Reduced-scale Barlow Twins on CIFAR-10
=========================================================
Course: DL-070 Deep Learning Laboratory
Author: Khagesh Ranjan (Roll: 24cs2021)
Institution: Rajiv Gandhi Institute of Petroleum Technology (RGIPT)
Assignment: Paper Reproduction and Ablation Study of Barlow Twins (ICML 2021)

Reference:
  Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy Reduction",
  International Conference on Machine Learning (ICML), 2021.
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1. Imports and Constants
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

# Force UTF-8 stream handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.neighbors import KNeighborsClassifier
    HAS_SKL = True
except ImportError:
    HAS_SKL = False

# Default Hyperparameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_DIM = 512
DEFAULT_LR = 0.05
DEFAULT_LAMBD = 0.0051
DEFAULT_EPOCHS = 50
DEFAULT_MAX_NORM = 5.0
DEFAULT_SEED = 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. Reproducibility and Device Utilities
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 0) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def device_for(requested: str = "cuda") -> torch.device:
    """Resolve and verify device availability (asserts CUDA when requested)."""
    requested = requested.lower().strip()
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a CUDA-enabled GPU.")
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(dev)
    free, total = torch.cuda.mem_get_info(dev)
    print(f"[Device] {name} | Free VRAM: {free/1e9:.2f} GB / {total/1e9:.2f} GB")
    return dev


def require_cuda(device: torch.device) -> None:
    """Assert device is CUDA for hardware-accelerated training."""
    if device.type != "cuda":
        raise RuntimeError(f"Training requires CUDA device, got '{device.type}'.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CIFAR-10 Augmentations and Data Loading
# ══════════════════════════════════════════════════════════════════════════════

class TwoCropsTransform:
    """
    Generate two stochastically augmented views (x_A, x_B) of the same input image.
    Follows Barlow Twins augmentation policy adapted for 32x32 CIFAR-10 images.
    """
    def __init__(self, color_jitter: bool = True):
        cifar_mean = (0.4914, 0.4822, 0.4465)
        cifar_std  = (0.2470, 0.2435, 0.2616)

        base_ops_a = [
            T.RandomResizedCrop(32, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
        ]
        base_ops_b = [
            T.RandomResizedCrop(32, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
        ]

        if color_jitter:
            base_ops_a += [
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            ]
            base_ops_b += [
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
                T.RandomSolarize(threshold=128, p=0.2),
            ]

        norm_ops = [T.ToTensor(), T.Normalize(mean=cifar_mean, std=cifar_std)]
        self.transform_a = T.Compose(base_ops_a + norm_ops)
        self.transform_b = T.Compose(base_ops_b + norm_ops)

    def __call__(self, x):
        return self.transform_a(x), self.transform_b(x)


def build_cifar10(data_dir: str = "data",
                  color_jitter: bool = True,
                  subset: int = -1,
                  num_workers: int = 0,
                  batch_size: int = 64,
                  seed: int = 0,
                  include_test: bool = True):
    """
    Construct CIFAR-10 loaders:
    - Pretraining train split: 45,000 images (TwoCropsTransform)
    - Evaluation val split:     5,000 images (Single standard transform)
    - Evaluation test split:   10,000 images (Single standard transform, optional)
    """
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)

    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=cifar_mean, std=cifar_std),
    ])

    raw_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(raw_train))
    val_indices   = indices[:5000].tolist()
    train_indices = indices[5000:].tolist()

    if subset > 0:
        train_indices = train_indices[:subset]

    train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False,
                                           transform=TwoCropsTransform(color_jitter=color_jitter))
    val_ds   = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False,
                                           transform=eval_transform)

    train_subset = Subset(train_ds, train_indices)
    val_subset   = Subset(val_ds, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = None
    if include_test:
        raw_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                                transform=eval_transform)
        test_loader = DataLoader(
            raw_test, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# 4. Encoder and Projector Networks
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_resnet18_for_cifar() -> nn.Module:
    """Adapt ResNet-18 for 32x32 CIFAR-10 input (conv1: 3x3 s1 p1, no maxpool)."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Identity()
    return model


class BarlowEncoder(nn.Module):
    """
    Barlow Twins model:
    - Backbone: CIFAR-adapted ResNet-18 producing 512-dim representation h.
    - Projector: 3-layer MLP (512 -> hidden -> hidden -> dim) with BatchNorm & ReLU.
    """
    def __init__(self, dim: int = 512, hidden: int = 512):
        super().__init__()
        self.backbone = _adapt_resnet18_for_cifar()
        enc_dim = 512

        self.projector = nn.Sequential(
            nn.Linear(enc_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim, bias=False),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frozen 512-dim representations used for downstream evaluation."""
        h = self.backbone(x)
        return h.flatten(1)

    def forward(self, x: torch.Tensor):
        """Forward pass returns representation h and projection embedding z."""
        h = self.encode(x)
        z = self.projector(h)
        return h, z


# ══════════════════════════════════════════════════════════════════════════════
# 5. Cross-Correlation Matrix Computation
# ══════════════════════════════════════════════════════════════════════════════

def _standardize(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Minibatch feature standardization: center and scale each coordinate across batch N.
    Shape: [N, D] -> [N, D]
    """
    return (z - z.mean(dim=0)) / (z.std(dim=0) + eps)


def _off_diag(C: torch.Tensor) -> torch.Tensor:
    """Extract flattened view of all D*(D-1) off-diagonal elements of square matrix C."""
    n = C.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=C.device)
    return C[mask]


def compute_cross_correlation(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Compute empirical cross-correlation matrix C between standardized embeddings.
    C_ij = (1 / N) * sum_n (z1_hat_{n,i} * z2_hat_{n,j})
    """
    N = z1.size(0)
    A = _standardize(z1, eps=eps)
    B = _standardize(z2, eps=eps)
    return torch.matmul(A.T, B) / N


# ══════════════════════════════════════════════════════════════════════════════
# 6. Barlow Twins Loss
# ══════════════════════════════════════════════════════════════════════════════

def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambd: float = 0.0051, eps: float = 1e-5) -> torch.Tensor:
    """
    Barlow Twins loss function (Zbontar et al., ICML 2021).
    
    Principles:
      - z1 and z2 are representations of two augmentations of the same images.
      - diagonal entries measure agreement between corresponding dimensions (pushed toward 1).
      - off-diagonal entries measure redundancy between different dimensions (pushed toward 0).
      - no negative image pairs are required.
    
    Equation:
      Loss = sum_i (1 - C_ii)^2 + lambd * sum_i sum_{j != i} C_ij^2
    """
    C = compute_cross_correlation(z1, z2, eps=eps)

    # Invariance term: forces diagonal correlations to 1
    on_diag = (C.diagonal().add(-1.0)).pow(2).sum()

    # Redundancy reduction term: forces off-diagonal correlations to 0
    off_diag = _off_diag(C).pow(2).sum()

    return on_diag + lambd * off_diag


def barlow_loss_bt(z_a: torch.Tensor, z_b: torch.Tensor, lambd: float = 0.0051) -> torch.Tensor:
    """Alias for standard Barlow Twins loss."""
    return barlow_twins_loss(z_a, z_b, lambd=lambd)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Ablation and Control Losses
# ══════════════════════════════════════════════════════════════════════════════

def barlow_loss_no_rr(z_a: torch.Tensor, z_b: torch.Tensor, lambd: float = 0.0051) -> torch.Tensor:
    """
    Invariance-only ablation (no_rr):
    Strictly omits the off-diagonal redundancy reduction penalty (lambda = 0).
    Loss = sum_i (1 - C_ii)^2
    """
    C = compute_cross_correlation(z_a, z_b)
    return (C.diagonal().add(-1.0)).pow(2).sum()


def barlow_loss_scalar(z_a: torch.Tensor, z_b: torch.Tensor, lambd: float = 0.001) -> torch.Tensor:
    """
    Scalar attenuation control:
    Evaluates sensitivity to weakened regularization strength (lambda = 0.001 vs canonical 0.0051).
    """
    return barlow_twins_loss(z_a, z_b, lambd=lambd)


def barlow_loss_fixed(z_a: torch.Tensor, z_b: torch.Tensor, lambd: float = 0.0051, threshold: float = 0.05) -> torch.Tensor:
    """
    Static hinge threshold control:
    Applies constant cutoff T = threshold^2 to off-diagonal correlations:
      rr_loss = sum_{i != j} ReLU(C_ij^2 - T)
    Correlations |C_ij| <= threshold receive zero gradient, inducing rank-0 collapse.
    """
    C = compute_cross_correlation(z_a, z_b)
    on_diag = (C.diagonal().add(-1.0)).pow(2).sum()
    sq = _off_diag(C).pow(2)
    T = threshold ** 2
    rr_loss = torch.relu(sq - T).sum()
    return on_diag + lambd * rr_loss


def barlow_loss_noise(z_a: torch.Tensor, z_b: torch.Tensor, lambd: float = 0.0051, beta: float = 1.0, floor: float = 0.1) -> torch.Tensor:
    """
    Noise-aware pilot control:
    Estimates per-coordinate sampling error using non-overlapping half-batch differences (C1 - C2) / 2.
    """
    N = z_a.size(0)
    assert N >= 2, "Batch size too small for half-batch noise estimation."
    half = N // 2

    A = _standardize(z_a)
    B = _standardize(z_b)
    C = torch.matmul(A.T, B) / N

    C1 = torch.matmul(A[:half].T, B[:half]) / half
    C2 = torch.matmul(A[half:].T, B[half:]) / (N - half)

    T = beta * _off_diag((C1 - C2).pow(2) / 4.0).detach()
    sq = _off_diag(C).pow(2)

    rr_loss = (floor * sq + (1.0 - floor) * torch.relu(sq - T)).sum()
    on_diag = (C.diagonal().add(-1.0)).pow(2).sum()
    return on_diag + lambd * rr_loss


def compute_loss(method: str, za: torch.Tensor, zb: torch.Tensor, lambd: float = 0.0051,
                 floor: float = 0.1, beta: float = 1.0) -> tuple:
    """
    Unified loss dispatch across all supported experimental variants.
    Returns: (loss_tensor, diagnostic_info_dict)
    """
    method = method.lower().strip()
    if method == "bt":
        loss = barlow_twins_loss(za, zb, lambd=lambd)
    elif method == "no_rr":
        loss = barlow_loss_no_rr(za, zb, lambd=lambd)
    elif method == "scalar":
        loss = barlow_loss_scalar(za, zb, lambd=lambd)
    elif method == "fixed":
        loss = barlow_loss_fixed(za, zb, lambd=lambd)
    elif method == "noise":
        loss = barlow_loss_noise(za, zb, lambd=lambd, beta=beta, floor=floor)
    else:
        raise ValueError(f"Unknown method '{method}'. Supported: bt, no_rr, scalar, fixed, noise")

    return loss, {}


# ══════════════════════════════════════════════════════════════════════════════
# 8. Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def pretrain(args: argparse.Namespace) -> None:
    """
    Main self-supervised pretraining routine on CIFAR-10.
    Implements standard PyTorch Automatic Mixed Precision (AMP) with GradScaler
    and gradient norm clipping at max_norm=5.0.
    """
    device = device_for(args.device)
    require_cuda(device)
    set_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_manifest(out_dir, args, status="running")

    train_loader, val_loader, _ = build_cifar10(
        data_dir    = args.data,
        color_jitter = not getattr(args, "no_color", False),
        subset      = args.subset,
        num_workers = args.workers,
        batch_size  = args.batch_size,
        seed        = args.seed,
    )

    model = BarlowEncoder(dim=args.dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    ckpt_path = out_dir / "checkpoint.pth"
    loss_log = []
    log_path = out_dir / "loss_log.json"

    if ckpt_path.exists():
        print(f"[Train] Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        if log_path.exists():
            loss_log = json.loads(log_path.read_text())

    print(f"\n[Train] Method={args.method} | Epochs={args.epochs} | Batch={args.batch_size} | Dim={args.dim} | Seed={args.seed} | AMP={args.amp}")

    skipped_amp = 0
    total_optimizer_updates = 0

    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        t_epoch    = time.time()
        last_grad_norm = 0.0

        for (xa, xb), _ in train_loader:
            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=args.amp):
                _, za = model(xa)
                _, zb = model(xb)

            # Correlation computation runs in FP32 for numerical stability
            with torch.amp.autocast("cuda", enabled=False):
                za_f = za.float()
                zb_f = zb.float()
                loss, _ = compute_loss(
                    args.method, za_f, zb_f,
                    lambd=args.lambd,
                    floor=getattr(args, "floor", 0.1),
                    beta=getattr(args, "beta", 1.0) or 1.0
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            max_norm = getattr(args, "max_norm", 5.0)
            if max_norm is not None and max_norm > 0:
                last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm).item()

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()

            if scale_after < scale_before:
                skipped_amp += 1
            else:
                total_optimizer_updates += 1
                epoch_loss += loss.item()
                n_batches  += 1

            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now   = scheduler.get_last_lr()[0]
        elapsed  = time.time() - t_epoch
        mem_mb   = torch.cuda.max_memory_allocated(device) / 1e6

        log_entry = {
            "epoch": epoch,
            "loss": round(avg_loss, 6),
            "lr": round(lr_now, 8),
            "epoch_seconds": round(elapsed, 2),
            "peak_mem_mb": round(mem_mb, 1),
            "skipped_amp": skipped_amp,
            "optimizer_updates": total_optimizer_updates,
            "amp_scale": scale_after,
            "grad_norm_before_clip": round(last_grad_norm, 4),
        }
        loss_log.append(log_entry)

        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}]  Loss: {avg_loss:.4f}  LR: {lr_now:.5f}  Time: {elapsed:.1f}s  VRAM: {mem_mb:.1f}MB")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        log_path.write_text(json.dumps(loss_log, indent=2))

    total_time = time.time() - t_start
    torch.save(model.backbone.state_dict(), out_dir / "encoder.pth")
    save_manifest(out_dir, args, status="complete", pretrain_seconds=round(total_time, 2))
    print(f"\n[Train] Pretraining complete in {total_time/60:.2f} minutes. Encoder saved to {out_dir / 'encoder.pth'}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(model: BarlowEncoder, loader: DataLoader, device: torch.device):
    """
    Extract frozen 512-dim representations from the ResNet-18 backbone.
    Returns: (features_np, labels_np)
    """
    model.eval()
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            feats = model.encode(x)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(y.numpy())

    return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# 10. Linear Probe Evaluation
# ══════════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    """Linear classification head (512 -> 10 classes) evaluated on frozen features."""
    def __init__(self, in_features: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_linear_probe(train_feats, train_labels, val_feats, val_labels,
                       device: torch.device, candidate_lrs=None,
                       probe_epochs: int = 50, seed: int = 0):
    """
    Train a linear probe over candidate learning rates using the validation split.
    Hyperparameter tuning is strictly isolated to the validation split.
    """
    if candidate_lrs is None:
        candidate_lrs = [0.01, 0.05, 0.1, 0.3]

    set_seed(seed)

    X_tr = torch.FloatTensor(train_feats)
    y_tr = torch.LongTensor(train_labels)
    X_vl = torch.FloatTensor(val_feats).to(device)
    y_vl = torch.LongTensor(val_labels).to(device)

    tr_ds     = torch.utils.data.TensorDataset(X_tr, y_tr)
    tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True, drop_last=False)

    best_val_acc = -1.0
    best_probe   = None
    best_lr      = None
    all_results  = {}

    for lr in candidate_lrs:
        probe = LinearProbe().to(device)
        opt   = optim.SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        for _ in range(probe_epochs):
            probe.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                F.cross_entropy(probe(xb), yb).backward()
                opt.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(X_vl).argmax(1)
            acc   = (preds == y_vl).float().mean().item()

        all_results[lr] = round(acc * 100, 3)
        print(f"  [Probe] LR={lr:.4f}  Val Acc={acc*100:.2f}%")

        if acc > best_val_acc:
            best_val_acc = acc
            best_probe   = probe
            best_lr      = lr

    return best_probe, best_lr, best_val_acc, all_results


# ══════════════════════════════════════════════════════════════════════════════
# 11. k-Nearest Neighbors and Rank Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_knn(train_feats, train_labels, test_feats, test_labels, k: int = 20) -> float:
    """
    Non-parametric k-NN classification using cosine distance in feature space.
    Evaluates topological representation clustering without training parameters.
    """
    if not HAS_SKL:
        print("[Warning] scikit-learn not available; skipping k-NN evaluation.")
        return 0.0

    # L2-normalize features for cosine metric
    tr_norm = train_feats / (np.linalg.norm(train_feats, axis=1, keepdims=True) + 1e-8)
    te_norm = test_feats / (np.linalg.norm(test_feats, axis=1, keepdims=True) + 1e-8)

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(tr_norm, train_labels)
    acc = knn.score(te_norm, test_labels)
    return round(float(acc) * 100, 3)


def effective_rank(features: np.ndarray, threshold: float = 1e-3) -> int:
    """
    Compute effective feature rank via Singular Value Decomposition (SVD).
    Counts singular values exceeding threshold relative to max singular value.
    """
    s = np.linalg.svd(features, compute_uv=False)
    s_norm = s / (s[0] + 1e-12)
    return int((s_norm > threshold).sum())


# ══════════════════════════════════════════════════════════════════════════════
# 12. Evaluation Dispatch and Result Reporting
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args: argparse.Namespace) -> None:
    """
    Downstream representation evaluation:
    1. Extracts frozen features from pre-trained ResNet-18 backbone.
    2. Tunes linear probe on 5k validation split; evaluates on 10k test split.
    3. Evaluates 20-NN cosine classification and effective feature rank.
    """
    device = device_for(args.device)
    set_seed(args.seed)

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    train_args = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        train_args = manifest.get("args", {})
        print(f"[Eval] Run: Method={train_args.get('method')} | Dim={train_args.get('dim')} | Seed={train_args.get('seed')}")

    dim = int(train_args.get("dim", 512))
    model = BarlowEncoder(dim=dim).to(device)

    encoder_path = run_dir / "encoder.pth"
    ckpt_path    = run_dir / "checkpoint.pth"

    if encoder_path.exists():
        model.backbone.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"[Eval] Loaded encoder from {encoder_path}")
    elif ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[Eval] Loaded model from {ckpt_path}")
    else:
        raise FileNotFoundError(f"No checkpoint or encoder found in {run_dir}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    seed = int(train_args.get("seed", 0))
    val_only = getattr(args, "val_only", False)
    include_test = not val_only

    train_loader, val_loader, test_loader = build_cifar10(
        data_dir    = getattr(args, "data", "data"),
        color_jitter = True,
        num_workers = getattr(args, "workers", 0),
        batch_size  = 256,
        seed        = seed,
        include_test = include_test
    )

    t0 = time.time()
    print("[Eval] Extracting representations from frozen encoder ...")
    X_train, y_train = extract_features(model, train_loader, device)
    X_val,   y_val   = extract_features(model, val_loader, device)

    X_test, y_test = None, None
    if include_test and test_loader is not None:
        X_test, y_test = extract_features(model, test_loader, device)
    extract_sec = time.time() - t0
    print(f"[Eval] Features extracted in {extract_sec:.1f}s: Train={X_train.shape}, Val={X_val.shape}")

    # Feature rank
    val_rank = effective_rank(X_val)
    print(f"[Eval] Effective Feature Rank: {val_rank} / {X_val.shape[1]}")

    # Linear probe
    candidate_lrs = [float(x) for x in args.probe_lrs.split(",")]
    best_probe, best_lr, val_acc, probe_grid = train_linear_probe(
        X_train, y_train, X_val, y_val, device=device,
        candidate_lrs=candidate_lrs, probe_epochs=args.probe_epochs, seed=args.seed
    )

    test_acc = None
    knn_test = None
    if include_test and X_test is not None:
        best_probe.eval()
        with torch.no_grad():
            preds = best_probe(torch.FloatTensor(X_test).to(device)).argmax(1)
            test_acc = round(float((preds == torch.LongTensor(y_test).to(device)).float().mean().item()) * 100, 3)
        knn_test = evaluate_knn(X_train, y_train, X_test, y_test, k=20)

    knn_val = evaluate_knn(X_train, y_train, X_val, y_val, k=20)

    results = {
        "val_acc": round(val_acc * 100, 3),
        "test_acc": test_acc,
        "best_probe_lr": best_lr,
        "probe_grid": probe_grid,
        "knn_val_acc": knn_val,
        "knn_test_acc": knn_test,
        "effective_rank": val_rank,
        "total_dim": X_val.shape[1],
        "eval_seconds": round(time.time() - t0, 2),
    }

    out_file = run_dir / "eval_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\n[Eval Results] Val Probe: {results['val_acc']}% | Test Probe: {results['test_acc']}% | kNN Test: {knn_test}% | Rank: {val_rank}/512")
    print(f"[Eval Results] Saved to {out_file}")


def save_manifest(out_dir: Path, args: argparse.Namespace, status: str = "running", pretrain_seconds: float = None) -> None:
    """Save execution parameters and metadata to manifest.json."""
    manifest = {
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": vars(args),
    }
    if pretrain_seconds is not None:
        manifest["pretrain_seconds"] = pretrain_seconds
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def summarize(args: argparse.Namespace) -> None:
    """Print consolidated results table from all run directories."""
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    print(f"\n{'Run':<20} {'Method':<10} {'Val Acc':<10} {'Test Acc':<10} {'kNN Test':<10} {'Rank':<10}")
    print("-" * 75)

    for r in sorted(root.iterdir()):
        if not r.is_dir():
            continue
        ev = r / "eval_results.json"
        mp = r / "manifest.json"
        if not ev.exists():
            continue
        res = json.loads(ev.read_text())
        man = json.loads(mp.read_text()).get("args", {}) if mp.exists() else {}
        print(f"{r.name:<20} {man.get('method', '?'):<10} {res.get('val_acc', 'N/A')!s:<10} {res.get('test_acc', 'N/A')!s:<10} {res.get('knn_test_acc', 'N/A')!s:<10} {res.get('effective_rank', 'N/A')!s:<10}")


def plot_loss_curves(runs_root: str, out_dir: str) -> None:
    """Generate consolidated training loss curve figure."""
    if not HAS_MPL:
        print("[Warning] matplotlib not installed; skipping plot generation.")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for run_dir in sorted(Path(runs_root).iterdir()):
        log_path = run_dir / "loss_log.json"
        if not log_path.exists():
            continue
        log = json.loads(log_path.read_text())
        mp = run_dir / "manifest.json"
        man = json.loads(mp.read_text()).get("args", {}) if mp.exists() else {}

        epochs = [e["epoch"] + 1 for e in log]
        losses = [e["loss"] for e in log]
        ax.plot(epochs, losses, label=f"{man.get('method', run_dir.name)} (seed={man.get('seed', 0)})")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Composite Training Loss", fontsize=11)
    ax.set_title("Barlow Twins Convergence on CIFAR-10", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = Path(out_dir) / "loss_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Loss curves saved to {out_path}")


def run_selftest(device: torch.device) -> None:
    """Run CUDA verification, forward/backward pass, and numerical sanity checks."""
    print(f"[Selftest] Testing device: {device}")
    model = BarlowEncoder(dim=128).to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    h, z = model(x)
    assert h.shape == (4, 512), f"Unexpected representation shape: {h.shape}"
    assert z.shape == (4, 128), f"Unexpected embedding shape: {z.shape}"

    loss = barlow_twins_loss(z, z)
    loss.backward()
    print("[Selftest] Forward/backward pass passed successfully.")
    print("[Selftest] All core architecture checks OK.")


# ══════════════════════════════════════════════════════════════════════════════
# 13. CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Barlow Twins CIFAR-10 Reproduction (DL-070)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # selftest
    p_self = sub.add_parser("selftest", help="Run sanity checks.")
    p_self.add_argument("--device", default="cuda", help="Target device.")

    # train
    p_tr = sub.add_parser("train", help="Pretrain Barlow Twins encoder on CIFAR-10.")
    p_tr.add_argument("--method", default="bt", choices=["bt", "no_rr", "scalar", "fixed", "noise"],
                      help="Pretraining method variant.")
    p_tr.add_argument("--device", default="cuda")
    p_tr.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision.")
    p_tr.add_argument("--epochs", type=int, default=50)
    p_tr.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p_tr.add_argument("--dim", type=int, default=512, help="Projector output dimension.")
    p_tr.add_argument("--lr", type=float, default=0.05, help="Initial learning rate.")
    p_tr.add_argument("--lambd", type=float, default=0.0051, help="Off-diagonal weight lambda.")
    p_tr.add_argument("--beta", type=float, default=1.0, help="Noise threshold scale factor.")
    p_tr.add_argument("--floor", type=float, default=0.1, help="Minimum residual penalty fraction.")
    p_tr.add_argument("--max-norm", type=float, default=5.0, dest="max_norm", help="Gradient clipping norm.")
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.add_argument("--subset", type=int, default=-1, help="Subset size (-1 for full 45k).")
    p_tr.add_argument("--workers", type=int, default=0, help="DataLoader workers.")
    p_tr.add_argument("--data", default="data", help="Directory for CIFAR-10 data.")
    p_tr.add_argument("--out", required=True, help="Output directory for checkpoints/logs.")
    p_tr.add_argument("--no-color", action="store_true", dest="no_color", help="Disable color augmentations.")

    # evaluate
    p_ev = sub.add_parser("evaluate", help="Downstream frozen feature evaluation.")
    p_ev.add_argument("--run", required=True, help="Path to run directory.")
    p_ev.add_argument("--device", default="cuda")
    p_ev.add_argument("--probe-epochs", type=int, default=50, dest="probe_epochs")
    p_ev.add_argument("--probe-lrs", default="0.01,0.05,0.1,0.3", dest="probe_lrs")
    p_ev.add_argument("--seed", type=int, default=0)
    p_ev.add_argument("--data", default="data")
    p_ev.add_argument("--workers", type=int, default=0)
    p_ev.add_argument("--val-only", action="store_true", dest="val_only", help="Evaluate validation set only.")

    # summarize
    p_sm = sub.add_parser("summarize", help="Summarize results across runs.")
    p_sm.add_argument("--root", required=True, help="Root runs directory.")

    # plot
    p_pl = sub.add_parser("plot", help="Plot loss curves.")
    p_pl.add_argument("--root", required=True)
    p_pl.add_argument("--out", default="figures")

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    try:
        if args.command == "selftest":
            dev = device_for(args.device)
            run_selftest(dev)
        elif args.command == "train":
            pretrain(args)
        elif args.command == "evaluate":
            evaluate(args)
        elif args.command == "summarize":
            summarize(args)
        elif args.command == "plot":
            plot_loss_curves(args.root, args.out)
        else:
            parser.print_help()
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
