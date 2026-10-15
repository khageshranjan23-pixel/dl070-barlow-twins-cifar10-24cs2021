"""
test_core.py — Core Loss and Architecture Tests for Barlow Twins (DL-070 Assignment)
=====================================================================================
Student: Khagesh Ranjan (Roll: 24cs2021)
Institution: Rajiv Gandhi Institute of Petroleum Technology (RGIPT)
Course: DL-070 Deep Learning Laboratory

Verifies:
1. Minibatch feature standardization (mean=0, std=1 across batch dimension).
2. Off-diagonal extraction: exactly D*(D-1) off-diagonal elements.
3. Cross-correlation computation and matrix shape (D x D).
4. Barlow Twins loss numerical calculation (diagonal invariance + off-diagonal redundancy reduction).
5. Invariance-only ablation (no_rr): strictly omits off-diagonal penalty.
6. View-swap symmetry: Loss(A, B) == Loss(B, A).
7. Autograd gradient flow through ResNet-18 backbone and MLP projector.
8. Lightweight checkpoint round-trip saving and loading.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

# Add src to python path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from barlow_local import (
    _standardize,
    _off_diag,
    compute_cross_correlation,
    barlow_twins_loss,
    barlow_loss_bt,
    barlow_loss_no_rr,
    BarlowEncoder,
    extract_features,
    LinearProbe,
    effective_rank,
)


class TestBarlowTwinsCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N = 64
        self.D = 128

    def test_standardize(self):
        """Standardize should ensure zero mean and unit variance along batch dimension."""
        x = torch.randn(self.N, self.D, device=self.device) * 5.0 + 3.0
        x_std = _standardize(x)
        self.assertEqual(x_std.shape, (self.N, self.D))
        mean_err = x_std.mean(dim=0).abs().max().item()
        std_err = (x_std.std(dim=0, unbiased=False) - 1.0).abs().max().item()
        self.assertLess(mean_err, 1e-4, "Standardized mean differs from 0")
        self.assertLess(std_err, 0.02, "Standardized std differs from 1")

    def test_off_diag(self):
        """_off_diag should extract exactly all D*(D-1) off-diagonal elements."""
        M = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        off = _off_diag(M)
        self.assertEqual(off.numel(), 4 * 3)
        expected = torch.tensor([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14], dtype=torch.float32)
        self.assertTrue(torch.equal(off, expected))

    def test_cross_correlation_shape(self):
        """Cross-correlation matrix must have shape (D, D)."""
        z1 = torch.randn(self.N, self.D, device=self.device)
        z2 = torch.randn(self.N, self.D, device=self.device)
        C = compute_cross_correlation(z1, z2)
        self.assertEqual(C.shape, (self.D, self.D))

    def test_loss_identical_representations(self):
        """When za and zb are identical and orthogonal, invariance loss is near zero."""
        za = torch.randn(self.N, self.D, device=self.device)
        zb = za.clone()
        loss = barlow_loss_bt(za, zb, lambd=0.0051)
        C = compute_cross_correlation(za, zb)
        inv_loss = ((C.diagonal() - 1.0).pow(2)).sum().item()
        self.assertLess(inv_loss, 0.1, "Invariance loss is excessively large for identical views")

    def test_no_rr_ablation(self):
        """no_rr must compute strictly the invariance term and omit the redundancy reduction penalty."""
        za = torch.randn(self.N, self.D, device=self.device)
        zb = torch.randn(self.N, self.D, device=self.device)

        loss_no_rr = barlow_loss_no_rr(za, zb, lambd=0.0051)
        loss_bt = barlow_loss_bt(za, zb, lambd=0.0051)

        C = compute_cross_correlation(za, zb)
        expected_inv = ((C.diagonal() - 1.0).pow(2)).sum().item()

        self.assertAlmostEqual(loss_no_rr.item(), expected_inv, places=4)
        self.assertLess(loss_no_rr.item(), loss_bt.item(), "no_rr should be strictly smaller than BT with positive lambda")

    def test_view_swap_invariance(self):
        """Swapping view A and view B must produce identical loss."""
        za = torch.randn(self.N, self.D, device=self.device)
        zb = torch.randn(self.N, self.D, device=self.device)

        loss_ab = barlow_twins_loss(za, zb, lambd=0.0051)
        loss_ba = barlow_twins_loss(zb, za, lambd=0.0051)
        diff = abs(loss_ab.item() - loss_ba.item())
        self.assertLess(diff, 1e-5, f"View swap produced differing loss: diff={diff}")

    def test_model_forward_backward(self):
        """Test BarlowEncoder forward pass on CIFAR-10 shape (3, 32, 32) and gradient flow."""
        model = BarlowEncoder(dim=128).to(self.device)
        x1 = torch.randn(4, 3, 32, 32, device=self.device)
        x2 = torch.randn(4, 3, 32, 32, device=self.device)

        h1, z1 = model(x1)
        h2, z2 = model(x2)

        self.assertEqual(h1.shape, (4, 512), "Backbone representation shape mismatch")
        self.assertEqual(z1.shape, (4, 128), "Projector embedding shape mismatch")

        loss = barlow_twins_loss(z1, z2, lambd=0.0051)
        self.assertTrue(torch.isfinite(loss), "Loss must be finite")

        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} did not receive gradients")
                self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients")
                self.assertFalse(torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients")

    def test_checkpoint_roundtrip(self):
        """Lightweight test for saving and loading model checkpoint state dict."""
        model = BarlowEncoder(dim=64).to(self.device)
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            torch.save({"model": model.state_dict()}, tmp_path)
            ckpt = torch.load(tmp_path, map_location=self.device)
            model_loaded = BarlowEncoder(dim=64).to(self.device)
            model_loaded.load_state_dict(ckpt["model"])
            for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
                self.assertTrue(torch.equal(p1, p2))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_extract_features_formats(self):
        """extract_features must handle both single tensors and multi-crop tuples."""
        model = BarlowEncoder(dim=64).to(self.device)
        model.eval()

        # 1. Single tensor loader (evaluation mode)
        X1 = torch.randn(8, 3, 32, 32)
        y1 = torch.arange(8)
        ds1 = torch.utils.data.TensorDataset(X1, y1)
        loader1 = torch.utils.data.DataLoader(ds1, batch_size=4)

        feats1, labels1 = extract_features(model, loader1, self.device)
        self.assertEqual(feats1.shape, (8, 512))
        self.assertEqual(labels1.shape, (8,))

        # 2. Multi-crop tuple loader (defensive fallback)
        class TwoCropDummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8
            def __getitem__(self, idx):
                return (torch.randn(3, 32, 32), torch.randn(3, 32, 32)), idx

        loader2 = torch.utils.data.DataLoader(TwoCropDummyDataset(), batch_size=4)
        feats2, labels2 = extract_features(model, loader2, self.device)
        self.assertEqual(feats2.shape, (8, 512))
        self.assertEqual(labels2.shape, (8,))

    def test_linear_probe(self):
        """LinearProbe must map 512-dim features to 10 classes and backprop correctly."""
        probe = LinearProbe(in_features=512, num_classes=10).to(self.device)
        feats = torch.randn(16, 512, device=self.device)
        logits = probe(feats)
        self.assertEqual(logits.shape, (16, 10))
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(probe.fc.weight.grad)

    def test_effective_rank(self):
        """effective_rank should correctly reflect intrinsic dimensional span."""
        # Rank 1 matrix
        np.random.seed(42)
        v = np.random.randn(50, 1)
        w = np.random.randn(1, 10)
        rank1_mat = v @ w
        r1 = effective_rank(rank1_mat, threshold=1e-3)
        self.assertEqual(r1, 1)

        # Full rank matrix
        full_mat = np.random.randn(50, 10)
        r_full = effective_rank(full_mat, threshold=1e-3)
        self.assertEqual(r_full, 10)


if __name__ == "__main__":
    unittest.main()
