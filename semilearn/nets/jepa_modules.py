# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Shared building blocks for JEPAMatch (https://github.com/aah94/JEPAMatch).

These are backbone-agnostic: any network can opt into JEPAMatch support by
inheriting `JEPAProjectionMixin` and calling `_build_jepa_projection(...)`
once its final feature dimension is known (see semilearn/nets/wrn/wrn.py,
semilearn/nets/wrn/wrn_var.py and semilearn/nets/resnet/resnet.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (LeJEPA, Balestriero et al. 2025).

    Matches the empirical characteristic function of `proj` against that of a
    standard normal N(0, I) along `num_slices` random 1D projections.
    """

    def __init__(self, feature_dim=128, knots=17, num_slices=1024, range_max=5.0):
        super().__init__()
        self.num_slices = num_slices
        self.feature_dim = feature_dim
        t = torch.linspace(0, range_max, knots, dtype=torch.float32)
        dt = range_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # trapezoidal endpoints
        window = torch.exp(-t.square() / 2.0)  # real part of standard Gaussian CF

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(self.feature_dim, self.num_slices, device=proj.device)
        A = F.normalize(A, p=2, dim=0)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        ecf_real = x_t.cos().mean(dim=0)
        ecf_imag = x_t.sin().mean(dim=0)
        err = (ecf_real - self.phi).square() + ecf_imag.square()
        statistic = (err @ self.weights) * proj.size(0)
        return statistic.mean()


class AdaptiveSIGReg(nn.Module):
    """Adaptive Class-wise SIGReg (JEPAMatch).

    Extends SIGReg from a single global isotropic Gaussian to per-class
    isotropic Gaussians: confident samples are centered on their (pseudo-)
    class mean and regularized toward N(0, sigma(alpha)^2 I), with an Active
    Repulsion Loss keeping distinct class means apart. See paper Sec. 4.2.
    """

    def __init__(self, num_classes=10, feature_dim=128, knots=17, num_slices=1024, range_max=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_slices = num_slices

        t = torch.linspace(0, range_max, knots, dtype=torch.float32)
        dt = range_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        self.register_buffer("t", t)
        self.register_buffer("weights_base", weights)

    def _batch_class_means(self, z: torch.Tensor, y: torch.Tensor):
        """
        z: (N, D), y: (N,) long class ids.
        Returns per-class means (C, D) and a (C,) bool mask of classes present
        in this batch (absent classes get a zero mean that must not be used).
        """
        C, D = self.num_classes, z.size(1)
        device, dtype = z.device, z.dtype

        sums = torch.zeros(C, D, device=device, dtype=dtype)
        counts = torch.zeros(C, device=device, dtype=dtype)

        sums.index_add_(0, y, z)
        counts.index_add_(0, y, torch.ones_like(y, dtype=dtype))

        present = counts > 0
        means = sums / counts.clamp_min(1).unsqueeze(1)
        return means, present

    def forward(self, unlab_proj, pseudo_labels, mask, alpha, labeled_proj=None, labeled_targets=None):
        """
        unlab_proj: (B * num_rep, D) local-view projections (num_rep = K local crops)
        pseudo_labels: (B,) pseudo-label per unlabeled sample
        mask: (B,) confidence mask (0/1 or bool)
        alpha: float in [0, 1], warmup->main phase progress (0 = warmup)
        labeled_proj/labeled_targets: optional, included when computing class means
        """
        B = pseudo_labels.size(0)
        N = unlab_proj.size(0)
        D = unlab_proj.size(1)
        assert D == self.feature_dim, "feature_dim mismatch"
        assert N % B == 0, "unlab_proj length must be a multiple of pseudo_labels length"
        num_rep = N // B

        unlab_base = unlab_proj if num_rep == 1 else unlab_proj.view(num_rep, B, D).mean(dim=0)

        conf = mask.bool()
        z_u = unlab_base[conf]
        y_u = pseudo_labels[conf].long()

        alpha = float(alpha)
        if alpha <= 0.0 or z_u.numel() == 0:
            # Warmup phase: no class structure yet, regularize around the global origin.
            centered_proj = unlab_proj
            repulsion_loss = unlab_proj.new_tensor(0.0)
        else:
            if labeled_proj is not None and labeled_targets is not None:
                z_all = torch.cat([labeled_proj, z_u], dim=0)
                y_all = torch.cat([labeled_targets.long(), y_u], dim=0)
            else:
                z_all, y_all = z_u, y_u

            means, present = self._batch_class_means(z_all, y_all)

            centers = means[pseudo_labels.long()].detach()  # stop-gradient centering target
            dynamic_centers = conf.unsqueeze(-1).float() * (alpha * centers)
            dynamic_centers = dynamic_centers.repeat(num_rep, 1)
            centered_proj = unlab_proj - dynamic_centers

            # Active Repulsion Loss: only classes present in the current batch.
            means_present = means[present]
            if means_present.size(0) <= 1:
                repulsion_loss = unlab_proj.new_tensor(0.0)
            else:
                m = F.normalize(means_present, dim=1, eps=1e-8)
                sim = m @ m.t()
                off = sim[~torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)]
                repulsion_loss = F.relu(off).pow(2).mean()

        current_sigma = 1.0 - (0.9 * alpha)  # Eq. (variance annealing): 1.0 -> 0.1

        A = torch.randn(self.feature_dim, self.num_slices, device=unlab_proj.device, dtype=unlab_proj.dtype)
        A = F.normalize(A, p=2, dim=0)
        x_t = (centered_proj @ A).unsqueeze(-1) * self.t

        ecf_real = x_t.cos().mean(dim=0)
        ecf_imag = x_t.sin().mean(dim=0)

        target_window = torch.exp(-(self.t ** 2) * (current_sigma ** 2) / 2.0)
        weights = self.weights_base * target_window

        err = (ecf_real - target_window).square() + ecf_imag.square()
        sigreg_loss = ((err @ weights) * unlab_proj.size(0)).mean()

        return sigreg_loss, repulsion_loss


class JEPAProjectionHead(nn.Module):
    """3-layer MLP projection head feeding into (Adaptive)SIGReg / the JEPA prediction loss."""

    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class JEPAProjectionMixin:
    """
    Mixin adding an optional JEPAMatch projection head + (Adaptive)SIGReg to a
    backbone. A backbone opts in by calling `self._build_jepa_projection(...)`
    once its pooled feature dimension is known, then adding
    `proj = self.projector(feat)` (guarded by `self.projector is not None`) to
    its own `forward()`. Backbones that never call the builder pay no cost:
    `self.projector` stays unset and plain training is unaffected.
    """

    def _build_jepa_projection(self, in_dim, num_classes, proj_hidden_dim=512, proj_out_dim=128,
                                knots=17, num_slices=1024, range_max=5.0):
        self.projector = JEPAProjectionHead(in_dim=in_dim, hidden_dim=proj_hidden_dim, out_dim=proj_out_dim)
        self.sigreg = SIGReg(feature_dim=proj_out_dim, knots=knots, num_slices=num_slices, range_max=range_max)
        self.adaptive_sigreg = AdaptiveSIGReg(num_classes=num_classes, feature_dim=proj_out_dim,
                                               knots=knots, num_slices=num_slices, range_max=range_max)

    def loss_sigreg(self, proj):
        return self.sigreg(proj)

    def loss_adaptive_sigreg(self, proj_ulb, y_ulb, mask, alpha, proj_lb, y_lb):
        return self.adaptive_sigreg(proj_ulb, y_ulb, mask, alpha, proj_lb, y_lb)
