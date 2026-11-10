"""Binary cross-entropy for predicted instance masks (port of `inst_head._bce`).

Run `python algorithm/loss/common/mask_bce_loss.py` to check it and save
`mask_bce_loss.png` next to this file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- loss
def bce_cost(mask_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean BCE of every [K,n] mask logit row against every [G,n] target row -> [K,G].

    Per point, BCE is -log p where the target is 1 and -log(1 - p) where it is
    0.  Both terms are computed once per point, so all K x G pairs take two
    matmuls instead of K * G full passes.
    """
    n = mask_logit.shape[1]
    pos = F.binary_cross_entropy_with_logits(mask_logit, torch.ones_like(mask_logit), reduction="none")
    neg = F.binary_cross_entropy_with_logits(mask_logit, torch.zeros_like(mask_logit), reduction="none")
    return (pos @ target.t() + neg @ (1 - target).t()) / n


def mask_bce_loss(mask_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean BCE over matched rows: mask_logit [K,n] against target [K,n]."""
    if mask_logit.shape[0] == 0:
        return mask_logit.sum() * 0
    return F.binary_cross_entropy_with_logits(mask_logit, target)


class MaskBCELoss(nn.Module):
    def forward(self, mask_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mask_bce_loss(mask_logit, target)


# ----------------------------------------------------------------- checks
def _naive_cost(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.stack([F.binary_cross_entropy_with_logits(logit[i], target[j])
                        for i in range(logit.shape[0]) for j in range(target.shape[0])]
                       ).view(logit.shape[0], target.shape[0])


def self_check() -> None:
    torch.manual_seed(0)
    logit, target = torch.randn(6, 500) * 3, (torch.rand(5, 500) > 0.7).float()
    err = (bce_cost(logit, target) - _naive_cost(logit, target)).abs().max()
    print(f"pairwise      max|cost - naive| = {err:.2e}")
    assert err < 1e-5
    diag = bce_cost(logit[:5], target).diagonal().mean()
    print(f"matched       loss = {mask_bce_loss(logit[:5], target):.6f}  pairwise diagonal = {diag:.6f}")
    assert torch.allclose(mask_bce_loss(logit[:5], target), diag, atol=1e-6)
    print("self-check ok")


# ----------------------------------------------------------------- plot
def plot_mask_bce_loss(save_path: str | None = None, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    from algorithm.loss.common.plot_style import (AXIS, BLUE, INK, INK2, MUTED, ORANGE, SURFACE,
                                                  legend, save, style)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    fig.suptitle("Mask BCE: per-point loss and the pairwise matching cost",
                 x=0.06, ha="left", color=INK, fontsize=13, fontweight="bold")

    ax = axes[0]
    style(ax, "Per point", "mask logit", "BCE")
    z = torch.linspace(-8, 8, 401)
    ax.plot(z, F.binary_cross_entropy_with_logits(z, torch.ones_like(z), reduction="none"),
            color=BLUE, lw=2, label="target 1 (tooth):  −log σ(z)")
    ax.plot(z, F.binary_cross_entropy_with_logits(z, torch.zeros_like(z), reduction="none"),
            color=ORANGE, lw=2, label="target 0 (elsewhere):  −log(1 − σ(z))")
    ax.text(0.5, 0.96, "linear once wrong, flat once right", transform=ax.transAxes,
            ha="center", va="top", color=MUTED, fontsize=8)
    ax.set_ylim(bottom=0)
    legend(ax, loc="center left")

    ax = axes[1]
    style(ax, "Pairwise cost: two matmuls vs K × G passes", "naive per-pair BCE", "bce_cost")
    torch.manual_seed(0)
    logit = torch.randn(32, 3000) * 2
    target = (torch.rand(16, 3000) < torch.rand(16, 1) * 0.3).float()
    fast, slow = bce_cost(logit, target).flatten(), _naive_cost(logit, target).flatten()
    lo, hi = float(slow.min()), float(slow.max())
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1)
    ax.plot(slow, fast, "o", ms=3, mfc=BLUE, mec="none", alpha=0.6)
    ax.text(0.04, 0.96, f"32 queries × 16 teeth\nmax |difference| = {(fast - slow).abs().max():.1e}",
            transform=ax.transAxes, ha="left", va="top", color=INK2, fontsize=8)
    save(fig, save_path, show)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))     # segmentation3d/
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_suffix(".png")))
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    self_check()
    plot_mask_bce_loss(a.out, a.show)
