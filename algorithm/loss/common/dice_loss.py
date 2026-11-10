"""Soft Dice loss for predicted instance masks (port of `inst_head._dice`).

Run `python algorithm/loss/common/dice_loss.py` to check it and save
`dice_loss.png` next to this file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SMOOTH = 1.0          # keeps an empty mask against an empty target at loss 0, not 0/0


# ----------------------------------------------------------------- loss
def dice_cost(mask_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft dice of every [K,n] mask logit row against every [G,n] target row -> [K,G].

    The pairwise form is the Hungarian matching cost: one matmul scores every
    query against every tooth.
    """
    p = mask_logit.sigmoid()
    num = 2 * (p @ target.t())
    den = p.sum(-1)[:, None] + target.sum(-1)[None]
    return 1 - (num + SMOOTH) / (den + SMOOTH)


def dice_loss(mask_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean soft dice over matched rows: mask_logit [K,n] against target [K,n].

    Dice measures overlap relative to the mask's own size, so a small tooth
    counts as much as a large one -- BCE alone is dominated by the gingiva.
    """
    if mask_logit.shape[0] == 0:
        return mask_logit.sum() * 0
    return dice_cost(mask_logit, target).diagonal().mean()


class DiceLoss(nn.Module):
    def forward(self, mask_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(mask_logit, target)


# ----------------------------------------------------------------- checks
def self_check() -> None:
    torch.manual_seed(0)
    logit, target = torch.randn(6, 500), (torch.rand(6, 500) > 0.7).float()
    p = logit.sigmoid()
    naive = torch.stack([1 - (2 * (p[i] * target[j]).sum() + 1) / (p[i].sum() + target[j].sum() + 1)
                         for i in range(6) for j in range(6)]).view(6, 6)
    print(f"pairwise      max|cost - naive| = {(dice_cost(logit, target) - naive).abs().max():.2e}")
    assert torch.allclose(dice_cost(logit, target), naive, atol=1e-6)
    perfect = dice_loss(torch.where(target > 0, 30.0, -30.0), target)
    print(f"perfect mask  loss = {perfect:.2e}")
    assert perfect < 1e-6
    print("self-check ok")


# ----------------------------------------------------------------- plot
def _boundary_error_case(n: int, frac: float, wrong: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A tooth of `frac * n` points, predicted right except `wrong` boundary points missed."""
    size = max(int(n * frac), wrong + 1)
    target = torch.zeros(1, n)
    target[0, :size] = 1
    logit = torch.where(target > 0, 3.0, -3.0)
    logit[0, :wrong] = -3.0
    return logit, target


def plot_dice_loss(save_path: str | None = None, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    from algorithm.loss.common.plot_style import BLUE, INK, MUTED, ORANGE, SURFACE, legend, save, style

    n, wrong = 40000, 150
    fracs = torch.logspace(-2.3, -0.3, 60).tolist()           # 0.5 % .. 50 % of the scan
    dice, bce, g_dice, g_bce = [], [], [], []
    for f in fracs:
        logit, target = _boundary_error_case(n, f, wrong)
        logit.requires_grad_(True)
        d = dice_loss(logit, target)
        b = F.binary_cross_entropy_with_logits(logit, target)
        gd, = torch.autograd.grad(d, logit)
        gb, = torch.autograd.grad(b, logit)
        dice.append(d.item()); bce.append(b.item())
        g_dice.append(gd[0, :wrong].abs().mean().item()); g_bce.append(gb[0, :wrong].abs().mean().item())
    pct = [100 * f for f in fracs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    fig.suptitle(f"Dice vs BCE: the same {wrong} missed boundary points on teeth of every size",
                 x=0.06, ha="left", color=INK, fontsize=13, fontweight="bold")

    ax = axes[0]
    style(ax, "Loss", "tooth size (% of the scan's points)", "loss")
    ax.plot(pct, dice, color=BLUE, lw=2, label="dice")
    ax.plot(pct, bce, color=ORANGE, lw=2, label="BCE (mean over points)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.98, 0.95, "BCE barely moves: the gingiva\noutweighs any one tooth",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=8)
    legend(ax, loc="lower left")

    ax = axes[1]
    style(ax, "Gradient on a missed point", "tooth size (% of the scan's points)", "|∂loss/∂logit|")
    ax.plot(pct, g_dice, color=BLUE, lw=2, label="dice  ∝ 1 / tooth size")
    ax.plot(pct, g_bce, color=ORANGE, lw=2, label="BCE  ∝ 1 / scan size")
    ax.set_xscale("log")
    ax.set_yscale("log")
    legend(ax, loc="lower left")
    save(fig, save_path, show)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))     # segmentation3d/
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_suffix(".png")))
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    self_check()
    plot_dice_loss(a.out, a.show)
