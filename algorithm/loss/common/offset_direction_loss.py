"""Cosine loss on the direction of the predicted centroid offset (port of the
direction half of `train.offset_loss`).

Run `python algorithm/loss/common/offset_direction_loss.py` to check it and
save `offset_direction_loss.png` next to this file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

MIN_NORM = 1e-4       # mm: below this an offset has no direction to speak of


# ----------------------------------------------------------------- loss
def offset_direction_loss(offset: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the `mask` [N] points of 1 - cos(offset, target).

    L1 on the offset is dominated by the points far from their centroid; near
    the centre the vector is short and L1 barely cares where it points.  The
    cosine weighs every point's direction the same, which is what separates
    two touching teeth at their contact.
    """
    if not bool(mask.any()):
        return offset.sum() * 0
    p, g = offset[mask], target[mask]
    pn = p / p.norm(dim=1, keepdim=True).clamp_min(MIN_NORM)
    gn = g / g.norm(dim=1, keepdim=True).clamp_min(MIN_NORM)
    return (1 - (pn * gn).sum(1)).mean()


class OffsetDirectionLoss(nn.Module):
    def forward(self, offset: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return offset_direction_loss(offset, target, mask)


# ----------------------------------------------------------------- checks
def self_check() -> None:
    torch.manual_seed(0)
    g = torch.randn(200, 3)
    mask = torch.ones(200, dtype=torch.bool)
    for scale, want in ((1.0, 0.0), (-1.0, 2.0)):
        got = offset_direction_loss(g * 3 * scale, g, mask)
        print(f"offset = {scale:+.0f} * 3 * target  loss = {got:.6f}  (want {want})")
        assert abs(float(got) - want) < 1e-5
    ortho = torch.cross(g, torch.randn(200, 3), dim=1)
    assert abs(float(offset_direction_loss(ortho, g, mask)) - 1.0) < 1e-5
    print("self-check ok")


# ----------------------------------------------------------------- plot
def plot_offset_direction_loss(save_path: str | None = None, show: bool = False) -> None:
    import math

    import matplotlib.pyplot as plt

    from algorithm.loss.common.plot_style import BLUE, INK, MUTED, ORANGE, SURFACE, legend, save, style
    from algorithm.loss.common.offset_loss import offset_loss

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    fig.suptitle("Offset direction loss: every point's direction counts the same",
                 x=0.06, ha="left", color=INK, fontsize=13, fontweight="bold")

    ax = axes[0]
    style(ax, "Loss vs angular error", "angle between prediction and target (deg)", "loss")
    deg = torch.linspace(0, 180, 181)
    ax.plot(deg, 1 - torch.cos(torch.deg2rad(deg)), color=BLUE, lw=2, label="1 − cos θ")
    ax.set_ylim(bottom=0)
    legend(ax, loc="upper left")

    ax = axes[1]
    style(ax, "A 20° error at different distances from the centroid", "distance to centroid (mm)", "loss")
    dist = torch.linspace(0.05, 6, 120)
    g = torch.stack([dist, torch.zeros_like(dist), torch.zeros_like(dist)], 1)
    a = math.radians(20)
    p = torch.stack([dist * math.cos(a), dist * math.sin(a), torch.zeros_like(dist)], 1)
    one = torch.ones(1, dtype=torch.bool)
    l1 = [float(offset_loss(p[i:i + 1], g[i:i + 1], one)) for i in range(len(dist))]
    cos = [float(offset_direction_loss(p[i:i + 1], g[i:i + 1], one)) for i in range(len(dist))]
    ax.plot(dist, l1, color=ORANGE, lw=2, label="L1 offset loss")
    ax.plot(dist, cos, color=BLUE, lw=2, label="direction loss")
    ax.text(0.04, 0.96, "near the centre L1 hardly notices\nthe vector points the wrong way",
            transform=ax.transAxes, ha="left", va="top", color=MUTED, fontsize=8)
    legend(ax, loc="center right")
    save(fig, save_path, show)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))     # segmentation3d/
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_suffix(".png")))
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    self_check()
    plot_offset_direction_loss(a.out, a.show)
