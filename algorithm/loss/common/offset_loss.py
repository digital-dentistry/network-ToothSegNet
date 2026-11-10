"""L1 loss on the predicted offset from a tooth point to its tooth's centroid
(the auxiliary offset head of `train_inst.py`).

Run `python algorithm/loss/common/offset_loss.py` to check it and save
`offset_loss.png` next to this file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


# ----------------------------------------------------------------- loss
def offset_loss(offset: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the `mask` [N] points of |offset - target|_1 summed over xyz.

    L1, not L2: a point on a neighbouring tooth's margin with a wrong label has
    an offset centimetres off, and L2 would let those few points steer the head.
    """
    if not bool(mask.any()):
        return offset.sum() * 0                  # keeps the graph, contributes nothing
    return (offset[mask] - target[mask]).abs().sum(1).mean()


class OffsetLoss(nn.Module):
    def forward(self, offset: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return offset_loss(offset, target, mask)


# ----------------------------------------------------------------- checks
def self_check() -> None:
    torch.manual_seed(0)
    off, tgt = torch.randn(100, 3), torch.randn(100, 3)
    mask = torch.rand(100) > 0.4
    ref = sum((off[i] - tgt[i]).abs().sum() for i in range(100) if mask[i]) / int(mask.sum())
    print(f"masked mean   ours = {offset_loss(off, tgt, mask):.6f}  loop = {ref:.6f}")
    assert torch.allclose(offset_loss(off, tgt, mask), ref, atol=1e-6)
    empty = offset_loss(off.requires_grad_(True), tgt, torch.zeros(100, dtype=torch.bool))
    empty.backward()
    assert empty == 0 and off.grad.abs().sum() == 0
    print("self-check ok")


# ----------------------------------------------------------------- plot
def plot_offset_loss(save_path: str | None = None, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    from algorithm.loss.common.plot_style import BLUE, INK, MUTED, ORANGE, SURFACE, legend, save, style

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    fig.suptitle("Offset loss: L1 keeps mislabelled points from steering the head",
                 x=0.06, ha="left", color=INK, fontsize=13, fontweight="bold")

    ax = axes[0]
    style(ax, "Gradient vs error", "offset error e (mm, one axis)", "∂loss/∂e")
    e = torch.linspace(-6, 6, 241)
    ax.plot(e, torch.sign(e), color=BLUE, lw=2, label="L1  |e|  → ±1, bounded")
    ax.plot(e, e, color=ORANGE, lw=2, label="L2  ½e²  → e, grows with the error")
    ax.set_ylim(-6, 6)
    legend(ax, loc="upper left")

    ax = axes[1]
    style(ax, "Share of the gradient taken by outliers", "outlier error (mm)", "share of total |gradient|")
    torch.manual_seed(0)
    inliers = torch.randn(2000, 3) * 0.3                     # a well-trained head: ~0.3 mm off
    errs = torch.linspace(0.5, 30, 60).tolist()
    l1, l2 = [], []
    for big in errs:
        outliers = torch.randn(40, 3)
        outliers = outliers / outliers.norm(dim=1, keepdim=True) * big   # 2 % of points, `big` mm off
        err = torch.cat([inliers, outliers])
        g1, g2 = torch.sign(err).abs().sum(1), err.abs().sum(1)
        l1.append(float(g1[2000:].sum() / g1.sum()))
        l2.append(float(g2[2000:].sum() / g2.sum()))
    ax.plot(errs, l1, color=BLUE, lw=2, label="L1")
    ax.plot(errs, l2, color=ORANGE, lw=2, label="L2")
    ax.text(0.98, 0.04, "2 % of the points are outliers", transform=ax.transAxes,
            ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.set_ylim(0, 1)
    legend(ax, loc="center right")
    save(fig, save_path, show)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))     # segmentation3d/
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_suffix(".png")))
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    self_check()
    plot_offset_loss(a.out, a.show)
