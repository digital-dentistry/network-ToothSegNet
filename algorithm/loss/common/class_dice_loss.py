"""Multi-class soft Dice over per-point class logits (port of `train.dice_loss`).

Run `python algorithm/loss/common/class_dice_loss.py` to check it and save
`class_dice_loss.png` next to this file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SMOOTH = 1.0


# ----------------------------------------------------------------- loss
def class_dice_loss(logits: torch.Tensor, target: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Mean soft dice over the classes present in `target`: logits [N,C], target [N].

    Every present class weighs the same whatever its point count, so a third
    molar in 6 % of the scans is not drowned by the gingiva the way it is under
    cross-entropy.  Absent classes are left out rather than rewarded for
    predicting nothing.
    """
    p = logits.softmax(1)
    t = F.one_hot(target, n_classes).to(p.dtype)
    inter = (p * t).sum(0)
    denom = p.sum(0) + t.sum(0)
    present = t.sum(0) > 0
    d = 1 - (2 * inter + SMOOTH) / (denom + SMOOTH)
    return d[present].mean() if bool(present.any()) else d.sum() * 0


class ClassDiceLoss(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return class_dice_loss(logits, target, self.n_classes)


# ----------------------------------------------------------------- checks
def self_check() -> None:
    torch.manual_seed(0)
    logits, target = torch.randn(1000, 5), torch.randint(0, 4, (1000,))     # class 4 absent
    p = logits.softmax(1)
    ref = torch.stack([1 - (2 * p[target == c, c].sum() + 1) / (p[:, c].sum() + (target == c).sum() + 1)
                       for c in range(4)]).mean()
    print(f"present only  ours = {class_dice_loss(logits, target, 5):.6f}  loop = {ref:.6f}")
    assert torch.allclose(class_dice_loss(logits, target, 5), ref, atol=1e-6)
    perfect = class_dice_loss(F.one_hot(target, 5).float() * 40, target, 5)
    assert perfect < 1e-3
    print("self-check ok")


# ----------------------------------------------------------------- plot
def plot_class_dice_loss(save_path: str | None = None, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    from algorithm.loss.common.plot_style import BLUE, INK, MUTED, ORANGE, SURFACE, legend, save, style

    # gingiva plus 16 teeth; one rare class at a varying share of the points
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    fig.suptitle("Class dice vs cross-entropy: a rare class predicted entirely wrong",
                 x=0.06, ha="left", color=INK, fontsize=13, fontweight="bold")
    shares = torch.logspace(-3.5, -1, 40).tolist()
    n = 50000
    ce, dice = [], []
    for share in shares:
        rare = max(1, int(n * share))
        target = torch.zeros(n, dtype=torch.long)
        target[:rare] = 16                                  # the rare class
        target[rare:rare + 20000] = torch.randint(1, 16, (20000,))
        logits = F.one_hot(target, 17).float() * 4
        logits[:rare] = F.one_hot(torch.full((rare,), 15), 17).float() * 4      # all called its neighbour
        ce.append(F.cross_entropy(logits, target).item())
        dice.append(class_dice_loss(logits, target, 17).item())
    pct = [100 * s for s in shares]

    ax = axes[0]
    style(ax, "Loss as the rare class gets rarer", "rare class (% of points)", "loss")
    ax.plot(pct, ce, color=ORANGE, lw=2, label="cross-entropy")
    ax.plot(pct, dice, color=BLUE, lw=2, label="class dice (present classes)")
    ax.set_xscale("log")
    ax.text(0.04, 0.5, "CE fades as the class shrinks;\ndice charges a full class's worth",
            transform=ax.transAxes, ha="left", va="center", color=MUTED, fontsize=8)
    legend(ax, loc="upper left")

    ax = axes[1]
    style(ax, "Per-class term: the same noise costs the same at any size", "class", "1 − dice")
    ax.text(0.98, 0.04, "class 16 has 100 points, teeth 1–15 have 1 200,\nthe gingiva (0) 30 000",
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8)
    target = torch.cat([torch.zeros(30000, dtype=torch.long), torch.arange(1, 16).repeat_interleave(1200),
                        torch.full((100,), 16)])                # class 16 is rare
    logits = F.one_hot(target, 17).float() * 3 + torch.randn(target.numel(), 17)
    p = logits.softmax(1)
    t = F.one_hot(target, 17).float()
    per = 1 - (2 * (p * t).sum(0) + 1) / (p.sum(0) + t.sum(0) + 1)
    ax.bar(range(17), per, color=BLUE)
    ax.set_xticks(range(0, 17, 2))
    save(fig, save_path, show)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))     # segmentation3d/
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_suffix(".png")))
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    self_check()
    plot_class_dice_loss(a.out, a.show)
