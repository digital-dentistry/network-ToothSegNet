"""Shared figure style for the loss visualisations (same palette as
Pointcept/pointcept/loss/kl_divergence.py)."""
from __future__ import annotations

from typing import Any

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def legend(ax: Any, **kw: Any) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, **kw)


def save(fig: Any, save_path: str | None, show: bool) -> None:
    import matplotlib.pyplot as plt

    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5, w_pad=2.5)
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=SURFACE)
        print(f"saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)
