"""Test setup: segmentation3d on sys.path first, toothseg3d (the reference) last.

Run from anywhere with the `ai` environment:
    python -m pytest segmentation3d/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / "toothseg3d"
DATA = ROOT.parent / "data"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LEGACY) not in sys.path:
    sys.path.append(str(LEGACY))         # last: its `algorithm/` must not shadow ours


def teeth3ds_scans(jaw: str, n: int) -> list[Path]:
    found = sorted((DATA / "open/teeth3ds/raw").glob(f"*/{jaw}/*/*_{jaw}.obj"))[:n]
    if not found:
        pytest.skip("Teeth3DS raw scans not found under data/open/teeth3ds/raw")
    return found


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
