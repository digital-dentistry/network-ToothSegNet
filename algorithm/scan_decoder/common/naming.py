from __future__ import annotations

from pathlib import Path


def guess_jaw(path: Path, explicit: str | None = None) -> str:
    """'upper' / 'lower' from the file name (upper/lower/maxillary/mandibular).

    The jaw is one input channel; the model barely uses it (flipping it leaves
    the result almost unchanged, README §7.5), so a wrong guess is cheap.
    """
    if explicit in ("upper", "lower"):
        return explicit
    s = path.name.lower()
    if "upper" in s or "max" in s:
        return "upper"
    if "lower" in s or "man" in s:
        return "lower"
    return "lower"
