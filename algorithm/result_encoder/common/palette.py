from __future__ import annotations

import numpy as np

# 16 well-separated hues for the teeth, grey for gingiva, so neighbouring teeth
# never share a colour in MeshLab / CloudCompare (toothseg3d/infer.py)
PALETTE = np.array([
    (170, 170, 170),
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
], np.uint8)


def instance_colours(instances: np.ndarray) -> np.ndarray:
    """Instance id -> RGB; ids beyond 16 cycle through the tooth hues."""
    return PALETTE[np.where(instances > 0, (instances - 1) % 16 + 1, 0)]
