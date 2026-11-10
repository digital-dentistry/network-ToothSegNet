from __future__ import annotations

import numpy as np
import torch


def random_so3(u: np.ndarray) -> np.ndarray:
    """A rotation from three uniforms, uniform over SO(3) (Shoemake's quaternions).

    Same arithmetic as toothseg3d's `Teeth3DS.random_so3`, so a seed gives
    the same matrix to the last bit.
    """
    u1, u2, u3 = u
    a, b = np.sqrt(1 - u1), np.sqrt(u1)
    x, y, z, w = (a * np.sin(2 * np.pi * u2), a * np.cos(2 * np.pi * u2),
                  b * np.sin(2 * np.pi * u3), b * np.cos(2 * np.pi * u3))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tta_rotations(n: int, seed: int = 0) -> torch.Tensor:
    """[n,3,3] float64: the identity, then the original's TTA poses in order.

    `infer.segment_inst` draws pose i by seeding numpy's global generator from
    RandomState(0) and calling random_so3; a private RandomState with the same
    seed draws the same three uniforms without touching global state.
    """
    rs = np.random.RandomState(seed)
    mats = [np.eye(3)]
    for _ in range(n - 1):
        mats.append(random_so3(np.random.RandomState(rs.randint(1 << 30)).rand(3)))
    return torch.from_numpy(np.stack(mats))
