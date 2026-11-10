from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from tests.conftest import DEVICES, teeth3ds_scans


@pytest.fixture(scope="module")
def scan() -> tuple[np.ndarray, np.ndarray]:
    from t3ds_io import read_obj, voxel_downsample
    V, _, _ = read_obj(str(teeth3ds_scans("lower", 1)[0]))
    rep, _ = voxel_downsample(V, 0.25)
    c = V[rep].mean(0)
    return (V - c).astype(np.float32), (V[rep] - c).astype(np.float32)


@pytest.mark.parametrize("device", DEVICES)
def test_knn_matches_kdtree(scan: tuple[np.ndarray, np.ndarray], device: str) -> None:
    from common.spatial import knn_search
    query, ref = scan
    want_d, want_i = cKDTree(ref).query(query, k=5)
    d, i = knn_search(torch.from_numpy(query).to(device), torch.from_numpy(ref).to(device), 5, 0.5)
    d, i = d.cpu().numpy(), i.cpu().numpy()
    assert np.abs(d - want_d).max() < 1e-5
    assert (np.sort(i, 1) == np.sort(want_i, 1)).all()


@pytest.mark.parametrize("device", DEVICES)
def test_knn_bad_cell_guess_is_still_exact(scan: tuple[np.ndarray, np.ndarray], device: str) -> None:
    """Far too small a cell: everything is retried or brute-forced, never wrong."""
    from common.spatial import knn_search
    query, ref = scan[0][::50], scan[1]
    want_d, _ = cKDTree(ref).query(query, k=8)
    d, _ = knn_search(torch.from_numpy(query).to(device), torch.from_numpy(ref).to(device), 8, 0.01)
    assert np.abs(d.cpu().numpy() - want_d).max() < 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_radius_query(scan: tuple[np.ndarray, np.ndarray], device: str) -> None:
    from common.spatial import knn_search
    full = scan[0]
    a, b = full[:20000], full[20000:60000]
    want, _ = cKDTree(a).query(b, k=1)
    d, i = knn_search(torch.from_numpy(b).to(device), torch.from_numpy(a).to(device), 1, 0.2, max_radius=0.2)
    d, i = d[:, 0].cpu().numpy(), i[:, 0].cpu().numpy()
    assert ((d < 0.2) == (want < 0.2)).all()
    assert np.allclose(d[want < 0.2], want[want < 0.2], atol=1e-5)
    assert (i[want >= 0.2] == -1).all()


def test_fewer_references_than_k() -> None:
    from common.spatial import knn_search
    q, r = torch.rand(10, 3), torch.rand(3, 3)
    d, i = knn_search(q, r, 5, 0.1)
    want, _ = cKDTree(r.numpy()).query(q.numpy(), k=3)
    assert np.allclose(d[:, :3].numpy(), want, atol=1e-6)
    assert torch.isinf(d[:, 3:]).all() and (i[:, 3:] == -1).all()
    d, i = knn_search(q, torch.empty(0, 3), 2, 0.1)
    assert torch.isinf(d).all() and (i == -1).all()
