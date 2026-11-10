"""Distance from every query point to its nearest reference point.

The crops and the gingiva trim ask this of a full-resolution scan (~100 k
vertices against ~50 k), and most queries sit millimetres away from the
reference.  `common.spatial.knn_search` is built for the opposite case -- every
query next to its neighbours -- and its retries with growing cells take 2.3 s
on the CPU and 9 GB on the GPU here.  So:

* host: scipy's cKDTree, as toothseg3d did (18 ms);
* device: chunked brute force in matmul form (60 ms on an RTX 5080).  The
  matmul form loses ~1e-4 mm^2 to cancellation in d^2, which is 0.01 mm at
  d = 0 but 1e-5 mm at the 1 mm and up that the callers threshold at, so the
  decisions match cKDTree's.
"""
from __future__ import annotations

import torch

_CHUNK = 1024          # queries per [chunk, R] distance block: ~230 MB at R = 54 k


def nearest_distance(query: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """query [Q,3], ref [R,3] on the same device -> [Q] float32 distances."""
    if ref.shape[0] == 0:
        return torch.full((query.shape[0],), torch.inf, device=query.device)
    if query.device.type == "cpu":
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(ref.numpy()).query(query.numpy(), workers=-1)
        return torch.from_numpy(dist).float()
    # centring shrinks |x|^2, and with it the cancellation error
    centre = ref.mean(0)
    q, r = query - centre, ref - centre
    out = torch.empty(query.shape[0], device=query.device)
    for lo in range(0, query.shape[0], _CHUNK):
        out[lo:lo + _CHUNK] = torch.cdist(q[lo:lo + _CHUNK], r).amin(1)
    return out
