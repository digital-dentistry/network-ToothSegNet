"""Vertex normals: area-weighted from faces, or local PCA for a bare point cloud."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

from common.spatial import knn_search


def mesh_normals(positions: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals (port of `t3ds_io.vertex_normals`).

    The cross product's length is twice the face area, so summing raw cross
    products is the area weighting.  index_add_ per corner in face order is the
    original's np.add.at, sum for sum on the CPU.
    """
    tri = positions[faces]                                         # [F,3,3]
    fn = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = torch.zeros_like(positions)
    for corner in range(3):
        vn.index_add_(0, faces[:, corner], fn)
    return vn / torch.linalg.vector_norm(vn, dim=1, keepdim=True).clamp_min(1e-12)


def pca_normals(positions: torch.Tensor, k: int = 24) -> torch.Tensor:
    """Local-PCA normals with signs made consistent along an MST of the kNN graph.

    Port of `infer.pca_normals`.  PCA alone is sign-ambiguous and the network
    was trained on outward normals, so the sign is propagated from a seed (the
    point furthest along +z, made to face away from the centroid) to its
    neighbours, preferring edges whose normals already agree.

    The neighbour search and the batched 3x3 eigen-decomposition are torch;
    the spanning tree is scipy (torch has no graph algorithms).  The sign walk
    that the original ran as a Python loop over every vertex is a few rounds of
    pointer jumping here.
    """
    n_pts = positions.shape[0]
    dev = positions.device
    _, idx = knn_search(positions, positions, k, cell=_neighbourhood_cell(positions, k))
    idx = torch.where(idx < 0, torch.arange(n_pts, device=dev)[:, None], idx)
    nb = positions[idx].double()
    nb = nb - nb.mean(1, keepdim=True)
    cov = torch.einsum("nki,nkj->nij", nb, nb) / k
    normals = torch.linalg.eigh(cov).eigenvectors[:, :, 0]         # smallest eigenvalue

    rows = torch.arange(n_pts, device=dev).repeat_interleave(k - 1)
    cols = idx[:, 1:].reshape(-1)
    weight = 1.0 - (normals[rows] * normals[cols]).sum(1).abs() + 1e-6
    g = coo_matrix((weight.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())),
                   shape=(n_pts, n_pts))
    mst = minimum_spanning_tree(g + g.T)
    seed = int(positions[:, 2].argmax())
    root_sign = -1.0 if float(normals[seed] @ (positions[seed] - positions.mean(0)).double()) < 0 else 1.0
    order, pred = breadth_first_order(mst, seed, directed=False)
    parent = torch.as_tensor(np.where(pred < 0, -1, pred).astype(np.int64), device=dev)

    sign = _path_signs(normals, parent, seed, root_sign)
    return (normals * sign[:, None]).float()


def _path_signs(normals: torch.Tensor, parent: torch.Tensor, root: int, root_sign: float) -> torch.Tensor:
    """Sign of every vertex after walking the tree from the root.

    A child flips when it disagrees with its (already flipped) parent, so its
    final sign is the product of "disagrees with parent" along the path to the
    root.  Pointer jumping evaluates every path product in log2(depth) rounds.
    Vertices the tree does not reach keep their sign, as in the original.
    """
    n_pts = normals.shape[0]
    dev = normals.device
    reached = parent >= 0
    agree = torch.ones(n_pts, dtype=torch.float64, device=dev)
    p = parent.clamp_min(0)
    flips = (normals[reached] * normals[p[reached]]).sum(1) < 0
    agree[reached] = 1.0 - 2.0 * flips.to(agree.dtype)
    agree[root] = root_sign

    # node n_pts is a terminal: value 1, points at itself
    val = torch.cat([agree, agree.new_ones(1)])
    ptr = torch.full((n_pts + 1,), n_pts, dtype=torch.int64, device=dev)
    ptr[:n_pts][reached] = parent[reached]
    for _ in range(max(1, math.ceil(math.log2(n_pts + 1))) + 1):
        if bool((ptr[:n_pts] == n_pts).all()):
            break
        val = val * val[ptr]
        ptr = ptr[ptr]
    return val[:n_pts]


def _neighbourhood_cell(positions: torch.Tensor, k: int) -> float:
    """A grid cell that holds about k neighbours, from a sample's nearest-neighbour spacing."""
    n_pts = positions.shape[0]
    extent = (positions.amax(0) - positions.amin(0)).clamp_min(1e-3)
    guess = float(extent.prod() / max(n_pts, 1)) ** (1.0 / 3.0)
    sample = positions[torch.randperm(n_pts, device=positions.device)[:2048]]
    d, _ = knn_search(sample, positions, 2, cell=guess)
    spacing = float(d[:, 1][torch.isfinite(d[:, 1])].median()) if n_pts > 1 else guess
    # on a surface, k neighbours sit within ~sqrt(k / pi) spacings
    return max(spacing * math.sqrt(k / math.pi), 1e-4)
