"""Query-based instance head -- the Mask3D / SPFormer recipe (port of `inst_head.py`).

A fixed set of queries cross-attends to the scan's features; each ends up
owning at most one tooth and predicts that tooth's mask directly, plus a score
saying whether it found a tooth at all.  The Hungarian-matched training
criterion is `algorithm/loss`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Additive mask value for "do not attend here".  -inf is right in PyTorch; a
# TensorRT fp16 engine turns it into NaN inside softmax, which is why the ONNX
# export swaps it for -1e4 (toothseg3d/export_onnx.py).
NEG_INF = float("-inf")


class DecoderLayer(nn.Module):
    """masked cross-attention -> self-attention -> FFN, all pre-norm."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.h = heads
        self.n_x, self.n_s, self.n_f = (nn.LayerNorm(dim) for _ in range(3))
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.px = nn.Linear(dim, dim)
        self.sa = nn.MultiheadAttention(dim, heads, batch_first=True)
        hid = int(dim * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """q [B,K,D]; kv [B,M,D]; attn_bias [B,h,K,M] additive (-inf blocks)."""
        b, k, d = q.shape
        x = self.n_x(q)
        qh = self.q(x).view(b, k, self.h, d // self.h).transpose(1, 2)
        key, val = self.kv(kv).chunk(2, -1)
        kh = key.view(b, -1, self.h, d // self.h).transpose(1, 2)
        vh = val.view(b, -1, self.h, d // self.h).transpose(1, 2)
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_bias)
        q = q + self.px(o.transpose(1, 2).reshape(b, k, d))

        s = self.n_s(q)
        q = q + self.sa(s, s, s, need_weights=False)[0]
        return q + self.ff(self.n_f(q))


class InstanceHead(nn.Module):
    def __init__(self, point_dim: int, kv_dim: int, dim: int = 128, n_query: int = 32,
                 layers: int = 4, heads: int = 8) -> None:
        super().__init__()
        self.n_query = n_query
        self.query = nn.Embedding(n_query, dim)
        self.kv_in = nn.Sequential(nn.Linear(kv_dim, dim), nn.LayerNorm(dim))
        self.pt_in = nn.Sequential(nn.Linear(point_dim, dim), nn.LayerNorm(dim))
        self.kv_pos = nn.Linear(3, dim)
        self.layers = nn.ModuleList([DecoderLayer(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Linear(dim, 2)              # [is a tooth, nothing here]
        self.mask = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                  nn.Linear(dim, dim), nn.LayerNorm(dim))
        # A mask logit is a dot product over `dim` layer-normed channels; without
        # this its std is ~sqrt(dim) and the sigmoid saturates.
        self.scale = dim ** -0.5
        self.bias = nn.Parameter(torch.zeros(1))
        self.heads = heads

    def _masks(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Mask logits of every query over the cross-attention points."""
        return torch.einsum("bkd,bmd->bkm", self.mask(self.norm(q)), kv) * self.scale + self.bias

    def forward(self, pt_feat: torch.Tensor, counts: list[int], kv_feat: torch.Tensor,
                kv_coord: torch.Tensor, pad_index: torch.Tensor, pad_valid: torch.Tensor,
                seed: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """pt_feat [N,*] and kv_* are packed; pad_index/pad_valid [B,M] make the
        coarse level rectangular, `seed` [B,K] picks the query starting points.

        Returns cls [B,K,2] and the full-resolution mask logits, packed [N,K];
        in training also `aux`, every decoder layer's coarse mask logits
        [B,K,M] for deep supervision.
        """
        kv = self.kv_in(kv_feat)[pad_index]                     # [B,M,D]
        kv = kv + self.kv_pos(kv_coord[pad_index] * 0.03)
        kv = kv * pad_valid.unsqueeze(-1)

        q = self.query.weight[None].expand(pad_index.shape[0], -1, -1)
        if seed is not None:
            # Start each query at a real, spread-out point: learned cold, the
            # decoder spends its budget finding the teeth instead of shaping them.
            q = q + torch.gather(kv, 1, seed.unsqueeze(-1).expand(-1, -1, kv.shape[-1]))
        q = q.contiguous()
        blocked = ~pad_valid[:, None, :]                        # [B,1,M]
        aux = []
        for lyr in self.layers:
            m = self._masks(q, kv)                              # [B,K,M]
            if self.training:
                aux.append(m)
            # Mask2Former: attend only where the query already claims mass...
            blk = (m.detach().sigmoid() < 0.5) | blocked
            blk = torch.where(blk.all(-1, keepdim=True), blocked, blk)
            bias = torch.zeros_like(blk, dtype=q.dtype).masked_fill(blk, NEG_INF)
            q = lyr(q, kv, bias.unsqueeze(1).expand(-1, self.heads, -1, -1))

        qe = self.mask(self.norm(q))                            # [B,K,D]
        pe = self.pt_in(pt_feat)                                # [N,D]
        out = [p @ qe[b].t() * self.scale + self.bias
               for b, p in enumerate(torch.split(pe, counts))]  # per scan: [n,K]
        res = {"cls": self.cls(self.norm(q)), "mask": torch.cat(out, 0)}
        if self.training:
            res["aux"] = aux
        return res
