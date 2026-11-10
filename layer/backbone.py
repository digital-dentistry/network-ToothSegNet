"""Serialized Point Transformer U-Net (port of `toothseg3d/model.py`).

PTv3's recipe -- order the points along a space-filling curve, attend inside
fixed-size patches of that order, U-Net with grid pooling -- in stock PyTorch
ops only (`scaled_dot_product_attention`, `scatter_add_`, gathers).

Every module keeps its original attribute name, so a toothseg3d checkpoint's
state_dict loads with strict=True.  The forward reads a `Geometry` instead of
the original's dict; the indices inside are the same.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.model_input.common.geometry import Geometry, Layout


class PatchAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:            # [n_patch, patch, C]
        b, length, c = x.shape
        qkv = self.qkv(x).reshape(b, length, 3, self.h, c // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(o.transpose(1, 2).reshape(b, length, c))


class Block(nn.Module):
    """CPE -> patch attention -> MLP, pre-norm residual throughout."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0) -> None:
        super().__init__()
        # conditional position encoding: depthwise mixing along the curve plus a
        # projection of the point's own offset inside the patch.  Stands in for
        # PTv3's submanifold-conv xCPE without needing sparse convolution.
        self.cpe_conv = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.cpe_xyz = nn.Linear(3, dim)
        self.cpe_norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PatchAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.dp = drop_path

    def _drop(self, x: torch.Tensor) -> torch.Tensor:
        if self.dp <= 0.0 or not self.training:
            return x
        keep = 1 - self.dp
        return x * x.new_empty(x.shape[0], 1, 1).bernoulli_(keep) / keep

    def forward(self, x: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        h = self.cpe_conv(x.transpose(1, 2)).transpose(1, 2)
        # coords are mm; 0.2 puts a typical intra-patch offset near unity
        x = x + self.cpe_norm(h + self.cpe_xyz(rel * 0.2))
        x = x + self._drop(self.attn(self.norm1(x)))
        x = x + self._drop(self.mlp(self.norm2(x)))
        return x


class Stage(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, patch: int, level: int,
                 variants: list[int], drop_path: float = 0.0) -> None:
        super().__init__()
        self.patch, self.level, self.variants = patch, level, variants
        self.blocks = nn.ModuleList([Block(dim, heads, drop_path=drop_path) for _ in range(depth)])

    def forward(self, x: torch.Tensor, layouts: dict[tuple[int, int], Layout]) -> torch.Tensor:
        c = x.shape[1]
        for blk, var in zip(self.blocks, self.variants):
            lay = layouts[(self.level, var)]
            f = x[lay.sel].view(-1, self.patch, c)
            f = blk(f, lay.rel.view(-1, self.patch, 3)).reshape(-1, c)
            x = f[lay.inv_sel]                     # back to point order
        return x


def pool(x: torch.Tensor, inv: torch.Tensor, cnt: torch.Tensor, m: int) -> torch.Tensor:
    """Average the features of every point that shares a coarse voxel.

    Accumulates in fp32: under autocast x is bf16 and summing a voxel's members
    in bf16 loses more than the pooling is worth.  scatter_add_, not
    index_add_: the ONNX exporter lowers index_add_ to a ScatterND that keeps
    one contribution per voxel.
    """
    out = torch.zeros(m, x.shape[1], device=x.device, dtype=torch.float32)
    out.scatter_add_(0, inv.unsqueeze(1).expand(-1, x.shape[1]), x.float())
    return (out / cnt.unsqueeze(1)).to(x.dtype)


class ToothSegNet(nn.Module):
    def __init__(self,
                 in_ch: int = 7,
                 num_classes: int = 17,
                 enc_dims: tuple[int, ...] = (32, 64, 128, 256, 384),
                 enc_depths: tuple[int, ...] = (1, 1, 2, 4, 2),
                 enc_heads: tuple[int, ...] = (2, 4, 8, 16, 24),
                 dec_dims: tuple[int, ...] = (256, 128, 96, 64),
                 dec_depths: tuple[int, ...] = (1, 1, 1, 1),
                 dec_heads: tuple[int, ...] = (16, 8, 6, 4),
                 patch: int = 1024,
                 drop_path: float = 0.1,
                 n_orders: int = 4,
                 inst_head: nn.Module | None = None,
                 kv_level: int = 1) -> None:
        super().__init__()
        self.inst_head = inst_head
        self.kv_level = kv_level
        self.num_classes = num_classes
        self.patch = patch
        self.enc_depths, self.dec_depths = tuple(enc_depths), tuple(dec_depths)
        self.n_stage = len(enc_dims)
        self.stem = nn.Sequential(nn.Linear(in_ch, enc_dims[0]), nn.LayerNorm(enc_dims[0]), nn.GELU())
        dpr = torch.linspace(0, drop_path, sum(enc_depths)).tolist()

        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        k = 0
        for i, (d, dep, h) in enumerate(zip(enc_dims, enc_depths, enc_heads)):
            var = [(i + j) % n_orders for j in range(dep)]
            self.enc.append(Stage(d, dep, h, patch, i, var, drop_path=dpr[k]))
            k += dep
            if i + 1 < self.n_stage:
                self.down.append(nn.Sequential(nn.Linear(d, enc_dims[i + 1]),
                                               nn.LayerNorm(enc_dims[i + 1]), nn.GELU()))

        self.dec, self.up = nn.ModuleList(), nn.ModuleList()
        prev = enc_dims[-1]
        for i, (d, dep, h) in enumerate(zip(dec_dims, dec_depths, dec_heads)):
            lvl = self.n_stage - 2 - i
            var = [(i + 1 + j) % n_orders for j in range(dep)]
            self.up.append(nn.Sequential(nn.Linear(prev + enc_dims[lvl], d), nn.LayerNorm(d), nn.GELU()))
            self.dec.append(Stage(d, dep, h, patch, lvl, var))
            prev = d

        # The semantic / offset heads are auxiliary tasks of the instance model:
        # trained alongside it, not run at inference.
        self.head_sem = nn.Sequential(nn.Linear(prev, prev), nn.GELU(), nn.Linear(prev, num_classes))
        self.head_off = nn.Sequential(nn.Linear(prev, prev), nn.GELU(), nn.Linear(prev, 3))

    def backbone(self, feat: torch.Tensor, geo: Geometry) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """-> (L0 point features, decoder features by level)."""
        levels, layouts = geo.levels, geo.layouts
        x = self.stem(feat)
        skips, dec_feat = [], {}
        for i in range(self.n_stage):
            x = self.enc[i](x, layouts)
            if i + 1 < self.n_stage:
                skips.append(x)
                nxt = levels[i + 1]
                x = pool(self.down[i](x), nxt.inv, nxt.cnt, nxt.n)
        for i in range(len(self.dec)):
            lvl = self.n_stage - 2 - i
            x = self.up[i](torch.cat([x[levels[lvl + 1].inv], skips[lvl]], 1))
            x = self.dec[i](x, layouts)
            dec_feat[lvl] = x
        return x, dec_feat

    def forward(self, feat: torch.Tensor, geo: Geometry) -> dict:
        """feat [M, in_ch] packed over the batch -> cls [B,K,2] + mask [M,K] logits.

        In training the instance model also returns the decoder's per-layer
        coarse masks (`aux`) and its auxiliary semantic / offset predictions.
        A model without an instance head returns only those two (the 17-class
        toothseg3d_300um family, and the semantic pre-training stage).
        """
        x, dec_feat = self.backbone(feat, geo)
        if self.inst_head is None:
            return {"sem": self.head_sem(x), "off": self.head_off(x)}
        kv = dec_feat[self.kv_level]
        out = self.inst_head(x, geo.levels[0].counts, kv, geo.levels[self.kv_level].coord,
                             geo.pad_index, geo.pad_valid, geo.seed)
        if self.training:
            out["sem"], out["off"] = self.head_sem(x), self.head_off(x)
        return out
