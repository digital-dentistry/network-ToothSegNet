from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from algorithm.model_input.common.builder import CurveKey
from layer.backbone import ToothSegNet
from layer.instance_head import InstanceHead

# toothseg3d/train.py and train_inst.py: the trained architecture
ENC_DIMS = (32, 64, 128, 256, 384)
ENC_DEPTHS = (1, 1, 2, 4, 2)
ENC_HEADS = (2, 4, 8, 16, 24)
DEC_DIMS = (256, 128, 96, 64)
DEC_DEPTHS = (1, 1, 1, 1)
DEC_HEADS = (16, 8, 6, 4)
KV_LEVEL = 1          # the instance queries cross-attend to the 2x coarser level
NUM_POS = 9           # auxiliary arch-position classes of the instance model
NUM_CLASSES = 17      # the semantic model: gingiva + 16 FDI positions
PATCH = 1024


@dataclass(frozen=True)
class ModelSpec:
    """What a checkpoint says about itself; everything downstream reads this."""

    n_query: int = 32
    in_ch: int = 7
    head_dim: int = 128
    head_layers: int = 4
    voxel_mm: float = 0.25
    curve: CurveKey = CurveKey.LEGACY   # the curve key it was trained on; every toothseg3d checkpoint is legacy
    epoch: int | None = None
    source: str = "random init"

    @property
    def uses_colour(self) -> bool:
        return self.in_ch >= 10


@dataclass
class LoadedModel:
    spec: ModelSpec
    state_dict: dict[str, torch.Tensor] = field(repr=False)


def build_network(spec: ModelSpec) -> ToothSegNet:
    head = InstanceHead(
        point_dim=DEC_DIMS[-1], 
        kv_dim=DEC_DIMS[-1 - KV_LEVEL], 
        dim=spec.head_dim,
        n_query=spec.n_query, 
        layers=spec.head_layers, 
        heads=8)
    return ToothSegNet(
        in_ch=spec.in_ch, 
        num_classes=NUM_POS, 
        enc_dims=ENC_DIMS,
        enc_depths=ENC_DEPTHS, 
        enc_heads=ENC_HEADS, 
        dec_dims=DEC_DIMS,
        dec_depths=DEC_DEPTHS, 
        dec_heads=DEC_HEADS, 
        patch=PATCH,
        inst_head=head, 
        kv_level=KV_LEVEL)


def build_semantic_network(in_ch: int = 7) -> ToothSegNet:
    """The 17-class semantic model: the instance model's backbone without the
    query head.  Trained first, it initialises the instance model's backbone."""
    return ToothSegNet(in_ch=in_ch, num_classes=NUM_CLASSES, enc_dims=ENC_DIMS,
                       enc_depths=ENC_DEPTHS, enc_heads=ENC_HEADS, dec_dims=DEC_DIMS,
                       dec_depths=DEC_DEPTHS, dec_heads=DEC_HEADS, patch=PATCH)


def load_checkpoint(path: Path) -> LoadedModel:
    """A toothseg3d mask-transformer checkpoint (toothinst_* / toothseg3d_ios_*)."""
    # weights_only=False: our own checkpoints carry the argparse namespace as a dict
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "queries" not in ck:
        raise NotImplementedError(
            f"{path.name} is a 17-class semantic checkpoint (toothseg3d_300um family); "
            "only the mask-transformer instance models are ported")
    spec = ModelSpec(n_query=int(ck["queries"]), in_ch=int(ck.get("in_ch", 7)),
                     head_dim=int(ck.get("head_dim", 128)), head_layers=int(ck.get("head_layers", 4)),
                     voxel_mm=float(ck["args"]["res"]), curve=CurveKey[str(ck.get("curve", "legacy")).upper()],
                     epoch=ck.get("epoch"), source=str(path))
    return LoadedModel(spec, ck["model"])


def random_model(spec: ModelSpec, seed: int = 0) -> LoadedModel:
    """Untrained weights with the trained architecture -- shapes and timing only."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = build_network(spec)
    return LoadedModel(spec, net.state_dict())
