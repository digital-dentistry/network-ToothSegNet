from __future__ import annotations

import torch

from common.device import cuda_device
from interface.strategy import IStrategy
from layer.checkpoint import LoadedModel, build_network

from ..instance_network import InstanceNetworkInput, InstanceNetworkOutput


class TorchNetGPU(IStrategy[InstanceNetworkInput, InstanceNetworkOutput]):
    """bf16 autocast forward -- the precision the models were trained in.

    A geometry built on the host arrives here in two packed, pinned transfers
    (`Geometry.to`).  Outputs stay on the device for the post-processing.
    """

    def __init__(self, model: LoadedModel) -> None:
        self._device = cuda_device()
        self._net = build_network(model.spec)
        self._net.load_state_dict(model.state_dict)
        self._net.to(self._device).eval()

    @torch.inference_mode()
    def execute(self, inp: InstanceNetworkInput, out: InstanceNetworkOutput) -> bool:
        feat = inp.features.to(self._device, non_blocking=True)
        geo = inp.geometry.to(self._device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res = self._net(feat, geo)
        out.cls_logits, out.mask_logits = res["cls"].float(), res["mask"].float()
        return True
