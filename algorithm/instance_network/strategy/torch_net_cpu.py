from __future__ import annotations

import torch

from common.device import CPU
from interface.strategy import IStrategy
from layer.checkpoint import LoadedModel, build_network

from ..instance_network import InstanceNetworkInput, InstanceNetworkOutput


class TorchNetCPU(IStrategy[InstanceNetworkInput, InstanceNetworkOutput]):
    """fp32 forward on the host.  bf16 autocast exists on the CPU too, but on
    most CPUs without AMX it is slower than fp32."""

    def __init__(self, model: LoadedModel) -> None:
        self._net = build_network(model.spec)
        self._net.load_state_dict(model.state_dict)
        self._net.eval()

    @torch.inference_mode()
    def execute(self, inp: InstanceNetworkInput, out: InstanceNetworkOutput) -> bool:
        res = self._net(inp.features.to(CPU), inp.geometry.to(CPU))
        out.cls_logits, out.mask_logits = res["cls"].float(), res["mask"].float()
        return True
