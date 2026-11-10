from __future__ import annotations

import json

from interface.strategy import IStrategy

from ..result_encoder import ResultEncoderInput, ResultEncoderOutput


class JsonEncoderCPU(IStrategy[ResultEncoderInput, ResultEncoderOutput]):
    """Teeth3DS label format.  The instance model predicts no FDI number, so
    `labels` is just the tooth flag (1 = tooth), as toothseg3d's batch_infer."""

    def execute(self, inp: ResultEncoderInput, out: ResultEncoderOutput) -> bool:
        ins = inp.instances.cpu().tolist()
        inp.path.parent.mkdir(parents=True, exist_ok=True)
        with open(inp.path, "w") as fh:
            json.dump({"id_patient": inp.name, "jaw": inp.jaw,
                       "labels": [int(i > 0) for i in ins], "instances": ins}, fh)
        out.written = inp.path
        return True
