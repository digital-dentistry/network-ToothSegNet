"""The four steps of one segmentation, each owning the algorithms it runs.

A stage is single-threaded state: the synchronous `Segmenter` calls them in
order, and the threaded application gives each one to its own workflow thread.
Stages never share an algorithm object, so no locking is needed.

    LoadStage     file -> Scan                 decode, normals
    PrepareStage  Scan -> PreparedScan         downsample, centre, features + geometry per pose
    InferStage    PreparedScan -> InferredScan network, TTA fusion
    FinishStage   InferredScan -> Result       upsample, instances, scores, files
"""
from __future__ import annotations

from pathlib import Path

import torch

from algorithm.instance_extraction.instance_extraction import (
    InstanceExtraction, InstanceExtractionInput, InstanceExtractionOutput, InstanceExtractionType)
from layer.checkpoint import LoadedModel, ModelSpec
from algorithm.instance_network.instance_network import (
    InstanceNetwork, InstanceNetworkInput, InstanceNetworkOutput, InstanceNetworkType)
from algorithm.mask_upsample.mask_upsample import (
    MaskUpsample, MaskUpsampleInput, MaskUpsampleOutput, MaskUpsampleType)
from algorithm.model_input.common.features import standardise_colour, unit_normals
from algorithm.model_input.model_input import (
    GeometrySpec, ModelInput, ModelInputInput, ModelInputOutput, model_input_type)
from algorithm.normal_estimation.normal_estimation import (
    NormalEstimation, NormalEstimationInput, NormalEstimationOutput, NormalEstimationType)
from algorithm.query_fusion.query_fusion import (
    QueryFusion, QueryFusionInput, QueryFusionOutput, QueryFusionType)
from algorithm.result_encoder.result_encoder import (
    ResultEncoder, ResultEncoderInput, ResultEncoderOutput, ResultEncoderType)
from algorithm.scan_decoder.common.naming import guess_jaw
from algorithm.scan_decoder.scan_decoder import ScanDecoder, ScanDecoderInput, ScanDecoderOutput
from algorithm.voxel_downsample.voxel_downsample import (
    VoxelDownsample, VoxelDownsampleInput, VoxelDownsampleOutput, VoxelDownsampleType)
from common.device import CPU, StageTimer, cuda_device

from .config import Placement, SegmentationConfig, UpsampleMode
from .rotations import tta_rotations
from .types import InferredScan, PreparedScan, Scan, SegmentationResult


def _device(gpu: bool) -> torch.device:
    return cuda_device() if gpu else CPU


def _require(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(what)


class LoadStage:
    def __init__(self, config: SegmentationConfig, placement: Placement) -> None:
        self._config = config
        self._gpu = placement.normals
        self._device = _device(placement.normals)
        self._decoder = ScanDecoder()
        self._normals = NormalEstimation()

    def run(self, path: Path) -> Scan:
        timer = StageTimer(self._device, self._config.timing)
        dec = ScanDecoderOutput()
        self._decoder.set_strategy_for(path)
        _require(self._decoder.execute(ScanDecoderInput(path, self._config.merge_stl_vertices), dec),
                 f"could not decode {path.name}")
        timer.mark("load")

        # the file's own normals, then the faces, and PCA only as a last resort
        if dec.normals is not None:
            normals = unit_normals(dec.normals)
        else:
            mesh = dec.faces is not None
            kind = {(True, False): NormalEstimationType.MESH_CPU, (True, True): NormalEstimationType.MESH_GPU,
                    (False, False): NormalEstimationType.PCA_CPU, (False, True): NormalEstimationType.PCA_GPU}
            self._normals.set_strategy(kind[(mesh, self._gpu)])
            est = NormalEstimationOutput()
            _require(self._normals.execute(NormalEstimationInput(dec.positions, dec.faces), est),
                     f"could not estimate normals for {path.name}")
            normals = est.normals
        timer.mark("normals")
        return Scan(path=path, name=path.stem, jaw=guess_jaw(path, self._config.jaw),
                    positions=dec.positions, normals=normals, faces=dec.faces,
                    colours=dec.colours, timing=timer.ms)


class PrepareStage:
    def __init__(self, config: SegmentationConfig, placement: Placement, spec: ModelSpec) -> None:
        self._config = config
        self._spec = spec
        self._voxel_mm = config.voxel_mm or spec.voxel_mm
        self._device = _device(placement.downsample)
        self._downsample = VoxelDownsample()
        self._downsample.set_strategy(VoxelDownsampleType.GRID_GPU if placement.downsample
                                      else VoxelDownsampleType.GRID_CPU)
        self._model_input = ModelInput()
        self._model_input.set_strategy(model_input_type(config.curve or spec.curve, placement.geometry))
        self._geometry_spec = GeometrySpec(n_query=spec.n_query)
        self._rotations = tta_rotations(max(1, config.tta))

    def run(self, scan: Scan) -> PreparedScan:
        timer = StageTimer(self._device, self._config.timing)
        positions = scan.positions.to(self._device, non_blocking=True)
        ds = VoxelDownsampleOutput()
        _require(self._downsample.execute(VoxelDownsampleInput(positions, self._voxel_mm), ds),
                 f"{scan.name}: nothing to downsample")
        rep = ds.representative
        # centred in float64: the model is translation-free by construction
        # (every training scan is centred), so only consistency matters here
        centre = positions[rep].double().mean(0).float()
        points = positions[rep] - centre
        normals = unit_normals(scan.normals.to(self._device)[rep])
        colour = None
        if self._spec.uses_colour:
            rgb = (scan.colours.to(self._device)[rep] if scan.colours is not None
                   else torch.full_like(points, 0.5))
            colour = standardise_colour(rgb)
        timer.mark("downsample")

        batches = []
        jaw_upper = scan.jaw == "upper"
        for rot in torch.split(self._rotations, max(1, self._config.tta_batch)):
            out = ModelInputOutput()
            _require(self._model_input.execute(ModelInputInput(
                points, normals, jaw_upper, colour, rot, self._voxel_mm, self._geometry_spec,
                cells=ds.cells), out), f"{scan.name}: could not build the network input")
            batches.append(out)
        timer.mark("geometry")
        scan.timing.update(timer.ms)
        return PreparedScan(scan=scan, points=points, centre=centre, inverse=ds.inverse,
                            voxel_mm=self._voxel_mm, batches=batches)


class InferStage:
    def __init__(self, config: SegmentationConfig, placement: Placement, model: LoadedModel) -> None:
        self._config = config
        self._device = _device(placement.network)
        self._network = InstanceNetwork(model)
        self._network.set_strategy(InstanceNetworkType.TORCH_GPU if placement.network
                                   else InstanceNetworkType.TORCH_CPU)
        self._fusion = QueryFusion()
        self._fusion.set_strategy(QueryFusionType.HUNGARIAN_GPU if placement.fusion
                                  else QueryFusionType.HUNGARIAN_CPU)

    def run(self, prepared: PreparedScan) -> InferredScan:
        timer = StageTimer(self._device, self._config.timing)
        cls, mask = [], []
        for batch in prepared.batches:
            out = InstanceNetworkOutput()
            _require(self._network.execute(InstanceNetworkInput(batch.features, batch.geometry), out),
                     f"{prepared.scan.name}: forward failed")
            cls.append(out.cls_logits)
            mask.append(out.mask_logits)
        timer.mark("network")

        fused = QueryFusionOutput()
        self._fusion.execute(QueryFusionInput(torch.cat(cls), torch.cat(mask),
                                              prepared.points.shape[0]), fused)
        timer.mark("fusion")
        prepared.scan.timing.update(timer.ms)
        return InferredScan(prepared=prepared, confidence=fused.confidence, scores=fused.scores)


class FinishStage:
    def __init__(self, config: SegmentationConfig, placement: Placement,
                 out_dir: Path | None = None, write_json: bool = True) -> None:
        self._config = config
        self._device = _device(placement.upsample)
        self._out_dir = out_dir
        self._write_json = write_json
        self._upsample = MaskUpsample()
        knn = config.upsample is UpsampleMode.KNN
        self._upsample.set_strategy({(True, False): MaskUpsampleType.KNN_CPU,
                                     (True, True): MaskUpsampleType.KNN_GPU,
                                     (False, False): MaskUpsampleType.VOXEL_CPU,
                                     (False, True): MaskUpsampleType.VOXEL_GPU}[(knn, placement.upsample)])
        self._extraction = InstanceExtraction()
        self._extraction.set_strategy(InstanceExtractionType.ARGMAX_GPU if placement.extraction
                                      else InstanceExtractionType.ARGMAX_CPU)
        self._encoder = ResultEncoder()

    def run(self, inferred: InferredScan) -> SegmentationResult:
        prep, scan, cfg = inferred.prepared, inferred.prepared.scan, self._config
        timer = StageTimer(self._device, cfg.timing)
        positions = scan.positions.to(self._device, non_blocking=True)
        up = MaskUpsampleOutput()
        self._upsample.execute(MaskUpsampleInput(
            inferred.confidence, prep.points, positions - prep.centre.to(self._device),
            prep.inverse, prep.voxel_mm, cfg.score_thresh, cfg.knn), up)
        timer.mark("upsample")

        ext = InstanceExtractionOutput()
        self._extraction.execute(InstanceExtractionInput(
            up.confidence, up.active, inferred.scores, positions, cfg.score_thresh,
            cfg.min_frac, cfg.erode_mm, cfg.mask_confidence), ext)
        instances = ext.instances.cpu()
        timer.mark("instances")

        result = SegmentationResult(
            path=scan.path, name=scan.name, jaw=scan.jaw, n_vertices=positions.shape[0],
            instances=instances.numpy(), scores=ext.scores.cpu().tolist(),
            vertex_confidence=ext.vertex_confidence.cpu().numpy())
        if self._out_dir is not None:
            result.outputs = self._write(scan, instances)
            timer.mark("write")
        scan.timing.update(timer.ms)
        result.timing = scan.timing
        return result

    def _write(self, scan: Scan, instances: torch.Tensor) -> list[Path]:
        written = []
        jobs = [(ResultEncoderType.PLY_CPU, f"{scan.name}_seg.ply")]
        if self._write_json:
            jobs.append((ResultEncoderType.JSON_CPU, f"{scan.name}_pred.json"))
        for kind, file_name in jobs:
            self._encoder.set_strategy(kind)
            out = ResultEncoderOutput()
            self._encoder.execute(ResultEncoderInput(self._out_dir / file_name, scan.name, scan.jaw,
                                                     scan.positions, instances, scan.faces), out)
            written.append(out.written)
        return written
