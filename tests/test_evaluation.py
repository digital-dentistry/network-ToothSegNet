"""Metrics against `inst_metrics.APMeter` and `evaluate.jaw_metrics`."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tests.conftest import DEVICES, teeth3ds_scans


def _fake_scan(seed: int, n: int = 6000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground truth and a plausible prediction: blurred borders, a merge, a miss,
    specks under the point floor, and scores."""
    rng = np.random.RandomState(seed)
    g = rng.randint(6, 14)
    gt = np.sort(rng.randint(0, g + 1, n))               # contiguous runs, like teeth
    pred = gt.copy()
    pred[rng.rand(n) < 0.05] = 0                          # blurred borders
    pred[gt == 2] = 3                                     # two teeth merged
    pred[gt == 5] = 0                                     # one missed
    pred[rng.choice(n, 20, replace=False)] = g + 3        # a speck under 30 points
    pred[(gt == 0) & (rng.rand(n) < 0.1)] = g + 4         # a false positive
    perm = rng.permutation(np.arange(1, g + 5))           # ids carry no meaning
    pred = np.where(pred > 0, perm[pred - 1], 0)
    scores = rng.rand(g + 4)
    return pred, gt, scores


def _evaluate(typ, **kw):
    from algorithm.evaluation.evaluation import Evaluation, EvaluationInput, EvaluationOutput
    alg, out = Evaluation(), EvaluationOutput()
    alg.set_strategy(typ)
    assert alg.execute(EvaluationInput(**kw), out)
    return out


@pytest.mark.parametrize("device", DEVICES)
def test_instance_ap_matches_apmeter(device: str) -> None:
    from algorithm.evaluation.common.average_precision import InstanceAPMeter
    from algorithm.evaluation.evaluation import EvaluationType
    from inst_metrics import APMeter, masks_from_labels
    typ = EvaluationType.INSTANCE_GPU if device == "cuda" else EvaluationType.INSTANCE_CPU
    ref, ours = APMeter(), InstanceAPMeter()
    for seed in range(6):
        pred, gt, scores = _fake_scan(seed)
        pm, pid = masks_from_labels(pred, 30)
        gm, _ = masks_from_labels(gt, 1)
        ref_scan = ref.add(pm, [float(scores[i - 1]) for i in pid], gm)
        pt = np.zeros(len(pred), bool)
        for m in pm:
            pt |= m
        ref_tsa = 2 * (pt & (gt != 0)).sum() / max(pt.sum() + (gt != 0).sum(), 1)

        out = _evaluate(typ, pred_instances=torch.from_numpy(pred), gt_instances=torch.from_numpy(gt),
                        scores=torch.from_numpy(scores))
        ours.add(out.record)
        assert out.metrics["tsa"] == pytest.approx(ref_tsa, abs=1e-12)
        for k, v in ref_scan.items():
            assert out.metrics[k] == pytest.approx(v, abs=1e-12), k
    r, o = ref.result(), ours.result()
    for k in ("mAP", "AP50", "AP25", "n_gt", "n_pred"):
        assert o[k] == pytest.approx(r[k], abs=1e-12), k


@pytest.fixture(scope="module")
def jaw() -> dict[str, np.ndarray]:
    from t3ds_io import fdi_to_class, read_labels, read_obj
    path = teeth3ds_scans("upper", 1)[0]
    V, _, _ = read_obj(str(path))
    lab, ins, j = read_labels(str(path)[:-4] + ".json")
    return {"V": V.astype(np.float32), "cls": fdi_to_class(lab, j).astype(np.int64), "ins": ins.astype(np.int64)}


def _prediction(jaw: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[int, int]]:
    """Ground truth with one tooth missed, one mislabelled, one shrunk, and an extra blob."""
    gt, cls_, V = jaw["ins"], jaw["cls"], jaw["V"]
    ids = [i for i in np.unique(gt) if i > 0]
    pred = np.zeros_like(gt)
    label_of = {}
    for new, i in enumerate(ids[1:], 1):                 # ids[0] is missed
        m = gt == i
        if new == 2:                                     # shrink: drop the points far from the centroid
            d = np.linalg.norm(V[m] - V[m].mean(0), axis=1)
            keep = np.flatnonzero(m)[d < np.median(d)]
            pred[keep] = new
        else:
            pred[m] = new
        label_of[new] = int(np.bincount(cls_[m]).argmax())
    label_of[3] = (label_of[3] % 16) + 1                 # wrong label
    blob = (gt == 0) & (np.linalg.norm(V - V[gt == 0][0], axis=1) < 3.0)
    pred[blob] = len(ids) + 1
    label_of[len(ids) + 1] = 5
    return pred, label_of


@pytest.mark.parametrize("device", DEVICES)
def test_challenge_matches_jaw_metrics(jaw, device: str) -> None:
    from algorithm.evaluation.evaluation import EvaluationType
    from evaluate import jaw_metrics
    typ = EvaluationType.CHALLENGE_GPU if device == "cuda" else EvaluationType.CHALLENGE_CPU
    pred, label_of = _prediction(jaw)
    ref = jaw_metrics(jaw["V"], jaw["cls"], jaw["ins"], None, pred, label_of)
    classes = torch.tensor([label_of.get(i, 0) for i in range(1, int(pred.max()) + 1)])
    out = _evaluate(typ, pred_instances=torch.from_numpy(pred), gt_instances=torch.from_numpy(jaw["ins"]),
                    positions=torch.from_numpy(jaw["V"]), gt_classes=torch.from_numpy(jaw["cls"]),
                    pred_classes=classes)
    for k, v in ref.items():
        assert out.metrics[k] == pytest.approx(v, rel=1e-5, abs=1e-6), k


def test_challenge_without_labels(jaw) -> None:
    from algorithm.evaluation.evaluation import EvaluationType
    from evaluate import jaw_metrics
    pred, label_of = _prediction(jaw)
    ref = jaw_metrics(jaw["V"], jaw["cls"], jaw["ins"], None, pred, label_of)
    out = _evaluate(EvaluationType.CHALLENGE_CPU, pred_instances=torch.from_numpy(pred),
                    gt_instances=torch.from_numpy(jaw["ins"]), positions=torch.from_numpy(jaw["V"]),
                    gt_classes=torch.from_numpy(jaw["cls"]))
    assert math.isnan(out.metrics["tir"]) and math.isnan(out.metrics["label_acc"])
    for k in ("tsa", "tla", "centroid_mm", "missed", "extra"):          # identity-free ones still hold
        assert out.metrics[k] == pytest.approx(ref[k], rel=1e-5, abs=1e-6), k


def test_challenge_edge_cases(jaw) -> None:
    from algorithm.evaluation.evaluation import EvaluationType
    from evaluate import jaw_metrics
    none = np.zeros_like(jaw["ins"])
    for pred, gt in ((none, jaw["ins"]), (jaw["ins"], none)):
        ref = jaw_metrics(jaw["V"], jaw["cls"], gt, None, pred, {})
        out = _evaluate(EvaluationType.CHALLENGE_CPU, pred_instances=torch.from_numpy(pred),
                        gt_instances=torch.from_numpy(gt), positions=torch.from_numpy(jaw["V"]),
                        gt_classes=torch.from_numpy(jaw["cls"]), pred_classes=torch.zeros(40, dtype=torch.int64))
        for k, v in ref.items():
            assert out.metrics[k] == pytest.approx(v, rel=1e-5, abs=1e-6), k
