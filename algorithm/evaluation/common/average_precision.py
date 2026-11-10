"""ScanNet-style instance AP accumulated over scans (port of `inst_metrics.APMeter`).

Tooth identity plays no part: a prediction is right when its vertex set
matches a true tooth, whatever number either carries.  Predictions of all
scans are ranked by score together, matched greedily at an IoU threshold, and
precision is integrated over recall.  The ranking, the tie order and the
interpolation are the original's, so the numbers compare one to one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THRESHOLDS = np.arange(0.5, 1.0, 0.05)       # the ScanNet mAP sweep


@dataclass
class ScanRecord:
    """What AP needs from one scan."""

    scores: np.ndarray               # [P] score per predicted instance
    iou: np.ndarray                  # [P,G] against every true instance
    n_gt: int


class InstanceAPMeter:
    def __init__(self, thresholds: np.ndarray = THRESHOLDS) -> None:
        self._th = np.asarray(thresholds)
        self._recs: list[tuple[float, np.ndarray, int]] = []     # (score, iou row, scan)
        self._n_gt = 0
        self._n_scan = 0

    def add(self, record: ScanRecord) -> None:
        sid = self._n_scan
        self._n_scan += 1
        self._n_gt += record.n_gt
        order = np.argsort(-record.scores) if len(record.scores) else np.array([], int)
        for p in order:
            self._recs.append((float(record.scores[p]), record.iou[p], sid))

    def ap_at(self, threshold: float) -> float:
        """Average precision at one IoU threshold, scores ranked across all scans."""
        if not self._recs or self._n_gt == 0:
            return 0.0
        recs = sorted(self._recs, key=lambda r: -r[0])
        taken: dict[int, set[int]] = {}
        tp, fp = np.zeros(len(recs)), np.zeros(len(recs))
        for i, (_, iou, sid) in enumerate(recs):
            used = taken.setdefault(sid, set())
            best, bi = 0.0, -1
            for g in range(len(iou)):
                if g not in used and iou[g] > best:
                    best, bi = iou[g], g
            if best >= threshold and bi >= 0:
                used.add(bi)
                tp[i] = 1
            else:
                fp[i] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / self._n_gt
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        # all-point interpolation (VOC2010 / COCO style)
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        r_prev, ap = 0.0, 0.0
        for r, p in zip(rec, prec):
            ap += (r - r_prev) * p
            r_prev = r
        return float(ap)

    def result(self) -> dict[str, object]:
        aps = {float(t): self.ap_at(t) for t in self._th}
        return {"mAP": float(np.mean(list(aps.values()))), "AP50": self.ap_at(0.5),
                "AP25": self.ap_at(0.25), "per_threshold": aps,
                "n_gt": self._n_gt, "n_pred": len(self._recs)}
