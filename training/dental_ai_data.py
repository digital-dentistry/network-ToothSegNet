"""dental_ai의 데이터 (data/ios 캐시 -> dental_ai.dataset.ToothSegLoader) 위에 이 network의 geometry를 얹는다.

dental_ai의 ToothSegDataset이 augmentation · downsample · feature · target을 만든다.  TeethDataset의
포트라서 feature가 ("xyz", "normal", "jaw"[, "colour"])이면 같은 seed에서 TeethDataset과 같은 점과 값이
나온다 (색을 칠하는 경로만 다르다: dental_ai는 in-house 실측 컬러맵으로 칠한다).

network 전용 geometry (serialization hierarchy, query seed)는 dental_ai가 만들지 않으므로 여기서 worker
안에서 ModelInput으로 붙여 TrainingItem으로 넘긴다 — train.py의 collate와 학습 루프는 그대로다.
feature는 dental_ai 것을 쓰고, ModelInput이 함께 만드는 feature는 버린다.
"""
from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DENTAL_AI = ROOT.parent / "dental_ai/src"          # pip install -e 없이도 import되게
if DENTAL_AI.is_dir() and str(DENTAL_AI) not in sys.path:
    sys.path.insert(0, str(DENTAL_AI))

from dental_ai.dataset.ToothSegLoader import ToothSegConfig, ToothSegDataset, feature_slices  # noqa: E402
from dental_ai.dataset.algorithm.augmentation.augmentation import AugmentationParams  # noqa: E402, F401

from algorithm.model_input.common.builder import CurveKey  # noqa: E402
from algorithm.model_input.model_input import (GeometrySpec, ModelInput, ModelInputInput,  # noqa: E402
                                               ModelInputOutput, model_input_type)
from layer.checkpoint import DEC_DEPTHS, ENC_DEPTHS, KV_LEVEL, PATCH  # noqa: E402
from training.dataset import EYE, TrainingItem  # noqa: E402

IOS = ROOT.parent / "data/ios"
# --ios로 고른 데이터셋마다 학습에 쓰는 공식 split (test split은 쓰지 않는다).  None = split이 없어 전부
IOS_TRAIN_SPLITS: dict[str, list[str] | None] = {
    "teeth3ds": ["splits/Teeth3DS_train_test_split/training_upper.txt",
                 "splits/Teeth3DS_train_test_split/training_lower.txt"],
    "3d_iosseg": ["splits/train.txt"],
    "poseidon3d": ["splits/train.txt", "splits/validation.txt"],
    "cbct_ios_fusion_liu2023": None,
}


def ios_names(datasets: list[str]) -> tuple[list[Path], list[str]]:
    """data/ios 데이터셋들 -> (캐시 폴더, 학습에 쓸 스캔 이름)."""
    caches, names = [], []
    for d in datasets:
        if d not in IOS_TRAIN_SPLITS:
            raise SystemExit(f"--ios {d}: choose from {list(IOS_TRAIN_SPLITS)}")
        cache = IOS / d / "cache"
        caches.append(cache)
        files = IOS_TRAIN_SPLITS[d]
        if files is None:
            names += sorted(p.stem for p in cache.glob("*.npz"))
        else:
            for f in files:
                names += [ln.strip() for ln in (IOS / d / f).read_text().splitlines() if ln.strip()]
    return caches, names


class DentalAIDataset(Dataset):
    """dental_ai ToothSegDataset + segmentation3d geometry -> TrainingItem."""

    def __init__(self, cache_dirs: list[Path], names: list[str], config: ToothSegConfig, curve: CurveKey,
                 n_query: int) -> None:
        if "normal" not in config.features:
            raise ValueError('the geometry needs normals: keep "normal" in features')
        self.inner = ToothSegDataset(cache_dirs, names, config)
        self.config, self.curve = config, curve
        self.spec = GeometrySpec(enc_depths=ENC_DEPTHS, dec_depths=DEC_DEPTHS, patch=PATCH, pad_level=KV_LEVEL,
                                 n_query=n_query)
        self._normals = feature_slices(config)["normal"]
        self._model_input: ModelInput | None = None

    @property
    def names(self) -> list[str]:
        return self.inner.names

    def __len__(self) -> int:
        return len(self.inner)

    def load(self, name: str):
        """(ScanSample, jaw_upper) — train.py가 class 빈도를 셀 때 쓴다."""
        return self.inner.load(name)

    def __getitem__(self, i: int) -> TrainingItem:
        if self._model_input is None:                # worker 안에서 처음 쓸 때 만든다
            self._model_input = ModelInput()
            self._model_input.set_strategy(model_input_type(self.curve, gpu=False))
        it = self.inner[i]
        out = ModelInputOutput()
        normals = it.features[:, self._normals].contiguous()
        if not self._model_input.execute(ModelInputInput(it.coord, normals, it.jaw_upper, None, EYE, self.config.voxel_mm,
                                                         self.spec, cells=it.grid), out):
            raise ValueError(f"{it.name}: could not build the network geometry")
        return TrainingItem(it.name, it.features, out.geometry, it.classes, it.instances, it.arch_positions,
                            it.offset_targets)
