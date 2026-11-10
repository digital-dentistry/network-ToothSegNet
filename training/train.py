"""Train the tooth models on cached scans (port of `toothseg3d/train.py` and
`train_inst.py`).

Two stages, as toothseg3d trained them.  Starting the mask decoder on a random
backbone converges far too slowly, so the backbone is trained first:

    semantic   17-class model (gingiva + 16 FDI positions) + centroid offsets
    instance   mask transformer: queries matched to teeth, the model app.py runs

    PY=/home/sjy/anaconda3/envs/ai/bin/python
    $PY training/preprocess.py                                  # once: raw Teeth3DS -> npz cache
    $PY training/train.py --stage semantic --out runs/semantic
    $PY training/train.py --stage instance --init-from runs/semantic/best.pt --out runs/instance

`--init-from` also takes toothseg3d checkpoints.  The instance checkpoints
load straight into `app.py --ckpt`, which reads the curve key they were
trained on.

Data comes from the dental_ai library by default (`--data dental_ai`:
dental_ai.dataset.ToothSegLoader on the data/ios caches, training/dental_ai_data.py
adds this network's geometry); `--data seg3d` is the original training/dataset.py.
`--ios` picks data/ios datasets and their official train splits:

    $PY training/train.py --stage instance --init-from runs/semantic/best.pt --out runs/instance_ios \\
        --ios teeth3ds 3d_iosseg poseidon3d cbct_ios_fusion_liu2023

`--features` other than the default (xyz normal jaw [colour]) -- e.g. adding
curvature -- trains a model app.py cannot feed yet: the inference pipeline
builds only the default features.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# batch마다 점 수가 크게 달라 고정 블록 할당은 조각나 예약 메모리가 GPU를 넘는다 (RTX 5080 16 GB에서
# 23 GB까지 커져 시스템 RAM으로 넘치고 cuBLAS가 CUBLAS_STATUS_INTERNAL_ERROR로 죽었다).  torch보다 먼저 정한다
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.augmentation.common.params import AugmentationParams  # noqa: E402
from algorithm.evaluation.common.average_precision import InstanceAPMeter  # noqa: E402
from algorithm.evaluation.common.confusion import ConfusionMeter  # noqa: E402
from algorithm.evaluation.evaluation import (Evaluation, EvaluationInput, EvaluationOutput,  # noqa: E402
                                             EvaluationType)
from algorithm.loss.common.targets import class_weights, coarse_inverse  # noqa: E402
from algorithm.loss.loss import Loss, LossInput, LossOutput, LossType, LossWeights  # noqa: E402
from algorithm.model_input.common.builder import CurveKey  # noqa: E402
from layer.backbone import ToothSegNet  # noqa: E402
from layer.checkpoint import KV_LEVEL, NUM_CLASSES, ModelSpec, build_network, build_semantic_network  # noqa: E402
from training.dataset import DatasetConfig, TeethDataset, TrainingBatch, collate, read_split  # noqa: E402

DATA = ROOT.parent / "data/ios/teeth3ds"

# what toothseg3d trained each stage with.  The instance stage ran batch 12 x 1 there; on a 16 GB GPU a
# batch of 12 scans peaks past the memory (13.5 GB at 650 k points, up to 840 k), so it is 6 x 2 here --
# the same effective batch and schedule (the network has only LayerNorm, no batch statistics).
STAGE_DEFAULTS = {
    "semantic": dict(res=0.30, batch=4, accum=2, epochs=120, max_points=60000, in_ch=7, crop_p=0.15,
                     trim_p=0.0, colour_p=0.0, val_every=2),
    "instance": dict(res=0.25, batch=6, accum=2, epochs=300, max_points=70000, in_ch=10, crop_p=0.35,
                     trim_p=0.5, colour_p=0.6, val_every=3),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGE_DEFAULTS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", choices=["dental_ai", "seg3d"], default="dental_ai",
                    help="dental_ai: dental_ai.dataset.ToothSegLoader; seg3d: training/dataset.py")
    ap.add_argument("--ios", nargs="+", default=None,
                    help="data/ios datasets to train on: sets --cache and the names from their train splits")
    ap.add_argument("--features", nargs="+", default=None,
                    help="dental_ai features (default from --in-ch: 7 = xyz normal jaw, 10 = + colour)")
    ap.add_argument("--cache", type=Path, nargs="+", default=[DATA / "cache"],
                    help="cache directories; several are mixed (a name is read from whichever has it)")
    ap.add_argument("--split-dir", type=Path, default=DATA / "splits/Teeth3DS_train_test_split")
    ap.add_argument("--names", type=Path, nargs="+", default=None, help="scan names to train on, one per line")
    ap.add_argument("--val-names", type=Path, nargs="+", default=None,
                    help="explicit validation scans (otherwise a patient-level split of the training names)")
    ap.add_argument("--exclude", default="", help="comma-separated names to hold out")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="train on the first N scans (smoke tests)")
    ap.add_argument("--curve", choices=["legacy", "morton3d"], default="morton3d",
                    help="morton3d: the fixed 3D key; legacy: what the toothseg3d checkpoints used")
    for k in ("res", "max_points", "crop_p", "trim_p", "colour_p"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=float if k != "max_points" else int, default=None)
    for k in ("batch", "accum", "epochs", "in_ch", "val_every"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=5, help="epochs of linear warm-up")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rot-mode", choices=["so3", "limited"], default="so3")
    ap.add_argument("--crop-mix", type=float, nargs=3, default=(0.4, 0.35, 0.25), help="bridge / quadrant / half")
    ap.add_argument("--val-crop", default="", help="instance: also score a forced crop mode on validation")
    ap.add_argument("--queries", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--head-layers", type=int, default=4)
    ap.add_argument("--w-aux", type=float, default=None)
    ap.add_argument("--w-pos", type=float, default=None)
    ap.add_argument("--w-off", type=float, default=None)
    ap.add_argument("--w-dice", type=float, default=None, help="semantic: class dice")
    ap.add_argument("--w-dir", type=float, default=None, help="semantic: offset direction")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="start from a checkpoint's weights (fresh optimiser); shapes that differ stay fresh")
    ap.add_argument("--resume", type=Path, default=None)
    a = ap.parse_args()
    for k, v in STAGE_DEFAULTS[a.stage].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.ios_names = None
    if a.ios:
        from training.dental_ai_data import ios_names
        a.cache, a.ios_names = ios_names(a.ios)
    if a.data == "dental_ai":
        from training.dental_ai_data import ToothSegConfig
        if a.features is None:
            a.features = ["xyz", "normal", "jaw"] + (["colour"] if a.in_ch >= 10 else [])
        a.in_ch = ToothSegConfig(features=tuple(a.features)).in_ch
    elif a.features:
        raise SystemExit("--features needs --data dental_ai")
    return a


# ------------------------------------------------------------------ data
def _read_all(paths: Path | list[Path]) -> list[str]:
    return [n for p in (paths if isinstance(paths, list) else [paths]) for n in read_split(p)]


def split_names(a: argparse.Namespace) -> tuple[list[str], list[str]]:
    if getattr(a, "ios_names", None):
        names = list(a.ios_names)
    elif a.names:
        names = _read_all(a.names)
    else:
        names = read_split(a.split_dir / "training_upper.txt") + read_split(a.split_dir / "training_lower.txt")
    if a.exclude:
        drop = set(a.exclude.split(","))
        names = [n for n in names if n not in drop]
    if a.val_names:
        val = _read_all(a.val_names)
        held = set(val)
        train = [n for n in names if n not in held]
    else:
        # by *patient*: upper and lower of one patient on both sides of the
        # split would make the validation score read high
        rng = np.random.default_rng(0)
        by_pid: dict[str, list[str]] = {}
        for n in names:
            by_pid.setdefault(n.split("_")[0], []).append(n)
        pids = sorted(by_pid)
        rng.shuffle(pids)
        want, val = int(len(names) * a.val_frac), []
        for p in pids:
            if len(val) >= want:
                break
            val += by_pid[p]
        held = set(val)
        train = [n for n in names if n not in held]
    if not val:                                  # a handful of scans: score on the training ones
        val, train = list(names), list(names)
    if a.limit:
        train, val = train[:a.limit], val[:max(4, a.limit // 8)]
    return train, val


def make_loaders(a: argparse.Namespace, train_names: list[str], val_names: list[str]
                 ) -> tuple[DataLoader, dict[str, DataLoader]]:
    n_query = a.queries if a.stage == "instance" else 0
    colour = ("colour" in a.features) if a.data == "dental_ai" else a.in_ch >= 10
    if a.data == "dental_ai":
        from training.dental_ai_data import AugmentationParams as DentalAIAugmentation
        from training.dental_ai_data import DentalAIDataset, ToothSegConfig
        aug = DentalAIAugmentation(crop_p=a.crop_p, trim_p=a.trim_p, colour_p=a.colour_p, rotation=a.rot_mode,
                                   crop_mix=tuple(a.crop_mix))
        base = dict(voxel_mm=a.res, features=tuple(a.features), max_points=a.max_points)

        def dataset(names: list[str], **kw: object) -> DentalAIDataset:
            return DentalAIDataset(a.cache, names, ToothSegConfig(**base, **kw), CurveKey[a.curve.upper()], n_query)
    else:
        aug = AugmentationParams(crop_p=a.crop_p, trim_p=a.trim_p, colour_p=a.colour_p, rotation=a.rot_mode,
                                 crop_mix=tuple(a.crop_mix))
        base = dict(voxel_mm=a.res, in_ch=a.in_ch, n_query=n_query, curve=CurveKey[a.curve.upper()],
                    max_points=a.max_points)

        def dataset(names: list[str], **kw: object) -> TeethDataset:
            return TeethDataset(a.cache, names, DatasetConfig(**base, **kw))
    train = dataset(train_names, augmentation=aug)
    vals = {"grey": dataset(val_names, train=False)}
    if a.stage == "instance" and colour:
        # score the coloured path too: watching only the colourless one is how
        # toothseg3d's first colour model hid its own collapse
        vals["colour"] = dataset(val_names, train=False, colour_force=True)
    if a.stage == "instance" and a.val_crop:
        vals["crop"] = dataset(val_names, train=False, colour_force=colour, crop_force=a.val_crop)
    kw = dict(collate_fn=collate, persistent_workers=a.workers > 0, worker_init_fn=_worker_init)
    dl_train = DataLoader(train, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                          prefetch_factor=3 if a.workers else None, **kw)
    dl_vals = {k: DataLoader(v, batch_size=a.batch, shuffle=False, num_workers=max(1, a.workers // 4)
                             if a.workers else 0, **kw) for k, v in vals.items()}
    return dl_train, dl_vals


def _worker_init(_: int) -> None:
    torch.set_num_threads(1)                     # one thread per worker; there are many workers


# ----------------------------------------------------------------- model
def build_model(a: argparse.Namespace) -> ToothSegNet:
    if a.stage == "semantic":
        return build_semantic_network(a.in_ch)
    return build_network(
            ModelSpec(
                n_query=a.queries, 
                in_ch=a.in_ch, 
                head_dim=a.head_dim,
                head_layers=a.head_layers, 
                voxel_mm=a.res))


def init_from(net: ToothSegNet, path: Path, say) -> None:
    """Copy every tensor whose name and shape match.  A stem that grew when
    colour channels were added takes the old columns and zeros in the new ones:
    colour standardises to 0 on a colourless scan, so the model starts out
    behaving exactly as before."""
    src = torch.load(path, map_location="cpu", weights_only=False)["model"]
    own = net.state_dict()
    take, widened = {}, 0
    for k, v in src.items():
        if k not in own:
            continue
        o = own[k]
        if o.shape == v.shape:
            take[k] = v
        elif o.ndim == v.ndim == 2 and o.shape[0] == v.shape[0] and o.shape[1] > v.shape[1]:
            w = torch.zeros_like(o)
            w[:, :v.shape[1]] = v
            take[k] = w
            widened += 1
    net.load_state_dict(take, strict=False)
    say(f"initialised from {path}: {len(take)}/{len(own)} tensors"
        + (f" ({widened} widened for extra input channels)" if widened else ""))


def semantic_class_weights(a: argparse.Namespace, train: TeethDataset) -> torch.Tensor:
    """Class frequencies over (up to) 300 training scans, cached in the run folder."""
    path = a.out / "class_weights.json"
    if path.exists():
        freq = torch.tensor(json.loads(path.read_text()), dtype=torch.float64)
    else:
        cnt = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        for n in train.names[:300]:
            cls_ = train.load(n)[0].classes
            cnt += torch.bincount(cls_[cls_ < NUM_CLASSES], minlength=NUM_CLASSES).double()
        freq = cnt / cnt.sum()
        path.write_text(json.dumps(freq.tolist()))
    return class_weights(freq)


# ------------------------------------------------------------- one step
def loss_input(a: argparse.Namespace, batch: TrainingBatch, out: dict, geo, dev: torch.device,
               weights: LossWeights, cw: torch.Tensor | None) -> LossInput:
    ins = batch.instances.to(dev, non_blocking=True)
    offt = batch.offset_targets.to(dev, non_blocking=True)
    if a.stage == "semantic":
        return LossInput(ins, batch.counts, offsets=out["off"], offset_targets=offt, class_logits=out["sem"],
                         classes=batch.classes.to(dev, non_blocking=True), class_weights=cw, weights=weights)
    return LossInput(ins, batch.counts, cls_logits=out["cls"], mask_logits=out["mask"],
                     aux_mask_logits=out["aux"], coarse_inverse=coarse_inverse(geo.levels, KV_LEVEL),
                     pad_index=geo.pad_index, pad_valid=geo.pad_valid, position_logits=out["sem"],
                     arch_positions=batch.arch_positions.to(dev, non_blocking=True), offsets=out["off"],
                     offset_targets=offt, weights=weights)


@torch.no_grad()
def decode(out: dict, counts: list[int], score_thresh: float = 0.5) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Queries -> (instance id per point, query scores) per scan; id i is query i - 1."""
    prob = out["cls"].float().softmax(-1)[..., 0]                # [B,K] P(tooth)
    res = []
    for b, m in enumerate(torch.split(out["mask"].float().sigmoid(), counts)):
        conf = m * prob[b][None]
        best, arg = conf.max(1)
        res.append((torch.where(best > score_thresh, arg + 1, 0), prob[b]))
    return res


@torch.no_grad()
def validate(a: argparse.Namespace, net: ToothSegNet, loaders: dict[str, DataLoader],
             dev: torch.device) -> dict[str, object]:
    net.eval()
    ev = Evaluation()
    ev.set_strategy(EvaluationType.INSTANCE_GPU if a.stage == "instance" else EvaluationType.SEMANTIC_GPU)
    scored: dict[str, object] = {}
    for tag, loader in loaders.items():
        ap_meter, cm_meter, tsas, off_err, off_n = InstanceAPMeter(), ConfusionMeter(NUM_CLASSES), [], 0.0, 0
        for batch in loader:
            geo = batch.geometry.to(dev)
            with torch.autocast("cuda", torch.bfloat16):
                out = net(batch.features.to(dev, non_blocking=True), geo)
            gt_ins = torch.split(batch.instances.to(dev), batch.counts)
            if a.stage == "instance":
                for (ids, prob), gt in zip(decode(out, batch.counts), gt_ins):
                    res = EvaluationOutput()
                    ev.execute(EvaluationInput(ids, gt, scores=prob), res)
                    ap_meter.add(res.record)
                    tsas.append(res.metrics["tsa"])
            else:
                pred = out["sem"].float().argmax(1)
                classes = batch.classes.to(dev)
                for p, c, gt in zip(torch.split(pred, batch.counts), torch.split(classes, batch.counts), gt_ins):
                    res = EvaluationOutput()
                    ev.execute(EvaluationInput(gt, gt, gt_classes=c, point_classes=p), res)
                    cm_meter.add(res.confusion)
                tooth = (classes > 0) & (classes < NUM_CLASSES)
                err = (out["off"].float()[tooth] - batch.offset_targets.to(dev)[tooth]).norm(dim=1)
                off_err, off_n = off_err + float(err.sum()), off_n + int(tooth.sum())
        if a.stage == "instance":
            scored[tag] = {**ap_meter.result(), "tsa": float(np.mean(tsas)) if tsas else 0.0}
        else:
            scored[tag] = {**cm_meter.result(), "off_err_mm": off_err / max(off_n, 1)}
    net.train()
    return scored


def selection_score(a: argparse.Namespace, scored: dict[str, dict]) -> float:
    """Instance: the mean mAP over the validation paths, so no path may rot.
    Semantic: mIoU."""
    if a.stage == "instance":
        return float(np.mean([s["mAP"] for s in scored.values()]))
    return float(scored["grey"]["miou"])


def describe(a: argparse.Namespace, scored: dict[str, dict]) -> str:
    if a.stage == "semantic":
        r = scored["grey"]
        return (f"VAL mIoU {r['miou'] * 100:.2f}  acc {r['acc'] * 100:.2f}  TSA {r['tsa'] * 100:.2f}  "
                f"offErr {r['off_err_mm']:.2f} mm")
    return "  |  ".join(f"{tag.upper()} mAP {r['mAP'] * 100:.2f}  AP50 {r['AP50'] * 100:.2f}  TSA {r['tsa'] * 100:.2f}"
                        for tag, r in scored.items())


# ------------------------------------------------------------------ main
def _plain(v: object) -> object:
    """argparse 값 -> 체크포인트에 넣을 값 (Path는 문자열로)."""
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def main() -> None:
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    log = open(a.out / "train.log", "a")

    def say(s: str) -> None:
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    train_names, val_names = split_names(a)
    dl_train, dl_vals = make_loaders(a, train_names, val_names)
    say(f"=== {a.stage} -> {a.out} | train {len(dl_train.dataset)} / val {len(dl_vals['grey'].dataset)} scans "
        f"| {a.res * 1000:.0f} um | curve {a.curve} | in_ch {a.in_ch} | batch {a.batch}x{a.accum} "
        f"| {a.epochs} epochs | rot {a.rot_mode} crop {a.crop_p} trim {a.trim_p} colour {a.colour_p} ===")
    say(f"    data {a.data}" + (f" | features {' '.join(a.features)}" if a.data == "dental_ai" else "")
        + f" | caches {' '.join(str(c) for c in a.cache)}")
    if len(dl_train) == 0:
        raise SystemExit("no training batches: check --cache and the split (is the cache built?)")

    net = build_model(a).to(dev)
    say(f"params {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M")
    weights = LossWeights() if a.stage == "instance" else LossWeights.semantic()
    for arg, field_name in (("w_aux", "aux"), ("w_pos", "position"), ("w_off", "offset"),
                            ("w_dice", "class_dice"), ("w_dir", "direction")):
        if getattr(a, arg) is not None:
            setattr(weights, field_name, getattr(a, arg))
    cw = None
    if a.stage == "semantic":
        cw = semantic_class_weights(a, dl_train.dataset).to(dev)
        say(f"class weights {[round(float(x), 2) for x in cw]}")
    loss = Loss()
    loss.set_strategy(LossType.SEMANTIC if a.stage == "semantic" else LossType.MASK_TRANSFORMER)

    decay = [p for p in net.parameters() if p.ndim > 1]
    no_decay = [p for p in net.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    steps_per_epoch = max(1, len(dl_train) // a.accum)
    total = a.epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        wu = a.warmup * steps_per_epoch
        if step < wu:
            return a.lr * (step + 1) / wu
        t = (step - wu) / max(1, total - wu)
        return 0.02 * a.lr + 0.98 * a.lr * 0.5 * (1 + math.cos(math.pi * t))

    start_ep, gstep, best = 0, 0, -1.0
    if a.init_from:
        init_from(net, a.init_from, say)
    if a.resume and a.resume.exists():
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep, gstep, best = ck["epoch"] + 1, ck["gstep"], ck["best"]
        say(f"resumed from {a.resume} @ epoch {start_ep}")

    net.train()
    for ep in range(start_ep, a.epochs):
        t0, sums, nb = time.time(), {}, 0
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(dl_train):
            geo = batch.geometry.to(dev)
            with torch.autocast("cuda", torch.bfloat16):
                out = net(batch.features.to(dev, non_blocking=True), geo)
            res = LossOutput()
            loss.execute(loss_input(a, batch, out, geo, dev, weights, cw), res)
            (res.total / a.accum).backward()
            if (it + 1) % a.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(gstep)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
            for k, v in res.terms.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            nb += 1
        terms = " ".join(f"{k} {v / max(nb, 1):.4f}" for k, v in sums.items())
        say(f"ep {ep:3d} | {terms} | lr {lr_at(gstep):.2e} | {time.time() - t0:.0f}s")

        if ep % a.val_every == a.val_every - 1 or ep == a.epochs - 1:
            scored = validate(a, net, dl_vals, dev)
            score = selection_score(a, scored)
            say(f"      {describe(a, scored)}")
            args = {k: _plain(v) for k, v in vars(a).items()}
            ck = dict(model=net.state_dict(), opt=opt.state_dict(), epoch=ep, gstep=gstep, best=best,
                      args=args, val=scored, stage=a.stage, curve=a.curve, in_ch=a.in_ch,
                      features=list(a.features) if a.data == "dental_ai" else None)
            if a.stage == "instance":
                ck.update(queries=a.queries, kv_level=KV_LEVEL, head_dim=a.head_dim, head_layers=a.head_layers)
            torch.save(ck, a.out / "last.pt")
            if score > best:
                best = ck["best"] = score
                torch.save(ck, a.out / "best.pt")
                say(f"      new best {'mAP' if a.stage == 'instance' else 'mIoU'} {best * 100:.2f}")
    say("done")


if __name__ == "__main__":
    main()
