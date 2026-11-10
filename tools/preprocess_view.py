"""What preprocessing does to one scan, step by step, as pictures.

Three figures per scan, in --out:

* <scan>_cache.png    what `training/preprocess.py` stored: the 17 classes,
                      the cached normals, and the per-class point shares the
                      semantic stage's CE weights come from;
* <scan>_augment.png  each augmentation of TRAIN_RECIPE on its own, all from
                      the same scan, then two draws of the whole recipe;
* <scan>_input.png    what the network reads: the voxel hierarchy L0..L4, the
                      serialisation curve, the attention patches, the query
                      seeds and the feature channels.

Everything drawn comes out of the code training runs -- `TeethDataset`, the
augmentation strategies, `train.py`'s stage defaults -- so the pictures show
what the network gets, not a re-implementation of it.  Every view looks onto
the occlusal plane (xy in Teeth3DS) from the side the crowns face; grey is
gingiva and every tooth carries its FDI number.

    python tools/preprocess_view.py                                  # first cached scan, semantic stage
    python tools/preprocess_view.py --name 01328DDN_lower --stage instance --seed 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithm.augmentation.augmentation import (Augmentation, AugmentationInput, AugmentationOutput,  # noqa: E402
                                                 AugmentationType)
from algorithm.augmentation.common.params import AugmentationParams  # noqa: E402
from algorithm.augmentation.common.sample import AugmentationSample  # noqa: E402
from algorithm.loss.common.plot_style import INK, INK2, MUTED, SURFACE, style  # noqa: E402
from algorithm.model_input.common.builder import CurveKey  # noqa: E402
from algorithm.voxel_downsample.voxel_downsample import (VoxelDownsample, VoxelDownsampleInput,  # noqa: E402
                                                         VoxelDownsampleOutput, VoxelDownsampleType)
from layer.checkpoint import KV_LEVEL, PATCH  # noqa: E402
from training.dataset import DatasetConfig, TeethDataset, _name_generator  # noqa: E402
from training.train import DATA, STAGE_DEFAULTS  # noqa: E402

# The reference categorical slots in their validated order.  Along the arch the
# teeth run 18..11 | 21..28: quadrant 1 takes the slots by position, quadrant 2
# takes them reversed, so any two neighbouring teeth -- the central incisors
# included -- are two adjacent slots, the pairs the order is validated for
# (worst CVD dE 9.1, normal vision 19.6; the incisor pair 32.3).
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GINGIVA = "#d6d5ce"          # recedes, and stays >= 18.7 dE from every slot
GHOST = "#efeee9"            # what a crop or a trim removed
UNKNOWN = "#898781"          # class 17, a tooth of unknown position
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
ZOOM_MM = 7.0                # half-width of the hierarchy close-ups
FEATURES = ("x / 35 mm", "y / 35 mm", "z / 35 mm", "normal x", "normal y", "normal z", "jaw upper",
            "red", "green", "blue")


def _rgb(h: str) -> tuple[float, float, float]:
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


CLASS_RGB = np.array([_rgb(h) for h in (GINGIVA, *SLOTS, *SLOTS[::-1], UNKNOWN)])     # [18,3]


def fdi(c: int, upper: bool) -> str:
    """Class 0..17 -> FDI tooth number (`preprocess.fdi_to_class` backwards)."""
    if c == 0:
        return "gingiva"
    if c > 16:
        return "?"
    quadrant = (1 if c <= 8 else 2) + (0 if upper else 2)
    return f"{quadrant}{(c - 1) % 8 + 1}"


class View:
    """Onto the occlusal plane (xy), from the side the crowns face.

    The mean normal of the crowns points out of their occlusal surfaces.  When
    that is -z the view is from below, and x flips so the picture is not
    mirrored.  One view serves a whole figure, so a rotation shows as one.
    """

    def __init__(self, normals: np.ndarray, tooth: np.ndarray) -> None:
        self.sign = 1.0 if normals[tooth, 2].mean() >= 0 else -1.0

    def project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[n,3] -> [n,2] picture coordinates and [n] depth towards the viewer."""
        return np.stack([self.sign * p[:, 0], p[:, 1]], 1), self.sign * p[:, 2]

    def light(self, normals: np.ndarray) -> np.ndarray:
        """Headlight: 1 facing the viewer, 0 side-on or facing away."""
        return np.clip(self.sign * normals[:, 2], 0.0, 1.0)


# ------------------------------------------------------------------ drawing
def panel(ax: Any, title: str, sub: str = "") -> None:
    ax.set_axis_off()
    ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="bold", pad=17)
    ax.text(0.0, 1.015, sub, transform=ax.transAxes, color=MUTED, fontsize=7.5, va="bottom")


def frame(ax: Any, xy: np.ndarray, pad: float = 0.05) -> None:
    """Square limits around `xy`, so every panel draws at one aspect."""
    lo, hi = xy.min(0), xy.max(0)
    c, half = (lo + hi) / 2, float((hi - lo).max()) / 2 * (1 + pad)
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal", adjustable="box")


def scale_bar(ax: Any, mm: float) -> None:
    from matplotlib import patheffects
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    x, y = x0 + 0.04 * (x1 - x0), y0 + 0.03 * (y1 - y0)
    halo = [patheffects.withStroke(linewidth=2.4, foreground=SURFACE)]
    ax.plot([x, x + mm], [y, y], color=INK2, lw=2, solid_capstyle="butt", zorder=6)
    ax.text(x + mm / 2, y + 0.015 * (y1 - y0), f"{mm:g} mm", ha="center", va="bottom", color=INK2, fontsize=7,
            path_effects=halo, zorder=6)


def cloud(ax: Any, view: View, p: np.ndarray, rgb: Any, normals: np.ndarray | None = None,
          size: float = 0.5, shade: float = 0.25, limit: int = 150000) -> np.ndarray:
    """Scatter the points far to near, lit by the headlight when there are
    normals.  Returns the projected points, for `frame`."""
    xy, depth = view.project(p)
    rgb = np.broadcast_to(np.asarray(rgb, np.float64), (len(p), 3))
    if normals is not None:
        rgb = rgb * (1 - shade + shade * view.light(normals))[:, None]
    idx = np.arange(len(p))
    if len(idx) > limit:
        idx = np.random.default_rng(0).choice(len(p), limit, replace=False)
    idx = idx[np.argsort(depth[idx], kind="stable")]
    ax.scatter(xy[idx, 0], xy[idx, 1], c=rgb[idx], s=size, linewidths=0, rasterized=True)
    return xy


def tooth_labels(ax: Any, view: View, p: np.ndarray, cls: np.ndarray, ins: np.ndarray, upper: bool) -> None:
    """The FDI number on every tooth: three slots sit under 3:1 on the surface,
    and the number carries the identity the colour alone may not."""
    from matplotlib import patheffects
    halo = [patheffects.withStroke(linewidth=2.4, foreground=SURFACE)]
    for i in np.unique(ins[ins > 0]):
        m = ins == i
        xy, _ = view.project(p[m].mean(0, keepdims=True))
        ax.text(xy[0, 0], xy[0, 1], fdi(int(np.bincount(cls[m]).argmax()), upper), ha="center", va="center",
                color=INK, fontsize=6.5, fontweight="bold", path_effects=halo, zorder=5)


def labelled(ax: Any, view: View, s: AugmentationSample, upper: bool, size: float,
             colour: bool = False) -> np.ndarray:
    """A scan in its class colours (or its vertex colour), lit, numbered."""
    p, n = s.positions.numpy(), s.normals.numpy()
    cls, ins = s.classes.numpy(), s.instances.numpy()
    rgb = s.colour.numpy() if colour and s.colour is not None else CLASS_RGB[cls]
    xy = cloud(ax, view, p, rgb, n, size=size)
    tooth_labels(ax, view, p, cls, ins, upper)
    return xy


def save(fig: Any, path: Path, top: float) -> None:
    import matplotlib.pyplot as plt
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=2.0, w_pad=1.5)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {path}")


def header(fig: Any, title: str, sub: str) -> None:
    fig.text(0.012, 0.985, title, ha="left", va="top", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.012, 0.955, sub, ha="left", va="top", color=INK2, fontsize=8.5)


# ------------------------------------------------------------- 1: the cache
def figure_cache(name: str, s: AugmentationSample, upper: bool, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    p, n = s.positions.numpy(), s.normals.numpy()
    cls, ins = s.classes.numpy(), s.instances.numpy()
    view = View(n, cls > 0)
    teeth = np.unique(ins[ins > 0])
    mixed = [int(i) for i in teeth if np.unique(cls[ins == i]).size > 1]      # one tooth should be one class

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=(1, 1, 0.8)))
    header(fig, f"{name}  ·  {'upper' if upper else 'lower'} jaw  ·  {len(p):,} vertices  ·  {len(teeth)} teeth"
                "  —  what training/preprocess.py cached",
           "xyz float32 mm  ·  nrm float16, area-weighted from the faces  ·  cls 0..16 from FDI  ·  "
           "ins 1..K, ordered by class  ·  jaw" + (f"   ·   instances with mixed classes: {mixed}" if mixed else
                                                   "   ·   every instance is one class"))

    ax = axes[0]
    q1, q2 = (1, 2) if upper else (3, 4)
    panel(ax, "classes  (cls)", f"0 gingiva  ·  1–8 quadrant {q1}  ·  9–16 quadrant {q2}")
    frame(ax, labelled(ax, view, s, upper, size=1.6))
    scale_bar(ax, 10)

    ax = axes[1]
    panel(ax, "normals  (nrm)", "headlight shading: bright faces the viewer, dark faces away")
    frame(ax, cloud(ax, view, p, 0.93, n, size=1.6, shade=0.8))
    scale_bar(ax, 10)

    ax = axes[2]
    k = 18 if (cls == 17).any() else 17
    share = np.bincount(cls, minlength=k)[:k] / len(cls)
    rows = np.arange(k)
    style(ax, "points per class", "share of the scan's vertices", "")
    ax.barh(rows, share * 100, height=0.72, color=CLASS_RGB[:k])
    ax.set_yticks(rows, [fdi(int(c), upper) for c in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlim(0, share.max() * 100 * 1.18)
    for y, v in zip(rows, share):
        ax.text(v * 100 + share.max() * 1.5, y, f"{v:.1%}", va="center", color=INK2, fontsize=7)
    ax.text(0.98, 0.04, "the semantic stage weights its CE by\n1/√share, measured over 300 scans",
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=7.5)
    save(fig, out / f"{name}_cache.png", top=0.93)


# -------------------------------------------------------- 2: augmentations
def augmented(s: AugmentationSample, typ: AugmentationType, params: AugmentationParams,
              seed: int) -> tuple[AugmentationSample, bool]:
    aug, res = Augmentation(), AugmentationOutput()
    aug.set_strategy(typ)
    aug.execute(AugmentationInput(s, params, torch.Generator().manual_seed(seed)), res)
    return res.sample, res.applied


def figure_augment(name: str, s: AugmentationSample, upper: bool, stage: str, seed: int, out: Path) -> None:
    import matplotlib.pyplot as plt

    d = STAGE_DEFAULTS[stage]
    recipe = AugmentationParams(crop_p=d["crop_p"], trim_p=d["trim_p"], colour_p=d["colour_p"])
    T, P = AugmentationType, AugmentationParams
    # (title, strategy, params, note, draw what it removed)
    steps = [
        ("scan as cached", None, None, "the input of every step below", False),
        ("mirror", T.MIRROR_CPU, P(),
         f"x → −x: every tooth changes side and quadrant ({fdi(6, upper)} → {fdi(14, upper)})", False),
        ("gingiva trim · 4.6 mm", T.GINGIVA_TRIM_CPU, P(trim_reach_mm=(4.6, 4.6)),
         "the in-house scanner's reach; training draws 2.5–14 mm", True),
        ("synthetic colour", T.SYNTH_COLOUR_CPU, P(), "painted on colourless scans for the RGB channels", False),
        ("crop · bridge", T.SEGMENT_CROP_CPU, P(crop_mode="bridge"), "2–5 neighbouring teeth, 1–3 mm collar", True),
        ("crop · quadrant", T.SEGMENT_CROP_CPU, P(crop_mode="quadrant"), "a quarter to half of the teeth", True),
        ("crop · half", T.SEGMENT_CROP_CPU, P(crop_mode="half"), "45–60 % of the teeth", True),
        ("crop · wedge", T.WEDGE_CROP_CPU, P(), "an angular slice of the arch; it cuts through teeth", True),
        ("rotation · SO(3)", T.RANDOM_SO3_CPU, P(), "uniform rotation, ±10 % scale, 0.05 mm jitter", False),
        ("rotation · limited", T.SMALL_ROTATION_CPU, P(), "±10° tilt, ±25° turn  (--rot-mode limited)", False),
        (f"{stage} recipe · draw 1", T.TRAIN_RECIPE_CPU, recipe, None, False),
        (f"{stage} recipe · draw 2", T.TRAIN_RECIPE_CPU, recipe, None, False),
    ]
    view = View(s.normals.numpy(), s.classes.numpy() > 0)
    fig, axes = plt.subplots(3, 4, figsize=(17, 14.2), facecolor=SURFACE)
    header(fig, f"{name}  —  the augmentations, one at a time (seed {seed})",
           "TRAIN_RECIPE runs them in this order: mirror → crop → gingiva trim → rotation → colour, before "
           f"downsampling.  {stage} stage: mirror 0.5 · crop {d['crop_p']} · trim {d['trim_p']} · "
           f"colour {d['colour_p']}.  Pale: what a crop or trim removed.")
    for k, (ax, (title, typ, params, note, ghost)) in enumerate(zip(axes.ravel(), steps)):
        r, applied = (s, True) if typ is None else augmented(s, typ, params, seed * 100 + k)
        if note is None:
            note = f"colour {'painted' if r.colour is not None else 'off'}"
        sub = f"{note}  ·  {r.n:,} vertices" + ("" if applied else "  ·  declined, unchanged")
        panel(ax, title, sub)
        xy = cloud(ax, view, s.positions.numpy(), _rgb(GHOST), size=0.9) if ghost else None
        drawn = labelled(ax, view, r, upper, size=0.9, colour=typ is T.SYNTH_COLOUR_CPU)
        frame(ax, drawn if xy is None else xy)
        scale_bar(ax, 10)
    save(fig, out / f"{name}_augment.png", top=0.935)


# ------------------------------------------------------ 3: network input
def raw_in_item_frame(s: AugmentationSample, name: str, cfg: DatasetConfig, l0: torch.Tensor) -> np.ndarray:
    """The cached vertices, shifted the way `TeethDataset` centres its L0.

    With train=False the dataset only downsamples, thins with the scan's own
    generator and centres, so redoing those two draws gives the shift -- and
    the check below says the result is the dataset's L0, point for point.
    """
    down, ds = VoxelDownsample(), VoxelDownsampleOutput()
    down.set_strategy(VoxelDownsampleType.GRID_CPU)
    down.execute(VoxelDownsampleInput(s.positions, cfg.voxel_mm), ds)
    keep = torch.arange(ds.representative.numel())
    if cfg.max_points and keep.numel() > cfg.max_points:
        keep = torch.randperm(keep.numel(), generator=_name_generator(name))[:cfg.max_points]
    picked = s.positions[ds.representative[keep]]
    shift = picked.double().mean(0).float()
    if not torch.equal(picked - shift, l0):
        raise RuntimeError("could not line the cached vertices up with TeethDataset's L0")
    return (s.positions - shift).numpy()


def figure_input(name: str, ds: TeethDataset, s: AugmentationSample, upper: bool, stage: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cfg = ds.config
    item = ds[0]
    geo, levels = item.geometry, item.geometry.levels
    feats = item.features.numpy()
    p0, n0 = levels[0].coord.numpy(), feats[:, 3:6]
    cls0, ins0 = item.classes.numpy(), item.instances.numpy()
    raw = raw_in_item_frame(s, name, cfg, levels[0].coord)
    view = View(n0, cls0 > 0)

    # the close-ups sit on a first molar if there is one: the biggest crown
    target = next((c for c in (6, 14) if (cls0 == c).any()), int(np.bincount(cls0[cls0 > 0]).argmax()))
    centre = view.project(p0[cls0 == target].mean(0, keepdims=True))[0][0]
    window = np.array([centre - ZOOM_MM, centre + ZOOM_MM])

    def inside(xy: np.ndarray) -> np.ndarray:
        return ((xy >= window[0]) & (xy <= window[1])).all(1)

    def zoom(ax: Any) -> None:
        frame(ax, window, pad=0.0)
        scale_bar(ax, 2)

    n_cols, fig_w = 6, 19.0
    ppm = fig_w / n_cols * 0.78 * 72 / (2 * ZOOM_MM)                     # points per mm in a close-up
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_w, 8.4), facecolor=SURFACE)
    chain = "  →  ".join(f"L{i} {lv.n:,}" for i, lv in enumerate(levels))
    header(fig, f"{name}  ·  {stage} stage  ·  {cfg.voxel_mm * 1000:.0f} µm voxels  ·  in_ch {cfg.in_ch}  ·  "
                f"curve {cfg.curve.name.lower()}  —  what the network reads",
           f"TeethDataset (train=False): downsample → thin to {cfg.max_points:,} → centre → features + geometry.  "
           f"{s.n:,} vertices  →  {chain} points.  Training augments first (see _augment.png).")

    # row 1: the voxel hierarchy over one tooth.  Coarse points from every
    # depth would pile up in a top view, so a level draws only the points whose
    # pooled L0 points are mostly that tooth -- the rule `coarse_targets` pools
    # the instance masks by.
    on_target = (cls0 == target).astype(np.float64)
    m0 = inside(view.project(p0)[0])
    ax = axes[0, 0]
    panel(ax, "cached vertices", f"close-up on tooth {fdi(target, upper)}  ·  every vertex")
    m = inside(view.project(raw)[0])
    cloud(ax, view, raw[m], CLASS_RGB[s.classes.numpy()[m]], s.normals.numpy()[m], size=(0.5 * 0.2 * ppm) ** 2)
    zoom(ax)
    to_level = np.arange(levels[0].n)                                  # L0 point -> its point on this level
    for lvl in range(len(levels)):
        ax = axes[0, lvl + 1]
        cell = cfg.voxel_mm * 2 ** lvl
        size = (0.5 * cell * ppm) ** 2
        if lvl == 0:
            panel(ax, f"L0 · {cell:.2f} mm", f"one vertex per voxel  ·  {levels[0].n:,} points")
            cloud(ax, view, p0[m0], CLASS_RGB[cls0[m0]], n0[m0], size=size)
        else:
            to_level = levels[lvl].inv.numpy()[to_level]
            n = levels[lvl].n
            tooth = np.bincount(to_level, on_target, n) / np.bincount(to_level, minlength=n).clip(1) > 0.5
            panel(ax, f"L{lvl} · {cell:.2f} mm",
                  f"voxel centroids  ·  {n:,} points  ·  {int(tooth.sum())} over {fdi(target, upper)}")
            cloud(ax, view, p0[m0], _rgb(GINGIVA), size=(0.5 * cfg.voxel_mm * ppm) ** 2)
            cloud(ax, view, levels[lvl].coord.numpy()[tooth], CLASS_RGB[target], size=size)
        zoom(ax)

    # row 2: order, patches, seeds, features
    ax = axes[1, 0]
    order = geo.layouts[(0, 0)].inv_sel.numpy()                        # point -> slot along the curve
    ramp = LinearSegmentedColormap.from_list("order", BLUE_RAMP)
    panel(ax, "curve order · L0", f"{cfg.curve.name.lower()} curve, light → dark; a patch is a run of {PATCH}")
    frame(ax, cloud(ax, view, p0, ramp(order / order.max())[:, :3], size=0.6))
    ax.add_patch(Rectangle(window[0], 2 * ZOOM_MM, 2 * ZOOM_MM, fill=False, ec=INK, lw=1))
    scale_bar(ax, 10)

    # three anchor points on teeth spread along the arch; each panel colours the
    # patch that holds each anchor, so the panels compare the same three points
    teeth = np.unique(ins0[ins0 > 0])
    picks = teeth[[len(teeth) // 6, len(teeth) // 2, 5 * len(teeth) // 6]]
    anchors0 = np.array([int(np.argmin(((p0 - p0[ins0 == t].mean(0)) ** 2).sum(1))) for t in picks])
    anchors1 = levels[1].inv.numpy()[anchors0]
    keys = [k for k in ((0, 0), (1, 1), (1, 3)) if k in geo.layouts]
    for ax, (lvl, var) in zip(axes[1, 1:4], keys):
        lay, p = geo.layouts[(lvl, var)], levels[lvl].coord.numpy()
        sel, inv_sel = lay.sel.numpy(), lay.inv_sel.numpy()
        panel(ax, f"attention patches · L{lvl} · variant {var}",
              f"{sel.size // PATCH} patches  ·  the 3 that hold the same 3 points")
        frame(ax, cloud(ax, view, p, _rgb(GINGIVA), size=0.5 if lvl == 0 else 1.5))
        for slot, a in zip(SLOTS[:3], anchors0 if lvl == 0 else anchors1):
            first = inv_sel[a] // PATCH * PATCH
            members = np.unique(sel[first:first + PATCH])
            cloud(ax, view, p[members], _rgb(slot), size=1.2 if lvl == 0 else 3.5)
        scale_bar(ax, 10)
    for ax in axes[1, 1 + len(keys):4]:
        ax.set_axis_off()

    ax = axes[1, 4]
    p = levels[KV_LEVEL].coord.numpy()
    seeds = geo.seed[0].numpy()
    panel(ax, f"query seeds · L{KV_LEVEL}", f"{seeds.size} farthest-point seeds: where the queries start"
          if seeds.size else "the semantic stage has no queries (n_query 0)")
    frame(ax, cloud(ax, view, p, _rgb(GINGIVA), size=1.5))
    if seeds.size:
        xy = view.project(p[seeds])[0]
        ax.scatter(xy[:, 0], xy[:, 1], s=42, c=INK, edgecolors=SURFACE, linewidths=1.2, zorder=4)
    scale_bar(ax, 10)

    ax = axes[1, 5]
    panel(ax, f"features  [{feats.shape[0]:,} × {feats.shape[1]}]", "per channel, over the L0 points")
    rows = [f"{'channel':<11}{'min':>7}{'max':>7}{'mean':>7}"]
    rows += [f"{FEATURES[j]:<11}{feats[:, j].min():+7.2f}{feats[:, j].max():+7.2f}{feats[:, j].mean():+7.2f}"
             for j in range(feats.shape[1])]
    if feats.shape[1] > 7 and not feats[:, 7:].any():
        rows.append("rgb all 0: colourless scans standardise to zero")
    ax.text(0.0, 0.96, "\n".join(rows), transform=ax.transAxes, va="top", ha="left", family="monospace",
            fontsize=8, color=INK2, linespacing=1.6)
    print("\n".join(rows))
    save(fig, out / f"{name}_input.png", top=0.92)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=DATA / "cache")
    ap.add_argument("--name", default=None, help="scan to draw, e.g. 01328DDN_lower (default: the first cached)")
    ap.add_argument("--stage", choices=list(STAGE_DEFAULTS), default="semantic",
                    help="whose voxel size, point cap, input channels and recipe probabilities")
    ap.add_argument("--curve", choices=["legacy", "morton3d"], default="morton3d")
    ap.add_argument("--seed", type=int, default=0, help="for the augmentation draws")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/preprocess_view")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    name = a.name or sorted(a.cache.glob("*.npz"))[0].stem
    d = STAGE_DEFAULTS[a.stage]
    cfg = DatasetConfig(voxel_mm=d["res"], in_ch=d["in_ch"], n_query=32 if a.stage == "instance" else 0,
                        curve=CurveKey[a.curve.upper()], max_points=d["max_points"], train=False)
    ds = TeethDataset([a.cache], [name], cfg)
    if not ds.names:
        raise SystemExit(f"{name}.npz is not in {a.cache} (run training/preprocess.py first)")
    sample, upper = ds.load(name)
    a.out.mkdir(parents=True, exist_ok=True)
    figure_cache(name, sample, upper, a.out)
    figure_augment(name, sample, upper, a.stage, a.seed, a.out)
    figure_input(name, ds, sample, upper, a.stage, a.out)


if __name__ == "__main__":
    main()
