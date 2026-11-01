"""Do the paths of the Gaussians follow the motion? (PROGRESS.md, plans 2e and 2f.)

For every Gaussian on the moving object at its peak time t0, its path over the two clip frames
around t0 is compared with the true path of the point it sits on:

    arc        on the arc clips (``data.arc_clip``) the ball's path is known: the point at offset o
               from the ball's centre c(t) moves to c(t) + o.
    flow       on UVG clips the point is followed through the optical flow between consecutive
               frames of the sequence (stride 1), so also through the frames the fit never saw.

Per Gaussian: the mean distance between its centre and the true point at the clip times k - 1 + j/3,
j = 1 .. 6, starting where they coincide at k - 1 (k = t0 rounded); and for models with acceleration
the angle between the fitted 2a and the true acceleration, the second difference of the true path
over k - 1, k, k + 1. Only Gaussians with sigma_t >= 0.5, visible over that window, are kept, and
Gaussians off the object are the control.

    python -m stg.paths arc
    python -m stg.paths flow
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from stg import data, flow
from stg.model import SpaceTimeGaussians
from stg.slicing import POLY_TIME

STEPS = np.arange(1, 7) / 3          # clip times after k - 1


def centres(g: SpaceTimeGaussians, t: np.ndarray) -> np.ndarray:
    """(n, 2): the centre of each Gaussian at its own clip time t (n,)."""
    tau = t - g.t0
    mu = g.mu + g.vel * tau[:, None]
    if g.spec.accel:
        mu = mu + g.acc * (tau ** 2)[:, None]
    for j in range(g.spec.extra_powers):
        mu = mu + g.poly[:, j] * (tau ** 2 * (tau / POLY_TIME) ** (j + 1))[:, None]
    return mu


def chosen(g: SpaceTimeGaussians, n_frames: int) -> np.ndarray:
    """Indices of the Gaussians that are visible over k - 1 .. k + 1 inside the clip."""
    k = np.round(g.t0)
    return np.nonzero((g.sigma_t >= 0.5) & (k >= 1) & (k <= n_frames - 2))[0]


def summary(g, inside, start, true_paths, true_acc):
    """Statistics of the chosen Gaussians ``g`` given the true paths (n, len(STEPS), 2) of the points
    they start on at k - 1 and the true accelerations (n, 2), in clip frames."""
    fitted = np.stack([centres(g, start + s) for s in STEPS], 1)
    error = np.linalg.norm(fitted - true_paths, axis=-1).mean(1)
    out = {}
    for name, sel in (("object", inside), ("background", ~inside)):
        e = {"gaussians": int(sel.sum())}
        if sel.any():
            e["path_error_median"] = float(np.median(error[sel]))
            e["true_acc_median"] = float(np.median(np.linalg.norm(true_acc[sel], axis=1)))
            if g.spec.accel:
                fit = 2 * g.acc[sel]
                tru = true_acc[sel]
                big = (np.linalg.norm(fit, axis=1) > 0.1) & (np.linalg.norm(tru, axis=1) > 0.1)
                cos = np.sum(fit * tru, 1) / (np.linalg.norm(fit, axis=1) * np.linalg.norm(tru, axis=1) + 1e-9)
                e["fitted_acc_median"] = float(np.median(np.linalg.norm(fit, axis=1)))
                e["share_same_direction"] = float(np.mean(cos[big] > 0)) if big.any() else None
                e["cosine_median"] = float(np.median(cos[big])) if big.any() else None
        out[name] = e
    return out


def _sub(g, idx):
    return SpaceTimeGaussians(g.spec, **{n: a[idx] for n, a in g.params.items()})


def arc_one(folder):
    m = json.loads((folder / "metrics.json").read_text())
    (style, angle), stride = data.arc_style(m["sequence"]), m["stride"]
    g = SpaceTimeGaussians.load(folder / "params.npz")
    idx = chosen(g, m["frames"])
    sub = _sub(g, idx)
    start = np.round(sub.t0) - 1
    ball = lambda t: data.arc_centres(angle, stride * t, data.SIZE, style["speed"])   # clip time -> sequence frame
    radius = style["radius"]
    inside = np.linalg.norm(centres(sub, sub.t0) - ball(sub.t0), axis=1) < radius
    # a point on the ball moves with it, a point of the background stays
    moves = inside[:, None].astype(float)
    p0 = centres(sub, start)
    true_paths = np.stack([p0 + moves * (ball(start + s) - ball(start)) for s in STEPS], 1)
    true_acc = moves * (ball(start) - 2 * ball(start + 1) + ball(start + 2))
    return m, summary(sub, inside, start, true_paths, true_acc)


class Tracks:
    """The optical flow between consecutive frames of a sequence, and its object masks."""

    def __init__(self, sequence, n_frames, data_root=Path("data/uvg"), mask_root=Path("data/masks")):
        frames = data.load_clip(data_root, sequence, 0, n_frames, 1)
        self.flows = flow.flows(frames)
        self.masks = np.stack([np.asarray(Image.open(mask_root / sequence / f"{k:04d}.png").convert("L")) > 127
                               for k in range(n_frames)])

    def follow(self, points, frame, steps):
        """(n, steps + 1, 2): the points from sequence frame ``frame`` (n,) through ``steps`` flows."""
        import cv2
        height, width = self.flows.shape[1:3]
        out = [points]
        p = points.astype(np.float32)
        for j in range(steps):
            f = self.flows[frame + j]
            d = np.empty_like(p)
            for c in range(2):          # bilinear lookup of the flow at the points, frame by frame
                d[:, c] = cv2.remap(f[..., c], np.ascontiguousarray(p[:, 0:1]), np.ascontiguousarray(p[:, 1:2]), cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)[:, 0]
            p = p + d
            out.append(p.copy())
        return np.stack(out, 1)

    def inside(self, points, frame):
        height, width = self.masks.shape[1:]
        x = np.clip(points[:, 0].astype(int), 0, width - 1)
        y = np.clip(points[:, 1].astype(int), 0, height - 1)
        return self.masks[frame, y, x]


def flow_one(folder, tracks):
    m = json.loads((folder / "metrics.json").read_text())
    stride = m["stride"]
    g = SpaceTimeGaussians.load(folder / "params.npz")
    idx = chosen(g, m["frames"])
    sub = _sub(g, idx)
    start = np.round(sub.t0) - 1
    p0 = centres(sub, start)
    paths = np.empty((len(idx), 2 * stride + 1, 2), np.float32)
    first = (start * stride).astype(int)
    for f in np.unique(first):          # follow the points that start in the same frame together
        sel = first == f
        paths[sel] = tracks.follow(p0[sel], f, 2 * stride)
    true_paths = paths[:, 1:]
    true_acc = paths[:, 0] - 2 * paths[:, stride] + paths[:, 2 * stride]
    inside = tracks.inside(centres(sub, start + 1), (first + stride))
    return m, summary(sub, inside, start, true_paths, true_acc)


def table(results, key, models):
    """Markdown rows: per clip and budget, the mean over seeds of each model's statistics."""
    lines = []
    for (sequence, budget), per_model in sorted(results.items()):
        row = [sequence, str(budget)]
        for model in models:
            entries = per_model.get(model, [])
            def mean(region, field):
                v = [e[region].get(field) for e in entries if e[region].get(field) is not None]
                return np.mean(v) if v else np.nan
            row.append(f"{mean('object', 'path_error_median'):.2f} / {mean('background', 'path_error_median'):.2f}")
            if key and model != "M0":
                row.append(f"{mean('object', 'share_same_direction'):.0%} / {mean('background', 'share_same_direction'):.0%}")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("kind", choices=("arc", "flow"))
    p.add_argument("--root", type=Path, help="default runs/arc (arc) or runs/long_test (flow)")
    p.add_argument("--sequences", nargs="+", default=["ShakeNDry", "Jockey"])
    p.add_argument("--models", nargs="+", default=["M0", "M1"])
    p.add_argument("--out", type=Path, default=Path("runs/paths"))
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    root = a.root or Path("runs/arc" if a.kind == "arc" else "runs/long_test")
    results = defaultdict(lambda: defaultdict(list))
    tracks = {}
    for path in sorted(root.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        if m["model"] not in a.models or m["stride"] != 3:
            continue
        if a.kind == "arc":
            if not m["sequence"].startswith("arc"):
                continue
            m, s = arc_one(path.parent)
        else:
            if m["sequence"] not in a.sequences:
                continue
            if m["sequence"] not in tracks:
                tracks[m["sequence"]] = Tracks(m["sequence"], (m["frames"] - 1) * m["stride"] + 1)
            m, s = flow_one(path.parent, tracks[m["sequence"]])
        results[(m["sequence"], m["budget"])][m["model"]].append(s)
        print(path.parent.name, json.dumps(s["object"]), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    plain = {f"{seq}_B{b}": {mo: v for mo, v in per.items()} for (seq, b), per in results.items()}
    (a.out / f"{a.kind}.json").write_text(json.dumps(plain, indent=1))
    header = ["| 片段 | 存储量 | " + " | ".join(f"{mo} 路线误差（物体 / 背景）" + ("" if mo == "M0" else f" | {mo} 加速度方向一致（物体 / 背景）")
                                         for mo in a.models) + " |"]
    header.append("|---|---|" + "---|" * (len(a.models) * 2 - a.models.count("M0")))
    text = "\n".join(header + table(results, True, a.models))
    (a.out / f"{a.kind}.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
