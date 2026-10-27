"""Diagnostics of the test runs: why shrinking helps and where acceleration is used.

1. shrink    Where M2 reduces the error of M0: the per-pixel reduction of squared error, binned by how
             much the target changes over time at that pixel (|frame t - frame t-1|), and the error
             maps of one frame.
2. accel     Where M1 spends its acceleration: per Gaussian, the shift |a| (2 sigma_t)^2 that the
             acceleration adds within |tau| <= 2 sigma_t, inside and outside the object mask, and the
             agreement of the fitted acceleration 2a with the acceleration of the optical flow at the
             Gaussian's centre (both in pixels per frame^2).
3. lifetime  Fitted durations sigma_t and the number of Gaussians visible per frame (w > 0.1), M0 vs M1.
4. worst     The worst frame of each model with error maps and a crop around the object.

Writes ``runs/diagnostics/diagnostics.json`` and PNGs.

    python -m stg.diagnose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image

from stg import data, flow
from stg.model import SpaceTimeGaussians
from stg.render import ClipRenderer

MODELS = ("M0", "M1", "M2", "M3")


def run_dir(root, sequence, stride, model, budget, seed, frames=10):
    hits = sorted(root.glob(f"{sequence}_k0_n{frames}_s{stride}_{model}_B{budget}_seed{seed}_*"))
    assert len(hits) == 1, (sequence, stride, model, budget, seed, hits)
    return hits[0]


class Clip:
    def __init__(self, sequence, stride, frames=10, device="cuda"):
        self.sequence, self.stride = sequence, stride
        self.frames = data.load_clip(Path("data/uvg"), sequence, 0, frames, stride)
        self.masks = np.stack([np.asarray(Image.open(f"data/masks/{sequence}/{k:04d}.png")) > 127
                               for k in data.frame_indices(0, frames, stride)])
        self.renderer = ClipRenderer(*self.frames.shape[:3], device)

    def render(self, folder):
        m = json.loads((folder / "metrics.json").read_text())
        g = SpaceTimeGaussians.load(folder / "params.npz")
        return g, self.renderer.render(g, m["s_min"])


def save(path, image):
    Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def error_map(image, target, scale=4.0):
    return np.repeat(np.clip(np.abs(np.clip(image, 0, 1) - target).mean(-1, keepdims=True) * scale, 0, 1), 3, -1)


# --- 1. shrinking ------------------------------------------------------------------------------

def shrink(root, out, budget, seed):
    result = {}
    edges = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0])
    for sequence in ("Jockey", "ShakeNDry", "YachtRide", "Bosphorus"):
        for stride in (1, 3):
            clip = Clip(sequence, stride)
            _, m0 = clip.render(run_dir(root, sequence, stride, "M0", budget, seed))
            _, m2 = clip.render(run_dir(root, sequence, stride, "M2", budget, seed))
            target = clip.frames
            e0 = ((np.clip(m0, 0, 1) - target) ** 2).mean(-1)[1:]
            e2 = ((np.clip(m2, 0, 1) - target) ** 2).mean(-1)[1:]
            change = np.abs(target[1:] - target[:-1]).mean(-1)
            bins = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (change >= lo) & (change < hi)
                if sel.sum() == 0:
                    continue
                bins.append({"change": [float(lo), float(hi)], "share_of_pixels": float(sel.mean()),
                             "mse_M0": float(e0[sel].mean()), "mse_M2": float(e2[sel].mean()),
                             "share_of_reduction": float((e0[sel] - e2[sel]).sum() / (e0 - e2).sum())})
            result[f"{sequence}_s{stride}"] = bins
            if stride == 1:
                t = 5
                row = np.concatenate([target[t], np.clip(m0[t], 0, 1), np.clip(m2[t], 0, 1),
                                      error_map(m0[t], target[t]), error_map(m2[t], target[t])], axis=1)
                save(out / f"shrink_{sequence}.png", row)
    return result


# --- 2. acceleration ---------------------------------------------------------------------------

def flow_acceleration(frames):
    """(T-2, H, W, 2): d_{k+1}(p + d_k(p)) - d_k(p), the acceleration of the point at p in frame k."""
    import cv2
    f = flows(frames)
    height, width = f.shape[1:3]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.stack([cv2.remap(d_next, xx + d[..., 0], yy + d[..., 1], cv2.INTER_LINEAR) - d
                     for d, d_next in zip(f[:-1], f[1:])])


flows = flow.flows


def accel(root, out, budget, seed):
    result = {}
    for sequence in ("Jockey", "ShakeNDry", "YachtRide", "Bosphorus"):
        for stride in (1, 3):
            clip = Clip(sequence, stride)
            g, _ = clip.render(run_dir(root, sequence, stride, "M1", budget, seed))
            n_frames, height, width = clip.frames.shape[:3]
            k = np.clip(np.round(g.t0).astype(int), 0, n_frames - 1)
            x = np.clip(g.mu[:, 0].astype(int), 0, width - 1)
            y = np.clip(g.mu[:, 1].astype(int), 0, height - 1)
            inside = clip.masks[k, y, x]
            shift = np.linalg.norm(g.acc, axis=1) * (2 * g.sigma_t) ** 2
            fitted = 2 * g.acc
            facc = flow_acceleration(clip.frames)
            kk = np.clip(k, 0, len(facc) - 1)
            measured = facc[kk, y, x]
            big = (np.linalg.norm(fitted, axis=1) > 0.1) & (np.linalg.norm(measured, axis=1) > 0.1)
            cos = np.sum(fitted * measured, 1) / (np.linalg.norm(fitted, axis=1) * np.linalg.norm(measured, axis=1) + 1e-9)
            entry = {"gaussians": len(g), "share_inside": float(inside.mean())}
            for name, sel in (("inside", inside), ("outside", ~inside)):
                entry[name] = {
                    "count": int(sel.sum()),
                    "shift_median": float(np.median(shift[sel])) if sel.any() else None,
                    "share_shift_below_half_pixel": float(np.mean(shift[sel] < 0.5)) if sel.any() else None,
                    "fitted_acc_median": float(np.median(np.linalg.norm(fitted[sel], axis=1))) if sel.any() else None,
                    "flow_acc_median": float(np.median(np.linalg.norm(measured[sel], axis=1))) if sel.any() else None,
                    "cosine_median": float(np.median(cos[sel & big])) if (sel & big).any() else None,
                    "share_same_direction": float(np.mean(cos[sel & big] > 0)) if (sel & big).any() else None,
                }
            # the flow's own acceleration in each region, over all pixels
            obj = clip.masks[:len(facc)]
            mag = np.linalg.norm(facc, axis=-1)
            entry["flow_acc_pixels_inside"] = float(np.nanmean(mag[obj]))
            entry["flow_acc_pixels_outside"] = float(np.nanmean(mag[~obj]))
            result[f"{sequence}_s{stride}"] = entry
    return result


# --- 3. lifetimes ------------------------------------------------------------------------------

def lifetime(root, budgets, seed):
    result = {}
    for sequence in ("Jockey", "ShakeNDry", "Bosphorus"):
        for stride in (1, 3):
            for budget in budgets:
                entry = {}
                for model in ("M0", "M1"):
                    g = SpaceTimeGaussians.load(run_dir(root, sequence, stride, model, budget, seed) / "params.npz")
                    t = np.arange(10)[:, None]
                    w = np.exp(-0.5 * ((t - g.t0[None]) / g.sigma_t[None]) ** 2)
                    entry[model] = {"gaussians": len(g), "sigma_t_median": float(np.median(g.sigma_t)),
                                    "visible_per_frame": float((w > 0.1).sum(1).mean()),
                                    "frames_visible_per_gaussian": float((w > 0.1).sum(0).mean())}
                result[f"{sequence}_s{stride}_B{budget}"] = entry
    return result


# --- 4. worst frames ---------------------------------------------------------------------------

def worst(root, out, sequence, stride, budget, seed):
    clip = Clip(sequence, stride)
    renders = {m: clip.render(run_dir(root, sequence, stride, m, budget, seed))[1] for m in MODELS}
    target = clip.frames
    ys, xs = np.nonzero(clip.masks.any(0))
    pad = 12
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, target.shape[1])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, target.shape[2])
    rows, info = [], {}
    for m in MODELS:
        err = ((np.clip(renders[m], 0, 1) - target) ** 2).mean(axis=(1, 2, 3))
        t = int(np.argmax(err))
        info[m] = {"worst_frame": t, "psnr": float(-10 * np.log10(err[t]))}
        crop = lambda im: np.asarray(Image.fromarray(np.round(np.clip(im[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8))
                                     .resize(((x1 - x0) * 2, (y1 - y0) * 2), Image.NEAREST), np.float32) / 255
        rows.append(np.concatenate([crop(target[t]), crop(renders[m][t]), crop(error_map(renders[m][t], target[t]))], 1))
    save(out / f"worst_{sequence}_s{stride}_B{budget}.png", np.concatenate(rows, 0))
    return info


# --- 5. velocity against the flow (M0) ---------------------------------------------------------

def velocity(root, budget, seed):
    """Does M0's fitted velocity follow the optical flow at the Gaussian's centre at its peak time?"""
    result = {}
    for sequence in ("Jockey", "ShakeNDry", "YachtRide", "Bosphorus"):
        for stride in (1, 3):
            clip = Clip(sequence, stride)
            g = SpaceTimeGaussians.load(run_dir(root, sequence, stride, "M0", budget, seed) / "params.npz")
            f = flows(clip.frames)
            n_frames, height, width = clip.frames.shape[:3]
            k = np.clip(np.round(g.t0).astype(int), 0, len(f) - 1)
            x = np.clip(g.mu[:, 0].astype(int), 0, width - 1)
            y = np.clip(g.mu[:, 1].astype(int), 0, height - 1)
            measured, fitted = f[k, y, x], g.vel
            err = np.linalg.norm(fitted - measured, axis=1)
            big = np.linalg.norm(measured, axis=1) > 0.5
            cos = np.sum(fitted * measured, 1) / (np.linalg.norm(fitted, axis=1) * np.linalg.norm(measured, axis=1) + 1e-9)
            result[f"{sequence}_s{stride}"] = {
                "fitted_speed_median": float(np.median(np.linalg.norm(fitted, axis=1))),
                "flow_speed_median": float(np.median(np.linalg.norm(measured, axis=1))),
                "error_median": float(np.median(err)),
                "cosine_median_where_flow_over_half_pixel": float(np.median(cos[big])) if big.any() else None,
                "share_where_flow_over_half_pixel": float(big.mean()),
            }
    return result


# --- 6. a still clip ---------------------------------------------------------------------------

def still(budgets, seed, device, iterations=3000):
    """The first frame of Jockey repeated 10 times: there is no motion at all, so any gain of M1-M3
    over M0 here cannot come from modelling motion."""
    from stg.fit import fit
    from stg.model import MODELS as SPECS, initial
    from stg.sweep import CHOSEN
    frame = data.load_clip(Path("data/uvg"), "Jockey", 0, 1, 1)[0]
    frames = np.repeat(frame[None], 10, axis=0)
    result = {}
    for budget in budgets:
        result[budget] = {}
        for model in MODELS:
            c = CHOSEN[model]
            g = initial(SPECS[model], budget, frames, seed)
            fit(g, frames, iterations, device, c["s_min"], {"vel": c["lr_vel"]}, c["acc_ratio"], log=lambda s: None)
            images = ClipRenderer(*frames.shape[:3], device).render(g, c["s_min"])
            result[budget][model] = float(-10 * np.log10(np.mean((np.clip(images, 0, 1) - frames) ** 2)))
            print(f"still B{budget} {model}: {result[budget][model]:.2f}", flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=Path("runs/test"))
    p.add_argument("--out", type=Path, default=Path("runs/diagnostics"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", choices=("velocity_still",), help="run only these and add them to diagnostics.json")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    wp.init()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.only == "velocity_still":
        path = a.out / "diagnostics.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        result["velocity_M0_B12000_seed0"] = velocity(a.root, 12000, 0)
        result["still_clip_seed0"] = still((3000, 12000), 0, a.device)
        path.write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result[k] for k in ("velocity_M0_B12000_seed0", "still_clip_seed0")}, indent=1))
        return
    result = {
        "shrink_B3000_seed0": shrink(a.root, a.out, 3000, 0),
        "accel_B12000_seed0": accel(a.root, a.out, 12000, 0),
        "lifetime_seed0": lifetime(a.root, (3000, 24000), 0),
        "worst": {f"{s}_s{st}_B{b}": worst(a.root, a.out, s, st, b, 0)
                  for s, st, b in (("Jockey", 3, 6000), ("ShakeNDry", 3, 6000), ("YachtRide", 1, 3000))},
    }
    (a.out / "diagnostics.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
