"""One fit: one model, budget, clip and seed. Writes ``<out>/<name>/`` with

    config.json    the arguments of the run
    metrics.json   PSNR and SSIM per frame, PSNR inside and outside the motion mask, timings, trace
    params.npz     the fitted Gaussians, float32: the stored model
    *.png          the target, render and error map of the worst frame

Example:
    python -m stg.run --model M1 --budget 6000 --sequence Jockey --stride 3 --seed 0 --device cuda
    python -m stg.run --model M0 --budget 3000 --sequence synthetic --iterations 200 --device cpu
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import gaussian_image_video as course
from stg import data, flow
from stg.fit import RATES, fit
from stg.metrics import masked_psnr, psnr_per_frame, ssim
from stg.model import MODELS, initial
from stg.render import render_clip


def parser():
    p = argparse.ArgumentParser(description="Fit one clip with space-time Gaussians.")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--budget", type=int, required=True, help="stored numbers (float32) for the whole clip")
    p.add_argument("--sequence", required=True, help=f"one of {data.SEQUENCES} or 'synthetic'")
    p.add_argument("--start", type=int, default=0, help="first frame of the clip in the sequence")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--s-min", type=float, default=0.1, help="shrink floor (M2, M3)")
    p.add_argument("--acc-ratio", type=float, default=1.0, help="learning rate of a / learning rate of v")
    p.add_argument("--sigma-t-init", type=float, default=2.0, help="initial duration, in frames")
    for name, rate in RATES.items():
        p.add_argument(f"--lr-{name.replace('_', '-')}", type=float, default=rate)
    p.add_argument("--data-root", type=Path, default=Path("data/uvg"))
    p.add_argument("--mask-root", type=Path, default=Path("data/masks"),
                   help="hand-drawn masks <mask-root>/<Sequence>/<frame>.png; otherwise flow masks are used")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--name", help="run directory name; by default built from the arguments")
    p.add_argument("--device", default="cuda")
    p.add_argument("--trace-every", type=int, default=100)
    return p


def run_name(a) -> str:
    return (f"{a.sequence}_k{a.start}_n{a.frames}_s{a.stride}_{a.model}_B{a.budget}_seed{a.seed}"
            f"_smin{a.s_min}_acc{a.acc_ratio}")


def load_masks(a, frames) -> tuple[np.ndarray, str]:
    """Hand-drawn masks if present for every frame of the clip, else optical-flow masks."""
    folder = a.mask_root / a.sequence
    indices = data.frame_indices(a.start, a.frames, a.stride)
    paths = [folder / f"{k:04d}.png" for k in indices]
    if all(p.exists() for p in paths):
        from PIL import Image
        h, w = frames.shape[1:3]
        return np.stack([np.asarray(Image.open(p).convert("L").resize((w, h), Image.NEAREST)) > 127
                         for p in paths]), "hand"
    return flow.motion_masks(frames), "flow"


def save_png(path, image):
    course.save(path, image)


def main(argv=None):
    a = parser().parse_args(argv)
    wp.init()
    spec = MODELS[a.model]
    out = a.out / (a.name or run_name(a))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                                                indent=1))

    frames = data.load_clip(a.data_root, a.sequence, a.start, a.frames, a.stride)
    n_frames, height, width, _ = frames.shape
    g = initial(spec, a.budget, frames, a.seed, a.sigma_t_init)
    assert g.stored_numbers() <= a.budget
    rates = {n: getattr(a, f"lr_{n}") for n in RATES}
    print(f"{out.name}: {len(g)} Gaussians, {g.stored_numbers()} numbers", flush=True)

    start = time.perf_counter()
    trace = fit(g, frames, a.iterations, a.device, a.s_min, rates, a.acc_ratio, a.trace_every,
                log=lambda s: print(s, flush=True))
    train_seconds = time.perf_counter() - start
    g.save(out / "params.npz")

    start = time.perf_counter()
    images = render_clip(g, n_frames, width, height, a.device, a.s_min)
    render_seconds = (time.perf_counter() - start) / n_frames
    per_frame = psnr_per_frame(images, frames)
    masks, mask_kind = load_masks(a, frames)
    metrics = {
        "model": a.model, "budget": a.budget, "sequence": a.sequence, "start": a.start, "frames": a.frames,
        "stride": a.stride, "seed": a.seed, "s_min": a.s_min, "acc_ratio": a.acc_ratio,
        "gaussians": len(g), "stored_numbers": g.stored_numbers(), "bytes": 4 * g.stored_numbers(),
        "psnr": float(per_frame.mean()),
        "psnr_per_frame": per_frame.tolist(),
        "ssim_per_frame": [ssim(images[t], frames[t]) for t in range(n_frames)],
        "mask_kind": mask_kind,
        "mask_fraction": float(masks.mean()),
        "psnr_in_mask_per_frame": [masked_psnr(images[t], frames[t], masks[t]) for t in range(n_frames)],
        "psnr_out_mask_per_frame": [masked_psnr(images[t], frames[t], ~masks[t]) for t in range(n_frames)],
        "train_seconds": train_seconds,
        "seconds_per_iteration": train_seconds / max(a.iterations, 1),
        "render_seconds_per_frame": render_seconds,
        "trace": trace,
        "python": sys.version.split()[0], "warp": wp.config.version, "platform": platform.platform(),
    }
    metrics["ssim"] = float(np.mean(metrics["ssim_per_frame"]))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    worst = int(np.argmin(per_frame))
    save_png(out / f"frame{worst}_target.png", frames[worst])
    save_png(out / f"frame{worst}_render.png", images[worst])
    save_png(out / f"frame{worst}_error.png", np.abs(np.clip(images[worst], 0, 1) - frames[worst]) * 4)
    print(f"{out.name}: PSNR {metrics['psnr']:.2f} dB, SSIM {metrics['ssim']:.4f}, "
          f"{train_seconds:.0f}s", flush=True)
    return metrics


if __name__ == "__main__":
    main()
