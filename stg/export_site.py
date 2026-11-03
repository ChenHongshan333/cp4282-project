"""Export the results in ``runs/`` for the project page in ``docs/`` (PROGRESS.md, step 33). Trains nothing.

Writes to ``docs/data/``:

    models/<id>.bin    the fitted Gaussians of the runs the page renders, float32 little-endian, the
                       stored attributes one after another in the order of ``ORDER``
    models.json        for every model id: model, count, s_min, clip, per-frame PSNR, file
    frames/<clip>/<frame>.webp   the frames of the clips the page shows, at 480 x 270
    masks/<clip>/<frame>.png     their object masks
    runs.json          one compact record per run of every stage, with its trace

and, with ``--check``, reference renders by the Python renderer to ``--check-dir``, which
``docs/assets/js/check.mjs`` compares with the renderer of the page.

    python -m stg.export_site
    python -m stg.export_site --check --check-dir <folder>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stg import data
from stg.model import SpaceTimeGaussians

ORDER = ("mu", "vel", "acc", "poly", "t0", "sigma_t", "chol", "colour")

# The clips the page plays: (sequence, first frame, number of frames). Each covers the 10 fitted
# frames and the frames between them at the largest stride used on it.
CLIPS = [("Jockey", 0, 28), ("ShakeNDry", 0, 28), ("YachtRide", 0, 28), ("Bosphorus", 0, 28),
         ("ShakeNDry", 30, 46), ("Jockey", 120, 46), ("Jockey", 200, 46)]

STAGES = ("synthetic", "validation", "test", "still", "probe_dense", "long", "converge", "long_validation",
          "long_transfer", "long_test", "poly", "arc", "arc_slow", "confirm")


def clip_name(sequence, start):
    return f"{sequence}_k{start}"


def model_id(m, iterations):
    return (f"{m['sequence']}_k{m['start']}_s{m['stride']}_{m['model']}_B{m['budget']}"
            f"_seed{m['seed']}_{iterations // 1000}k")


def mean(values):
    v = np.array(values, float)
    return float(np.nanmean(v)) if v.size and not np.all(np.isnan(v)) else None


def record(stage, m, config):
    """The numbers of one run the page needs."""
    r = {"stage": stage, "sequence": m["sequence"], "start": m["start"], "frames": m["frames"],
         "stride": m["stride"], "model": m["model"], "budget": m["budget"], "seed": m["seed"],
         "iterations": config["iterations"], "lr_vel": m.get("lr_vel"), "s_min": m["s_min"],
         "acc_ratio": m["acc_ratio"], "gaussians": m["gaussians"],
         "psnr": m["psnr"], "ssim": m["ssim"], "vmaf": m.get("vmaf"),
         "psnr_in": mean(m["psnr_in_mask_per_frame"]), "psnr_out": mean(m["psnr_out_mask_per_frame"]),
         "train_seconds": m["train_seconds"], "seconds_per_iteration": m["seconds_per_iteration"],
         "render_seconds_per_frame": m["render_seconds_per_frame"],
         "trace": [[int(s), round(float(p), 3), round(float(t), 2)] for s, _, p, t in m["trace"]]}
    if "psnr_unseen_per_frame" in m:
        r["unseen"] = mean(m["psnr_unseen_per_frame"])
        r["unseen_in"] = mean(m.get("psnr_unseen_in_mask_per_frame", []))
        r["unseen_out"] = mean(m.get("psnr_unseen_out_mask_per_frame", []))
    return r


def runs(root: Path):
    """(stage, metrics, config, folder) for every run of the stages."""
    for stage in STAGES:
        for path in sorted((root / stage).glob("*/metrics.json")):
            config = json.loads((path.parent / "config.json").read_text())
            yield stage, json.loads(path.read_text()), config, path.parent


def rendered(stage, m, config):
    """Whether the page renders this run: seed 0 of the clips it plays, the arcs of plan 2e'."""
    if m["seed"] != 0:
        return False
    if stage in ("test", "long_test"):
        return m["frames"] == 10 and m["stride"] == 3 and m["start"] == 0
    return stage in ("poly", "confirm") or (stage == "arc_slow" and m["sequence"].startswith("arcs"))


def save_model(g: SpaceTimeGaussians, path: Path):
    parts = [g.params[n].astype("<f4").ravel() for n in ORDER if n in g.params]
    np.concatenate(parts).tofile(path)


def times(m):
    """Clip times of the fitted frames and of the frames between them, with their PSNR over the whole
    frame and inside the object mask."""
    def entry(t, p, p_in, fitted):
        return {"t": round(float(t), 4), "psnr": p, "psnr_in": None if p_in is None or np.isnan(p_in) else p_in,
                "fitted": fitted}
    inside = m.get("psnr_in_mask_per_frame") if m.get("mask_kind") == "object" else None
    out = [entry(k, p, inside[k] if inside else None, True) for k, p in enumerate(m["psnr_per_frame"])]
    unseen_in = m.get("psnr_unseen_in_mask_per_frame")
    for j, (t, p) in enumerate(zip(m.get("unseen_times", []), m.get("psnr_unseen_per_frame", []))):
        out.append(entry(t, p, unseen_in[j] if unseen_in else None, False))
    return sorted(out, key=lambda x: x["t"])


def export_frames(out: Path, data_root: Path, mask_root: Path):
    from PIL import Image
    for sequence, start, count in CLIPS:
        folder = out / "frames" / clip_name(sequence, start)
        masks = out / "masks" / clip_name(sequence, start)
        folder.mkdir(parents=True, exist_ok=True)
        masks.mkdir(parents=True, exist_ok=True)
        frames = data.load_clip(data_root, sequence, start, count, 1)
        for i, frame in enumerate(frames):
            k = start + i
            Image.fromarray(np.round(frame * 255).astype(np.uint8)).save(folder / f"{k:04d}.webp", quality=90, method=6)
            src = mask_root / sequence / f"{k:04d}.png"
            if src.exists():
                mask = Image.open(src).convert("L").resize(data.SIZE, Image.NEAREST)
                Image.fromarray((np.asarray(mask) > 127).astype(np.uint8) * 255).convert("1").save(masks / f"{k:04d}.png", optimize=True)
        print(f"frames {clip_name(sequence, start)}: {count}", flush=True)


def check(models: dict, root: Path, out: Path, device: str):
    """Reference renders of a few models at fitted and in-between times, float32 (H, W, 3)."""
    import warp as wp
    from stg.render import ClipRenderer
    wp.init()
    out.mkdir(parents=True, exist_ok=True)
    cases = [("Jockey_k0_s3_M0_B6000_seed0_30k", (0.0, 4.0, 4.3333)),
             ("Jockey_k0_s3_M1_B24000_seed0_30k", (0.0, 4.6667, 9.0)),
             ("ShakeNDry_k0_s3_M2_B6000_seed0_30k", (2.0, 2.3333)),
             ("Bosphorus_k0_s3_M3_B24000_seed0_3k", (5.0, 5.6667)),
             ("Jockey_k0_s3_D7_B6000_seed0_30k", (1.0, 1.5)),
             ("arcs270_k0_s3_M1_B6000_seed0_30k", (3.0, 3.3333))]
    index = []
    for mid, ts in cases:
        info = models[mid]
        g = SpaceTimeGaussians.load(root / info["run"] / "params.npz")
        width, height = data.SIZE
        images = ClipRenderer(len(ts), height, width, device, times=list(ts)).render(g, info["s_min"])
        for t, image in zip(ts, images):
            name = f"{mid}_t{t:.4f}.bin"
            image.astype("<f4").tofile(out / name)
            index.append({"model": mid, "t": t, "file": name})
    # the arc clip at fitted and in-between times, as the page draws it
    for angle in (0, 180, 360):
        for t in (0.0, 2.0, 4.3333):
            frame = arc_frame(f"arcs{angle}", t)
            name = f"arcs{angle}_truth_t{t:.4f}.bin"
            frame.astype("<f4").tofile(out / name)
            index.append({"arc": f"arcs{angle}", "t": t, "file": name})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} references in {out}")


def arc_frame(sequence, t):
    """The arc clip at clip time t (stride 3), by the code of ``data.arc_clip`` at a fractional frame."""
    style, angle = data.arc_style(sequence)
    width, height = data.SIZE
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32) + 0.5
    background = np.stack([0.3 + 0.3 * xx / width, 0.4 + 0.1 * np.sin(2 * np.pi * yy / height * 2),
                           0.5 - 0.2 * yy / height], axis=-1)
    (cx, cy), = data.arc_centres(angle, np.array([3.0 * t]), data.SIZE, style["speed"])
    dx, dy = xx - cx, yy - cy
    d = np.sqrt(dx ** 2 + dy ** 2)
    alpha = np.clip(style["radius"] + 1.0 - d, 0, 1)[..., None]
    stripes = 0.5 + 0.5 * np.sin(2 * np.pi * (dx + 0.5 * dy) / style["period"])
    ball = np.stack([0.9 - 0.4 * stripes, 0.2 + 0.6 * stripes, 0.15 + 0 * stripes], axis=-1)
    return ((1 - alpha) * background + alpha * ball).astype(np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("docs/data"))
    p.add_argument("--data-root", type=Path, default=Path("data/uvg"))
    p.add_argument("--mask-root", type=Path, default=Path("data/masks"))
    p.add_argument("--no-frames", action="store_true", help="skip the frames and masks")
    p.add_argument("--check", action="store_true", help="write reference renders and stop")
    p.add_argument("--check-dir", type=Path)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    (a.out / "models").mkdir(parents=True, exist_ok=True)
    records, models = [], {}
    for stage, m, config, folder in runs(a.root):
        records.append(record(stage, m, config))
        if not rendered(stage, m, config):
            continue
        mid = model_id(m, config["iterations"])
        models[mid] = {"run": f"{stage}/{folder.name}", "model": m["model"], "sequence": m["sequence"],
                       "start": m["start"], "stride": m["stride"], "budget": m["budget"],
                       "iterations": config["iterations"], "s_min": m["s_min"], "gaussians": m["gaussians"],
                       "psnr": m["psnr"], "unseen": mean(m.get("psnr_unseen_per_frame", [])),
                       "times": times(m), "file": f"models/{mid}.bin"}
        if not a.check:
            g = SpaceTimeGaussians.load(folder / "params.npz")
            assert len(g) == m["gaussians"]
            save_model(g, a.out / "models" / f"{mid}.bin")

    if a.check:
        check(models, a.root, a.check_dir, a.device)
        return
    (a.out / "models.json").write_text(json.dumps({"order": ORDER, "models": models}, separators=(",", ":")))
    (a.out / "runs.json").write_text(json.dumps(records, separators=(",", ":")))
    print(f"{len(records)} runs, {len(models)} models", flush=True)
    if not a.no_frames:
        export_frames(a.out, a.data_root, a.mask_root)


if __name__ == "__main__":
    main()
