"""Reconstruction videos for the project page (docs/). Trains nothing; renders the fitted runs in ``runs/``.

Three modules, each one MP4 per selection with all panels side by side, so they play in step:

    equal     Section 5.1.2: the original and M0 to M3 on the 10 fitted frames of the fully trained
              runs of ``long_test`` (start 0, 10 frames, stride 3, seed 0), budgets 6,000 and 24,000.
    confirm   Section 5.3.1: the original, M0 and M1 of the confirmatory test (``confirm``: Jockey
              from frame 120, ShakeNDry from frame 30, stride 3, seed 0) on all 28 frames from the
              first to the last fitted frame, the 18 unseen frames rendered by the model at their
              own times, and a fixed region of each clip enlarged 2x.
    arcs      Section 5.3.2: the original, M0 and M1 of the slow arcs (``arc_slow``, as in Figure 8)
              for the straight path (0 degrees) and the full turn (360 degrees), all 28 frames.

Clip time t of a run with start s and stride 3 is original frame s + 3t: fitted frames are at whole
t, and frame s + 3k + j (j = 1, 2) is drawn at t = k + j/3, as ``stg.unseen`` scores it. Every
rendered frame's PSNR is checked against the metrics.json of its run. Each selection is written in
two layouts, ``wide`` (3 panels across) and ``narrow`` (2 across, for phones):

    docs/media/recon/<module>_<clip>_B<budget>_<layout>.mp4   and .jpg (poster, first frame)
    docs/media/recon/videos.json                                runs, settings and frame list

    python -m stg.export_videos                 # needs runs/, data/uvg, data/masks, a GPU and ffmpeg
    python -m stg.export_videos --only arcs     # one module
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stg import data

LABEL = {"original": "Original", "M0": "Straight path · M0", "M1": "Curved path · M1",
         "M2": "Shrinking · M2", "M3": "Curved + shrinking · M3"}
BUDGETS = (6000, 24000)
SEED, STRIDE, ITERATIONS, FRAMES = 0, 3, 30000, 10
SPAN = STRIDE * (FRAMES - 1) + 1                 # 28 original frames from the first to the last fitted
W, H = data.SIZE                                 # 480 x 270, the resolution of every experiment
STRIP, HEADER = 24, 34                           # panel label and frame banner heights (pixels)
# Fixed enlarged regions (x, y, width, height), the same for every panel and frame of a clip: the
# horse's legs over frames 120-147 and the dog over frames 30-57, from their object masks. 2x
# nearest-neighbour fills one 480 x 270 panel.
REGION = {"Jockey_k120": (240, 135, 240, 135), "ShakeNDry_k30": (110, 103, 240, 135)}
ZOOM = 2
# The arcs: a 160 x 90 box around the middle of the frame holds the whole path of both arcs; 3x.
ARC_REGION, ARC_ZOOM = (160, 90, 160, 90), 3
FPS = {"equal": 2, "confirm": 6, "arcs": 6}      # display rates; UVG is 120 fps
COLOURS = {"fitted": (28, 92, 171), "unseen": (179, 84, 14), "ink": (20, 20, 20), "muted": (107, 106, 101),
           "paper": (255, 255, 255), "panel": (246, 245, 242)}


def font(size):
    return ImageFont.load_default(size=size)


def find_run(root: Path, stage: str, **want):
    """The one run folder of ``stage`` whose config matches ``want``."""
    hits = []
    for path in sorted((root / stage).glob("*/config.json")):
        c = json.loads(path.read_text())
        if all(c.get(k) == v for k, v in want.items()):
            hits.append(path.parent)
    if len(hits) != 1:
        raise RuntimeError(f"{stage} {want}: expected one run, found {len(hits)}")
    return hits[0]


def clip_times():
    """(original offset i, clip time t, fitted?) for the 28 frames of a stride-3 clip."""
    out = []
    for i in range(SPAN):
        k, j = divmod(i, STRIDE)
        out.append((i, float(k) if j == 0 else k + j / STRIDE, j == 0))
    return out


def psnr(a, b):
    return float(-10 * np.log10(np.mean((a - b) ** 2)))


def render(folder: Path, times, device):
    """Frames of a run at clip times, checked against the PSNR its metrics.json records."""
    import warp as wp
    from stg.model import SpaceTimeGaussians
    from stg.render import ClipRenderer
    m = json.loads((folder / "metrics.json").read_text())
    g = SpaceTimeGaussians.load(folder / "params.npz")
    images = ClipRenderer(len(times), H, W, device, times=list(times)).render(g, m["s_min"])
    recorded = {float(k): p for k, p in enumerate(m["psnr_per_frame"])}
    recorded.update({round(float(t), 6): p for t, p in zip(m.get("unseen_times", []), m.get("psnr_unseen_per_frame", []))})
    return images, m, {round(float(t), 6): recorded.get(round(float(t), 6)) for t in times}


def check(name, images, targets, times, recorded, tol=0.02):
    """Largest difference between the PSNR of the renders (as ``stg.metrics`` computes it, on the
    image clipped to [0, 1]) and the PSNR metrics.json records at the same clip times."""
    from stg.metrics import psnr_per_frame
    worst = 0.0
    for got, t in zip(psnr_per_frame(images, targets), times):
        want = recorded[round(float(t), 6)]
        if want is None:
            raise RuntimeError(f"{name}: no recorded PSNR at clip time {t}")
        worst = max(worst, abs(float(got) - want))
    if worst > tol:
        raise RuntimeError(f"{name}: rendered PSNR differs from metrics.json by {worst:.4f} dB")
    return worst


def to8(image):
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def enlarge(image, region, zoom):
    x, y, w, h = region
    return to8(image[y:y + h, x:x + w]).repeat(zoom, 0).repeat(zoom, 1)


def panel(img8, label):
    """A labelled panel: a strip with the label above a 480 x 270 image."""
    out = Image.new("RGB", (W, H + STRIP), COLOURS["paper"])
    out.paste(Image.fromarray(img8), (0, STRIP))
    d = ImageDraw.Draw(out)
    d.text((6, 4), label, fill=COLOURS["ink"], font=font(15))
    return out


def info_panel(lines):
    out = Image.new("RGB", (W, H + STRIP), COLOURS["panel"])
    d = ImageDraw.Draw(out)
    y = STRIP + 14
    for text, size, colour in lines:
        d.text((16, y), text, fill=COLOURS[colour], font=font(size))
        y += size + 9
    return out


def sheet(cells, cols, banner):
    """Cells in rows of ``cols`` under a banner: (text left, badge text, badge kind)."""
    rows = -(-len(cells) // cols)
    out = Image.new("RGB", (cols * W, HEADER + rows * (H + STRIP)), COLOURS["paper"])
    for n, c in enumerate(cells):
        out.paste(c, ((n % cols) * W, HEADER + (n // cols) * (H + STRIP)))
    d = ImageDraw.Draw(out)
    left, badge, kind = banner
    d.text((8, 7), left, fill=COLOURS["ink"], font=font(18))
    if badge:
        f = font(16)
        tw = d.textlength(badge, font=f)
        x0 = out.width - tw - 24
        d.rounded_rectangle((x0, 5, out.width - 8, HEADER - 5), radius=5, fill=COLOURS[kind])
        d.text((x0 + 8, 8), badge, fill=COLOURS["paper"], font=f)
    d.line((0, HEADER - 1, out.width, HEADER - 1), fill=(220, 218, 212))
    return out


def encode(frames, fps, path: Path, ffmpeg: str):
    """H.264 4:2:0 (what every browser plays) at high quality (CRF 16), a key frame every 7 frames so
    stepping to any frame decodes at most 6 others; decoded back to check the frame count and fidelity."""
    w, h = frames[0].size
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-g", "7",
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-movflags", "+faststart", "-an", str(path)]
    raw = b"".join(np.asarray(f, np.uint8).tobytes() for f in frames)
    subprocess.run(cmd, input=raw, check=True)
    frames[0].save(path.with_suffix(".jpg"), quality=90)
    # decode it back: frame count and fidelity
    out = subprocess.run([ffmpeg, "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    decoded = np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)
    if len(decoded) != len(frames):
        raise RuntimeError(f"{path.name}: {len(decoded)} frames decoded, {len(frames)} written")
    fidelity = min(psnr(decoded[i] / 255.0, np.asarray(f) / 255.0) for i, f in enumerate(frames))
    return {"file": path.name, "poster": path.with_suffix(".jpg").name, "width": w, "height": h,
            "bytes": path.stat().st_size, "decoded_psnr_min": round(fidelity, 2)}


def layouts(cells_wide, cells_narrow, banner):
    return {"wide": sheet(cells_wide, 3, banner), "narrow": sheet(cells_narrow, 2, banner)}


def run_info(folder, m):
    return {"run": folder.as_posix(), "psnr_fitted": round(m["psnr"], 3),
            "psnr_unseen": round(float(np.mean(m["psnr_unseen_per_frame"])), 3) if m.get("psnr_unseen_per_frame") else None,
            "gaussians": m["gaussians"]}


def write(module, key, sheets_per_frame, fps, out, ffmpeg, entry):
    entry["fps"] = fps
    entry["layouts"] = {}
    for layout in ("wide", "narrow"):
        path = out / f"{module}_{key}_{layout}.mp4"
        entry["layouts"][layout] = encode([s[layout] for s in sheets_per_frame], fps, path, ffmpeg)
    print(f"{module} {key}: {len(sheets_per_frame)} frames, decoded >= "
          f"{min(v['decoded_psnr_min'] for v in entry['layouts'].values()):.1f} dB", flush=True)
    return entry


def equal(a, out, ffmpeg):
    """Original and M0-M3 on the fitted frames of long_test."""
    entries = []
    for seq in data.SEQUENCES:
        targets = data.load_clip(a.data_root, seq, 0, FRAMES, STRIDE)
        times = [float(k) for k in range(FRAMES)]
        for b in BUDGETS:
            renders, runs = {}, {}
            for m_name in ("M0", "M1", "M2", "M3"):
                folder = find_run(a.root, "long_test", model=m_name, sequence=seq, start=0, frames=FRAMES,
                                  stride=STRIDE, budget=b, seed=SEED, iterations=ITERATIONS)
                images, m, recorded = render(folder, times, a.device)
                err = check(folder.name, images, targets, times, recorded)
                renders[m_name], runs[m_name] = images, dict(run_info(folder, m), check_dB=round(err, 4))
            sheets, frames = [], []
            for k in range(FRAMES):
                cells = [panel(to8(targets[k]), LABEL["original"])] + [panel(to8(renders[m][k]), LABEL[m]) for m in renders]
                info = info_panel([(f"{seq}, frames 0-27 at stride 3", 17, "ink"),
                                   (f"Budget {b:,} stored scalars", 15, "ink"),
                                   (f"{ITERATIONS:,} training steps, seed {SEED}", 15, "ink"),
                                   ("Only the 10 fitted frames are shown.", 14, "muted"),
                                   (f"Shown at {FPS['equal']} frames per second.", 14, "muted")])
                banner = (f"{seq} · original frame {STRIDE * k}", f"Fitted frame {k + 1} of {FRAMES}", "fitted")
                sheets.append(layouts(cells + [info], cells + [info], banner))
                frames.append({"frame": STRIDE * k, "t": k, "kind": "fitted"})
            entry = {"module": "equal", "clip": seq, "start": 0, "budget": b, "stride": STRIDE, "seed": SEED,
                     "iterations": ITERATIONS, "frames": frames, "runs": runs}
            entries.append(write("equal", f"{seq}_B{b}", sheets, FPS["equal"], out, ffmpeg, entry))
    return entries


def interpolation(a, clips, stage):
    """Targets and the renders of M0 and M1 on all 28 frames, per clip and budget."""
    entries = []
    spec = clip_times()
    times = [t for _, t, _ in spec]
    for seq, start, title in clips:
        if stage == "arc_slow":
            targets = data.arc_clip(seq, start, SPAN, 1)[0]
        else:
            targets = data.load_clip(a.data_root, seq, start, SPAN, 1)
        for b in BUDGETS:
            renders, runs = {}, {}
            for m_name in ("M0", "M1"):
                folder = find_run(a.root, stage, model=m_name, sequence=seq, start=start, frames=FRAMES,
                                  stride=STRIDE, budget=b, seed=SEED, iterations=ITERATIONS)
                images, m, recorded = render(folder, times, a.device)
                err = check(folder.name, images, targets, times, recorded)
                renders[m_name], runs[m_name] = images, dict(run_info(folder, m), check_dB=round(err, 4))
            entries.append((seq, start, title, b, targets, renders, runs))
    return entries


def confirm(a, out, ffmpeg):
    clips = [("Jockey", 120, "Jockey"), ("ShakeNDry", 30, "ShakeNDry")]
    result = []
    for seq, start, title, b, targets, renders, runs in interpolation(a, clips, "confirm"):
        region = REGION[f"{seq}_k{start}"]
        sheets, frames = [], []
        for i, t, fitted in clip_times():
            src = {"original": targets[i], "M0": renders["M0"][i], "M1": renders["M1"][i]}
            full = [panel(to8(src[k]), LABEL[k]) for k in ("original", "M0", "M1")]
            zoom = [panel(enlarge(src[k], region, ZOOM), f"{LABEL[k]} · region {ZOOM}x") for k in ("original", "M0", "M1")]
            kind = "fitted" if fitted else "unseen"
            banner = (f"{title} · original frame {start + i}", "Fitted frame" if fitted else "Unseen frame", kind)
            narrow = [c for pair in zip(full, zoom) for c in pair]
            sheets.append(layouts(full + zoom, narrow, banner))
            frames.append({"frame": start + i, "t": round(t, 4), "kind": kind})
        entry = {"module": "confirm", "clip": seq, "start": start, "budget": b, "stride": STRIDE, "seed": SEED,
                 "iterations": ITERATIONS, "region": dict(zip("xywh", region)), "zoom": ZOOM, "frames": frames, "runs": runs}
        result.append(write("confirm", f"{seq}_k{start}_B{b}", sheets, FPS["confirm"], out, ffmpeg, entry))
    return result


def arcs(a, out, ffmpeg):
    clips = [("arcs0", 0, "0° (straight)"), ("arcs360", 0, "360° (full turn)")]
    got = {}
    for seq, start, title, b, targets, renders, runs in interpolation(a, clips, "arc_slow"):
        got[(seq, b)] = (title, targets, renders, runs)
    result = []
    for b in BUDGETS:
        sheets, frames = [], []
        for i, t, fitted in clip_times():
            rows = []
            for seq, _, _ in clips:
                title, targets, renders, _ = got[(seq, b)]
                src = {"original": targets[i], "M0": renders["M0"][i], "M1": renders["M1"][i]}
                rows.append([panel(enlarge(src[k], ARC_REGION, ARC_ZOOM), f"{title.split(' ')[0]} · {LABEL[k]}")
                             for k in ("original", "M0", "M1")])
            wide = rows[0] + rows[1]
            narrow = [c for pair in zip(rows[0], rows[1]) for c in pair]
            kind = "fitted" if fitted else "unseen"
            banner = (f"Synthetic arcs · frame {i}", "Fitted frame" if fitted else "Unseen frame", kind)
            sheets.append(layouts(wide, narrow, banner))
            frames.append({"frame": i, "t": round(t, 4), "kind": kind})
        entry = {"module": "arcs", "clip": "arcs0+arcs360", "start": 0, "budget": b, "stride": STRIDE, "seed": SEED,
                 "iterations": ITERATIONS, "region": dict(zip("xywh", ARC_REGION)), "zoom": ARC_ZOOM, "frames": frames,
                 "runs": {f"{seq}_{m}": got[(seq, b)][3][m] for seq, _, _ in clips for m in ("M0", "M1")}}
        result.append(write("arcs", f"B{b}", sheets, FPS["arcs"], out, ffmpeg, entry))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("docs/media/recon"))
    p.add_argument("--data-root", type=Path, default=Path("data/uvg"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    p.add_argument("--only", choices=("equal", "confirm", "arcs"))
    a = p.parse_args(argv)
    import warp as wp
    wp.init()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest_path = a.out / "videos.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for module, fn in (("equal", equal), ("confirm", confirm), ("arcs", arcs)):
        if a.only and a.only != module:
            continue
        manifest[module] = fn(a, a.out, a.ffmpeg)
    manifest["command"] = "python -m stg.export_videos" + (f" --only {a.only}" if a.only else "")
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
