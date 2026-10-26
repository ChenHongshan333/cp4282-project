"""VMAF of fitted runs, added to their metrics.json after the fit.

The renders are rebuilt from ``params.npz``, so VMAF can be computed for runs made before it was
added. Each clip and its reference are written as 8-bit PNGs, scaled to 1920x1080 with bicubic
filtering (the viewing condition the default model vmaf_v0.6.1 was trained for) and compared by
ffmpeg's libvmaf filter. The frames are written under the run root, not the system temp folder.

    python -m stg.vmaf runs/trial                 # every run in the folder that has no VMAF yet
    python -m stg.vmaf runs/test --ffmpeg F:/Tools/FFmpeg/ffmpeg-9.0.1-full_build/bin/ffmpeg.exe
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

import gaussian_image_video as course
from stg import data
from stg.model import SpaceTimeGaussians
from stg.render import ClipRenderer

SCALE = "scale=1920:1080:flags=bicubic,format=yuv420p"


def vmaf_of(images: np.ndarray, frames: np.ndarray, work: Path, ffmpeg: str) -> list[float]:
    """Per-frame VMAF of ``images`` against ``frames``, both (T, H, W, 3) in [0, 1]."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for t in range(len(frames)):
        course.save(work / f"dist_{t:03d}.png", images[t])
        course.save(work / f"ref_{t:03d}.png", frames[t])
    log = work / "vmaf.json"
    # libvmaf takes the distorted stream first and the reference second.
    graph = f"[0:v]{SCALE}[d];[1:v]{SCALE}[r];[d][r]libvmaf=log_fmt=json:log_path={log.name}"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "dist_%03d.png", "-i", "ref_%03d.png",
                    "-lavfi", graph, "-f", "null", "-"], cwd=work, check=True)
    result = json.loads(log.read_text())
    values = [f["metrics"]["vmaf"] for f in result["frames"]]
    shutil.rmtree(work)
    return values


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path)
    p.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    p.add_argument("--data-root", type=Path, default=Path("data/uvg"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--redo", action="store_true", help="recompute runs that already have VMAF")
    a = p.parse_args(argv)
    wp.init()
    for path in sorted(a.root.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        if "vmaf" in m and not a.redo:
            continue
        frames = data.load_clip(a.data_root, m["sequence"], m["start"], m["frames"], m["stride"])
        g = SpaceTimeGaussians.load(path.parent / "params.npz")
        images = ClipRenderer(*frames.shape[:3], a.device).render(g, m["s_min"])
        values = vmaf_of(images, frames, a.root / "_vmaf_work", a.ffmpeg)
        m["vmaf_per_frame"] = values
        m["vmaf"] = float(np.mean(values))
        path.write_text(json.dumps(m, indent=1))
        print(f"{path.parent.name}: VMAF {m['vmaf']:.2f}", flush=True)


if __name__ == "__main__":
    main()
