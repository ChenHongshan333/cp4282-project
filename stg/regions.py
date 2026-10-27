"""PSNR inside and outside the object masks, recomputed for fitted runs from their params.npz.

Runs made before the object masks existed (``stg.masks``) used optical-flow masks, which are wrong
for Jockey and YachtRide (PROGRESS.md, step 11). This recomputes every run with the object masks, so
that all runs use the same regions, and overwrites the region fields of metrics.json.

    python -m stg.regions runs/test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from stg import data
from stg.metrics import masked_psnr
from stg.model import SpaceTimeGaussians
from stg.render import ClipRenderer
from stg.run import load_masks


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path)
    p.add_argument("--data-root", type=Path, default=Path("data/uvg"))
    p.add_argument("--mask-root", type=Path, default=Path("data/masks"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--redo", action="store_true", help="also recompute runs that already used the object masks")
    a = p.parse_args(argv)
    wp.init()
    for path in sorted(a.root.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        if m.get("mask_kind") == "object" and not a.redo:
            continue
        frames = data.load_clip(a.data_root, m["sequence"], m["start"], m["frames"], m["stride"])
        args = argparse.Namespace(mask_root=a.mask_root, sequence=m["sequence"], start=m["start"],
                                  frames=m["frames"], stride=m["stride"])
        masks, kind = load_masks(args, frames)
        if kind != "object":
            raise SystemExit(f"no object masks for every frame of {path.parent.name}; run stg.masks first")
        g = SpaceTimeGaussians.load(path.parent / "params.npz")
        images = ClipRenderer(*frames.shape[:3], a.device).render(g, m["s_min"])
        m["mask_kind"] = kind
        m["mask_fraction"] = float(masks.mean())
        m["psnr_in_mask_per_frame"] = [masked_psnr(images[t], frames[t], masks[t]) for t in range(len(frames))]
        m["psnr_out_mask_per_frame"] = [masked_psnr(images[t], frames[t], ~masks[t]) for t in range(len(frames))]
        path.write_text(json.dumps(m, indent=1))
        print(f"{path.parent.name}: in {np.mean(m['psnr_in_mask_per_frame']):.2f}, "
              f"out {np.mean(m['psnr_out_mask_per_frame']):.2f}", flush=True)


if __name__ == "__main__":
    main()
