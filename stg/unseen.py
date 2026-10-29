"""PSNR on the frames a fit never saw (PROGRESS.md, plan 2d).

A clip at stride s is fitted to the frames start, start + s, ...; the frames between them are not
used. This renders the fit at the clip times k + j / s, j = 1 .. s - 1, between its frames k and
k + 1, and compares them with those frames of the sequence: if the paths of the Gaussians follow the
motion, the frames between should be as good as the fitted ones. Adds ``psnr_unseen_per_frame`` and
the PSNR inside and outside the object masks to every metrics.json of the folder.

    python -m stg.unseen runs/long_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from stg import data
from stg.metrics import masked_psnr, psnr_per_frame
from stg.model import SpaceTimeGaussians
from stg.render import ClipRenderer
from stg.run import load_masks


def unseen(m: dict, data_root: Path, mask_root: Path):
    """The frames between the fitted ones as (frames, clip times, masks or None)."""
    stride, n = m["stride"], m["frames"] - 1
    frames, times, masks = [], [], []
    for j in range(1, stride):
        frames.append(data.load_clip(data_root, m["sequence"], m["start"] + j, n, stride))
        times += [k + j / stride for k in range(n)]
        args = argparse.Namespace(mask_root=mask_root, sequence=m["sequence"], start=m["start"] + j,
                                  frames=n, stride=stride)
        masks.append(load_masks(args, frames[-1]))
    kinds = {kind for _, kind in masks}
    return np.concatenate(frames), times, np.concatenate([mk for mk, _ in masks]) if kinds == {"object"} else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path)
    p.add_argument("--data-root", type=Path, default=Path("data/uvg"))
    p.add_argument("--mask-root", type=Path, default=Path("data/masks"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--redo", action="store_true", help="also recompute runs that already have the fields")
    a = p.parse_args(argv)
    wp.init()
    for path in sorted(a.root.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        if m["stride"] < 2 or ("psnr_unseen_per_frame" in m and not a.redo):
            continue
        frames, times, masks = unseen(m, a.data_root, a.mask_root)
        g = SpaceTimeGaussians.load(path.parent / "params.npz")
        images = ClipRenderer(*frames.shape[:3], a.device, times=times).render(g, m["s_min"])
        m["unseen_times"] = times
        m["psnr_unseen_per_frame"] = psnr_per_frame(images, frames).tolist()
        if masks is not None:
            m["psnr_unseen_in_mask_per_frame"] = [masked_psnr(images[t], frames[t], masks[t]) for t in range(len(frames))]
            m["psnr_unseen_out_mask_per_frame"] = [masked_psnr(images[t], frames[t], ~masks[t]) for t in range(len(frames))]
        path.write_text(json.dumps(m, indent=1))
        print(f"{path.parent.name}: fitted frames {m['psnr']:.2f}, frames between "
              f"{np.mean(m['psnr_unseen_per_frame']):.2f}", flush=True)


if __name__ == "__main__":
    main()
