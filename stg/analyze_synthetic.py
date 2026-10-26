"""Analysis of the synthetic stage, where the true paths of the blobs are known.

For every run in ``runs/synthetic/``: the PSNR of the whole clip, the PSNR inside the disc of each
blob (parabola, swing, straight) over all frames, and for models with acceleration, the share of
Gaussians whose acceleration moves the centre by less than half a pixel within |tau| <= 2 sigma_t.
Writes ``analysis.json``, ``table.md`` and ``rd.png`` to the stage directory.

    python -m stg.analyze_synthetic --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import warp as wp

from stg import data
from stg.model import SpaceTimeGaussians
from stg.render import ClipRenderer

BLOBS = ("parabola", "swing", "straight")


def blob_masks(paths, n_frames, height, width, margin=3.0):
    """(T, H, W) disc around each blob's true centre, one per blob."""
    yy, xx = np.mgrid[0:height, 0:width] + 0.5
    radius = 0.05 * height + margin
    return {name: np.stack([(xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2 for cx, cy in centres[:n_frames]])
            for name, centres in paths.items()}


def pooled_psnr(images, frames, mask):
    err = (np.clip(images, 0, 1) - frames) ** 2
    return float(-10 * np.log10(err[mask].mean()))


def analyse_run(folder: Path, renderer, frames, masks):
    m = json.loads((folder / "metrics.json").read_text())
    g = SpaceTimeGaussians.load(folder / "params.npz")
    images = renderer.render(g, m["s_min"])
    row = {k: m[k] for k in ("model", "budget", "seed", "gaussians", "psnr", "ssim")}
    for name in BLOBS:
        row[f"psnr_{name}"] = pooled_psnr(images, frames, masks[name])
    if g.spec.accel:
        shift = np.linalg.norm(g.acc, axis=1) * (2 * g.sigma_t) ** 2
        row["acc_unused_share"] = float(np.mean(shift < 0.5))
        row["acc_shift_median"] = float(np.median(shift))
    row["sigma_t_median"] = float(np.median(g.sigma_t))
    return row


def summarise(rows):
    """Per (budget, model): mean and min-max over seeds, and the gain over M0 paired by seed."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["budget"], r["model"])][r["seed"]] = r
    keys = ["psnr"] + [f"psnr_{b}" for b in BLOBS]
    out = []
    for (budget, model), seeds in sorted(by.items()):
        base = by.get((budget, "M0"), {})
        entry = {"budget": budget, "model": model, "seeds": len(seeds)}
        for k in keys:
            values = np.array([s[k] for s in seeds.values()])
            entry[k] = [float(values.mean()), float(values.min()), float(values.max())]
            gains = np.array([seeds[s][k] - base[s][k] for s in seeds if s in base])
            entry[f"gain_{k}"] = [float(gains.mean()), float(gains.min()), float(gains.max())] if len(gains) else None
        if "acc_unused_share" in next(iter(seeds.values())):
            entry["acc_unused_share"] = float(np.mean([s["acc_unused_share"] for s in seeds.values()]))
        entry["sigma_t_median"] = float(np.mean([s["sigma_t_median"] for s in seeds.values()]))
        out.append(entry)
    return out


def table(summary):
    lines = ["| 存储量 | 方法 | PSNR（整体） | 比 M0 高 | 抛物线 | 比 M0 高 | 来回摆 | 比 M0 高 | 直线 | 比 M0 高 | 加速度几乎没用的光斑 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in summary:
        cells = [str(e["budget"]), e["model"]]
        for k in ("psnr", "psnr_parabola", "psnr_swing", "psnr_straight"):
            cells.append(f"{e[k][0]:.2f}")
            g = e[f"gain_{k}"]
            cells.append("—" if e["model"] == "M0" else f"{g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}]")
        cells.append(f"{e['acc_unused_share']:.0%}" if "acc_unused_share" in e else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, k, title in zip(axes, ("psnr", "psnr_parabola", "psnr_swing", "psnr_straight"),
                            ("whole clip", "parabola", "swing", "straight")):
        for model in ("M0", "M1", "M2", "M3"):
            es = [e for e in summary if e["model"] == model]
            b = [e["budget"] for e in es]
            mean = [e[k][0] for e in es]
            ax.plot(b, mean, marker="o", label=model)
            ax.fill_between(b, [e[k][1] for e in es], [e[k][2] for e in es], alpha=0.2)
        ax.set_xscale("log")
        ax.set_xticks([3000, 6000, 12000, 24000], ["3k", "6k", "12k", "24k"])
        ax.set_title(title)
        ax.set_xlabel("stored numbers")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=Path("runs/synthetic"))
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    wp.init()
    frames, paths = data.synthetic_clip()
    n_frames, height, width, _ = frames.shape
    renderer = ClipRenderer(n_frames, height, width, a.device)
    masks = blob_masks(paths, n_frames, height, width)
    rows = [analyse_run(f.parent, renderer, frames, masks) for f in sorted(a.root.glob("*/metrics.json"))]
    summary = summarise(rows)
    (a.root / "analysis.json").write_text(json.dumps({"runs": rows, "summary": summary}, indent=1))
    (a.root / "table.md").write_text(table(summary) + "\n", encoding="utf-8")
    plot(summary, a.root / "rd.png")
    print(table(summary))


if __name__ == "__main__":
    main()
