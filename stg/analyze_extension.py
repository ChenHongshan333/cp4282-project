"""Analysis of plans 2c and 2d (PROGRESS.md): the frames a fit never saw (``stg.unseen``), and the
centre paths of higher degree (D3, D7) against M0 (degree 1) and M1 (degree 2).

Gains over M0 are paired by seed, as the mean and the min-max over the seeds. Writes ``tables.md``
and ``degree.png`` to ``--out``.

    python -m stg.analyze_extension
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from stg.analyze_test import BUDGETS, fmt
from stg.data import SEQUENCES

DEGREES = {"M0": 1, "M1": 2, "D3": 3, "D7": 7}
VALUES = {"psnr": "训练用的帧", "unseen": "没见过的中间帧", "unseen_in": "中间帧的物体区域", "unseen_out": "中间帧的背景"}


def load(*roots):
    """(sequence, budget) -> model -> seed -> {value name: number}, for stride-3 runs with unseen frames."""
    out = defaultdict(lambda: defaultdict(dict))
    for root in roots:
        for path in root.glob("*/metrics.json"):
            m = json.loads(path.read_text())
            if m["stride"] != 3 or "psnr_unseen_per_frame" not in m:
                continue
            out[(m["sequence"], m["budget"])][m["model"]][m["seed"]] = {
                "psnr": m["psnr"], "unseen": float(np.mean(m["psnr_unseen_per_frame"])),
                "unseen_in": float(np.nanmean(m.get("psnr_unseen_in_mask_per_frame", [np.nan]))),
                "unseen_out": float(np.nanmean(m.get("psnr_unseen_out_mask_per_frame", [np.nan])))}
    return out


def gain(cell, model, value):
    seeds = sorted(set(cell.get(model, {})) & set(cell["M0"]))
    g = np.array([cell[model][s][value] - cell["M0"][s][value] for s in seeds], float)
    return (float(np.nanmean(g)), float(np.nanmin(g)), float(np.nanmax(g))) if len(g) else (np.nan,) * 3


def unseen_tables(cells, models=("M1", "M2", "M3")):
    lines = ["## 没见过的帧（计划 2d）", "",
             "**4 个存储量平均：和 M0 的差距（dB），训练用的帧 → 没见过的中间帧**", "",
             "| 视频 | " + " | ".join(f"{m} 比 M0" for m in models) + " |", "|---|" + "---|" * len(models)]
    for seq in SEQUENCES:
        row = []
        for m in models:
            fitted = np.nanmean([gain(cells[(seq, b)], m, "psnr")[0] for b in BUDGETS if (seq, b) in cells])
            between = np.nanmean([gain(cells[(seq, b)], m, "unseen")[0] for b in BUDGETS if (seq, b) in cells])
            row.append(f"{fitted:+.2f} → {between:+.2f}")
        lines.append(f"| {seq} | " + " | ".join(row) + " |")
    for value in ("unseen", "unseen_in", "unseen_out"):
        lines += ["", f"**{VALUES[value]}：和 M0 的差距（dB），种子平均 [最小, 最大]**", "",
                  "| 视频 | 存储量 | M0 | " + " | ".join(models) + " |", "|---|---|---|" + "---|" * len(models)]
        for seq in SEQUENCES:
            for budget in BUDGETS:
                cell = cells.get((seq, budget))
                if not cell or "M0" not in cell:
                    continue
                m0 = np.nanmean([r[value] for r in cell["M0"].values()])
                lines.append(f"| {seq} | {budget} | {m0:.2f} | " + " | ".join(fmt(gain(cell, m, value)) for m in models) + " |")
    return "\n".join(lines)


def degree_tables(cells):
    lines = ["## 更高次的拐弯（计划 2c）", ""]
    for value in ("psnr", "unseen", "unseen_in"):
        lines += [f"**{VALUES[value]}：和 M0（1 次）的差距（dB），种子平均 [最小, 最大]**", "",
                  "| 视频 | 存储量 | M0（1 次） | M1（2 次） | D3（3 次） | D7（7 次） |", "|---|---|---|---|---|---|"]
        for seq in ("Jockey", "ShakeNDry"):
            for budget in (6000, 24000):
                cell = cells.get((seq, budget))
                if not cell or "M0" not in cell:
                    continue
                m0 = np.nanmean([r[value] for r in cell["M0"].values()])
                lines.append(f"| {seq} | {budget} | {m0:.2f} | " +
                             " | ".join(fmt(gain(cell, m, value)) for m in ("M1", "D3", "D7")) + " |")
        lines.append("")
    return "\n".join(lines)


def plot(cells, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (value, title) in zip(axes, (("psnr", "fitted frames"), ("unseen", "frames between (never seen)"))):
        for i, (seq, budget) in enumerate((s, b) for s in ("Jockey", "ShakeNDry") for b in (6000, 24000)):
            cell = cells.get((seq, budget))
            if not cell:
                continue
            pts = [(DEGREES[m], gain(cell, m, value)) for m in DEGREES if m in cell]
            xs = [d for d, _ in pts]
            ax.errorbar(xs, [g[0] for _, g in pts], yerr=[[g[0] - g[1] for _, g in pts], [g[2] - g[0] for _, g in pts]],
                        marker="o", capsize=3, label=f"{seq}, {budget}", color=f"C{i}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(list(DEGREES.values()))
        ax.set_xlabel("degree of the centre path")
        ax.set_title(title)
    axes[0].set_ylabel("PSNR gain over M0 (dB)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "degree.png", dpi=130)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--roots", type=Path, nargs="+", default=[Path("runs/long_test"), Path("runs/poly")])
    p.add_argument("--out", type=Path, default=Path("runs/poly"))
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore", RuntimeWarning)
    cells = load(*a.roots)
    text = unseen_tables(cells) + "\n\n" + degree_tables(cells)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tables.md").write_text(text + "\n", encoding="utf-8")
    plot(cells, a.out)
    print(text)


if __name__ == "__main__":
    main()
