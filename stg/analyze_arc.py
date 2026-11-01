"""Analysis of plan 2e (PROGRESS.md): the gain of M1 over M0 on the arc clips against the bend of the
path, on the fitted frames and on the frames between them (``stg.unseen``), on the ball and on the
whole frame; with the paths of the Gaussians on the ball against the true path (``stg.paths arc``).
Writes ``tables.md`` and ``arc.png`` to ``--root``.

    python -m stg.analyze_arc
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from stg.analyze_extension import gain
from stg.analyze_test import fmt
from stg import data
from stg.data import ARC_ANGLES

BUDGETS = (6000, 24000)
VALUES = {"psnr_in": "训练用的帧，球", "unseen_in": "没见过的帧，球", "psnr": "训练用的帧，整个画面",
          "unseen": "没见过的帧，整个画面"}


def load(root):
    """(angle, budget) -> model -> seed -> {value name: number}."""
    out = defaultdict(lambda: defaultdict(dict))
    for path in root.glob("*/metrics.json"):
        m = json.loads(path.read_text())
        if not m["sequence"].startswith("arc") or "psnr_unseen_per_frame" not in m:
            continue
        out[(int(data.arc_style(m["sequence"])[1]), m["budget"])][m["model"]][m["seed"]] = {
            "psnr": m["psnr"], "unseen": float(np.mean(m["psnr_unseen_per_frame"])),
            "psnr_in": float(np.nanmean(m["psnr_in_mask_per_frame"])),
            "unseen_in": float(np.nanmean(m["psnr_unseen_in_mask_per_frame"]))}
    return out


def tables(cells):
    lines = ["## 合成视频：拐弯的好处随弯曲程度的变化（计划 2e）", ""]
    for value, title in VALUES.items():
        lines += [f"**{title}：M1 比 M0 高多少（dB），种子平均 [最小, 最大]；括号前是 M0 本身**", "",
                  "| 转过的角度 | " + " | ".join(f"存储量 {b}" for b in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
        for angle in ARC_ANGLES:
            row = []
            for budget in BUDGETS:
                cell = cells.get((angle, budget))
                if not cell or "M0" not in cell or "M1" not in cell:
                    row.append("—")
                    continue
                m0 = np.mean([r[value] for r in cell["M0"].values()])
                row.append(f"{m0:.2f}；{fmt(gain(cell, 'M1', value))}")
            lines.append(f"| {angle}° | " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def plot(cells, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (value, title) in zip(axes, (("psnr_in", "ball, fitted frames"), ("unseen_in", "ball, frames between"))):
        for i, budget in enumerate(BUDGETS):
            pts = [(a, gain(cells[(a, budget)], "M1", value)) for a in ARC_ANGLES if (a, budget) in cells]
            ax.errorbar([a for a, _ in pts], [g[0] for _, g in pts],
                        yerr=[[g[0] - g[1] for _, g in pts], [g[2] - g[0] for _, g in pts]],
                        marker="o", capsize=3, label=f"budget {budget}", color=f"C{i}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(ARC_ANGLES)
        ax.set_xlabel("bend of the path (degrees turned)")
        ax.set_title(title)
    axes[0].set_ylabel("PSNR gain of M1 over M0 (dB)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "arc.png", dpi=130)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=Path("runs/arc"))
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore", RuntimeWarning)
    cells = load(a.root)
    text = tables(cells)
    (a.root / "tables.md").write_text(text + "\n", encoding="utf-8")
    plot(cells, a.root)
    print(text)


if __name__ == "__main__":
    main()
