"""Analysis of plan 2 (PROGRESS.md, steps 22 and 23): the gains over M0 of fully trained fits (30000
iterations, stride 3), against the gains of the test stage (3000 iterations) on the same clips; both
the mean over the 3 seeds, paired by seed, and for the fully trained fits the min and max.

The failure test of the proposal takes the seed-to-seed range of the fully trained fits themselves,
or, while a cell has one seed only, that of the test stage. Writes ``tables.md`` and ``rd_psnr.png``.

    python -m stg.analyze_long
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

from stg.analyze_test import BUDGETS, MODELS, THRESHOLD, cells, fmt, gain, load, seed_range
from stg.data import SEQUENCES

METRICS = {"psnr": "PSNR（dB）", "ssim": "SSIM", "vmaf": "VMAF", "psnr_in": "物体区域 PSNR", "psnr_out": "背景 PSNR"}


def table(long, short, metric):
    lines = [f"\n**{METRICS[metric]}：和 M0 的差距。练完（30000 步）：种子平均 [最小, 最大]；3000 步：种子平均**\n",
             "| 视频 | 存储量 | 种子数 | M0 练完 | M1 练完 | M1 3000 步 | M2 练完 | M2 3000 步 | M3 练完 | M3 3000 步 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for seq in SEQUENCES:
        for budget in BUDGETS:
            cell, old = long.get((seq, 10, 3, budget)), short.get((seq, 10, 3, budget))
            if not cell or any(m not in cell for m in MODELS):
                continue
            m0 = np.nanmean([r[metric] for r in cell["M0"].values()])
            row = [fmt(gain(cell, m, metric)) + " | " + (f"{gain(old, m, metric)[0]:+.2f}" if old else "—")
                   for m in MODELS[1:]]
            lines.append(f"| {seq} | {budget} | {len(cell['M0'])} | {m0:.2f} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def averages(long, short):
    """Mean gain in PSNR over the 4 budgets, per sequence."""
    lines = ["\n**PSNR：4 个存储量平均的差距（dB），练完 / 3000 步**\n",
             "| 视频 | M1 比 M0 | M2 比 M0 | M3 比 M0 |", "|---|---|---|---|"]
    for seq in SEQUENCES:
        row = []
        for m in MODELS[1:]:
            new = np.nanmean([gain(long[(seq, 10, 3, b)], m, "psnr")[0] for b in BUDGETS])
            old = np.nanmean([gain(short[(seq, 10, 3, b)], m, "psnr")[0] for b in BUDGETS])
            row.append(f"{new:+.2f} / {old:+.2f}")
        lines.append(f"| {seq} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def failure(long, short):
    lines = ["\n**事先定好的标准（比 M0 高 0.3 dB 以上，且大于种子波动；种子波动取练完实验自己的值）：**\n"]
    for model in MODELS[1:]:
        verdict, passes = [], []
        for seq in ("Jockey", "ShakeNDry"):
            beats = 0
            for budget in BUDGETS:
                cell, old = long.get((seq, 10, 3, budget)), short.get((seq, 10, 3, budget))
                if not cell or model not in cell:
                    continue
                source = cell if len(cell["M0"]) > 1 else old
                rng = max(seed_range(source, model), seed_range(source, "M0")) if source else 0.0
                g = gain(cell, model, "psnr")[0]
                beats += g > THRESHOLD and g > rng
            verdict.append(f"{seq} {beats}/4 个存储量达标")
            passes.append(beats >= 3)
        lines.append(f"- {model}：" + "，".join(verdict) + ("，**通过**" if all(passes) else "，**失败**"))
    return "\n".join(lines)


def plot(long, short, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(SEQUENCES), figsize=(4.2 * len(SEQUENCES), 3.8))
    for ax, seq in zip(axes, SEQUENCES):
        for i, model in enumerate(MODELS):
            colour = f"C{i}"
            for runs, style, label in ((long, "-", f"{model} 30k"), (short, ":", f"{model} 3k")):
                pts = [(b, np.mean([r["psnr"] for r in runs[(seq, 10, 3, b)][model].values()]))
                       for b in BUDGETS if (seq, 10, 3, b) in runs and model in runs[(seq, 10, 3, b)]]
                if pts:
                    ax.plot(*zip(*pts), style, marker="o", color=colour, label=label)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(BUDGETS, [f"{b // 1000}k" for b in BUDGETS])
        ax.set_title(f"{seq}, stride 3")
        ax.set_xlabel("stored numbers")
        ax.set_ylabel("PSNR (dB)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(root / "rd_psnr.png", dpi=130)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=Path("runs/long_test"))
    p.add_argument("--test-root", type=Path, default=Path("runs/test"))
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore", RuntimeWarning)
    long, short = cells(load(a.root)), cells(load(a.test_root))
    text = "\n".join(["## 练完条件下的精简版正式实验", averages(long, short), *(table(long, short, m) for m in METRICS),
                      failure(long, short)])
    (a.root / "tables.md").write_text(text + "\n", encoding="utf-8")
    plot(long, short, a.root)
    print(text)


if __name__ == "__main__":
    main()
