"""Analysis of plan A (PROGRESS.md): the gains over M0 on still clips, against the same sequence moving.

A still clip is frame 0 of a sequence repeated 10 times. The criterion written before the runs: M2
beats M0 on an image if the mean gain is above 0.3 dB and above the seed-to-seed range (the larger
of the two models' max - min PSNR over the seeds), at both budgets. Writes ``runs/still/tables.md``.

    python -m stg.analyze_still
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from stg.analyze_test import THRESHOLD, cells, gain, load, seed_range
from stg.data import SEQUENCES

MODELS = ("M1", "M2", "M3")
BUDGETS = (3000, 12000)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=Path("runs/still"))
    p.add_argument("--test-root", type=Path, default=Path("runs/test"))
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore", RuntimeWarning)
    still = cells(load(a.root))
    moving = cells(load(a.test_root))
    lines = ["| 照片 | 存储量 | M0 | M1 比 M0 | M2 比 M0 | M3 比 M0 | 种子波动 | 同一段视频在动时 M2 比 M0（stride 1） |",
             "|---|---|---|---|---|---|---|---|"]
    verdicts = []
    for sequence in SEQUENCES:
        passes = []
        for budget in BUDGETS:
            cell = still.get((f"still_{sequence}", 10, 1, budget))
            if not cell or any(m not in cell for m in ("M0",) + MODELS):
                continue
            m0 = sum(r["psnr"] for r in cell["M0"].values()) / len(cell["M0"])
            gains = {m: gain(cell, m, "psnr") for m in MODELS}
            rng = max(seed_range(cell, "M2"), seed_range(cell, "M0"))
            passes.append(gains["M2"][0] > THRESHOLD and gains["M2"][0] > rng)
            video = moving.get((sequence, 10, 1, budget))
            video_gain = f"{gain(video, 'M2', 'psnr')[0]:+.2f}" if video else "—"
            lines.append(f"| {sequence} | {budget} | {m0:.2f} | " +
                         " | ".join(f"{g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}]" for g in gains.values()) +
                         f" | {rng:.2f} | {video_gain} |")
        if passes:
            verdicts.append(f"- {sequence}：{'✓ 两个存储量都达标' if all(passes) else '✗ 未全部达标'}")
    text = "\n".join(["## 计划 A：静止照片（PSNR，dB；3 个种子平均 [最小, 最大]）", "", *lines, "",
                      "**按事先定好的标准（M2 比 M0 高 0.3 dB 以上，且大于种子波动）：**", "", *verdicts])
    (a.root / "tables.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
