"""Analysis of plan 2g (PROGRESS.md): the gain of M1 over M0 on the frames between the fitted ones,
inside the object masks, on segments not used before, judged by the criterion fixed before the runs:
a cell (segment, stride, budget) passes when the gain is above 0 for every seed; 9 or more of the 12
cells passing means bending helps on the frames between, 3 or fewer means it does not. Writes
``tables.md`` to ``--root``.

    python -m stg.analyze_confirm
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
from stg.sweep import CONFIRM_CLIPS

STRIDES = (3, 5)
BUDGETS = (6000, 24000)
VALUES = {"unseen_in": "没见过的帧，物体区域（判断标准用这一项）", "psnr_in": "训练用的帧，物体区域",
          "unseen": "没见过的帧，整个画面", "psnr": "训练用的帧，整个画面"}


def load(root):
    """(sequence, start, stride, budget) -> model -> seed -> {value name: number}."""
    out = defaultdict(lambda: defaultdict(dict))
    for path in root.glob("*/metrics.json"):
        m = json.loads(path.read_text())
        if "psnr_unseen_in_mask_per_frame" not in m:
            continue
        out[(m["sequence"], m["start"], m["stride"], m["budget"])][m["model"]][m["seed"]] = {
            "psnr": m["psnr"], "unseen": float(np.mean(m["psnr_unseen_per_frame"])),
            "psnr_in": float(np.nanmean(m["psnr_in_mask_per_frame"])),
            "unseen_in": float(np.nanmean(m["psnr_unseen_in_mask_per_frame"]))}
    return out


def keys():
    return [(seq, start, stride, budget) for seq, start in CONFIRM_CLIPS for stride in STRIDES for budget in BUDGETS]


def tables(cells):
    lines = ["## 新片段上的正式检验（计划 2g）", ""]
    passed, done = 0, 0
    for value, title in VALUES.items():
        lines += [f"**{title}：M1 比 M0 高多少（dB），种子平均 [最小, 最大]；分号前是 M0 本身**", "",
                  "| 片段 | stride | 存储量 | M0；M1 比 M0 |" + (" 通过？ |" if value == "unseen_in" else ""),
                  "|---|---|---|---|" + ("---|" if value == "unseen_in" else "")]
        for k in keys():
            cell = cells.get(k)
            label = f"| {k[0]} 第 {k[1]} 帧起 | {k[2]} | {k[3]} |"
            if not cell or len(cell.get("M0", {})) < 3 or len(cell.get("M1", {})) < 3:
                lines.append(label + " — |" + (" — |" if value == "unseen_in" else ""))
                continue
            g = gain(cell, "M1", value)
            m0 = np.mean([r[value] for r in cell["M0"].values()])
            row = label + f" {m0:.2f}；{fmt(g)} |"
            if value == "unseen_in":
                done += 1
                passed += g[1] > 0
                row += " ✅ |" if g[1] > 0 else " ✗ |"
            lines.append(row)
        lines.append("")
    verdict = ("拐弯在没见过的帧上有真实的好处" if passed >= 9 else "没有好处" if passed <= 3 else "部分成立")
    lines += [f"**判断标准：** {passed} / {done} 个格子通过（3 个种子都 > 0）。"
              + (f"结论：{verdict}。" if done == 12 else "还没跑完。"), ""]
    for stride in STRIDES:
        g = [gain(cells[k], "M1", "unseen_in")[0] for k in keys() if k[2] == stride and k in cells and "M1" in cells[k]]
        lines.append(f"- stride {stride}：没见过的帧、物体区域的平均好处 {np.nanmean(g):+.2f} dB（{len(g)} 个格子）")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=Path("runs/confirm"))
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore", RuntimeWarning)
    text = tables(load(a.root))
    (a.root / "tables.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
