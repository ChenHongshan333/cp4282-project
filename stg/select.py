"""Choose each model's settings on the validation clip.

For every model and every combination of its swept settings (rate of v, and s_min or acc_ratio
where the model has them), the PSNR is averaged over the 4 budgets x 2 strides of the validation
stage, and the combination with the highest mean is chosen. Writes ``runs/validation/chosen.json``,
which ``stg.sweep`` reads for the test runs, and ``runs/validation/selection.md``.

    python -m stg.select
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stg.model import MODELS


def settings_of(m) -> tuple:
    spec = MODELS[m["model"]]
    return (m["lr_vel"], m["s_min"] if spec.shrink else None, m["acc_ratio"] if spec.accel else None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=Path("runs/validation"))
    a = p.parse_args(argv)
    cells = defaultdict(dict)          # (model, settings) -> {(budget, stride): psnr}
    for path in a.root.glob("*/metrics.json"):
        m = json.loads(path.read_text())
        cells[(m["model"], settings_of(m))][(m["budget"], m["stride"])] = m["psnr"]
    n_cells = max(len(c) for c in cells.values())
    chosen, lines = {}, ["| 方法 | 速度学习率 | s_min | acc_ratio | 8 种情况的平均 PSNR | 是否选中 |",
                         "|---|---|---|---|---|---|"]
    for model in MODELS:
        options = {s: c for (mo, s), c in cells.items() if mo == model}
        complete = {s: float(np.mean(list(c.values()))) for s, c in options.items() if len(c) == n_cells}
        if not complete:
            continue
        best = max(complete, key=complete.get)
        lr_vel, s_min, acc_ratio = best
        chosen[model] = {"lr_vel": lr_vel, "s_min": 0.1 if s_min is None else s_min,
                         "acc_ratio": 1.0 if acc_ratio is None else acc_ratio, "validation_psnr": complete[best]}
        for s, value in sorted(complete.items(), key=lambda kv: -kv[1]):
            cell = ["—" if x is None else str(x) for x in s]
            lines.append(f"| {model} | {cell[0]} | {cell[1]} | {cell[2]} | {value:.2f} | {'✓' if s == best else ''} |")
    (a.root / "chosen.json").write_text(json.dumps(chosen, indent=1))
    (a.root / "selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(chosen, indent=1))


if __name__ == "__main__":
    main()
