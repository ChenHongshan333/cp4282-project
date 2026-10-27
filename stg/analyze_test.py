"""Analysis of the test stage: gains over M0 at equal storage, and the failure tests of the proposal.

For every sequence, stride and budget, the gain of M1, M2 and M3 over M0 is paired by seed (same
seed, same clip, same budget), and reported as the mean and the min-max over the 3 seeds, for PSNR,
SSIM, VMAF and the PSNR inside and outside the object mask.

The failure test of the proposal (section 6): an extension fails if it does not beat M0 by more than
0.3 dB, and by more than the seed-to-seed range, at three or more of the four budgets, on both Jockey
and ShakeNDry at stride 3. Here "beats by more than the seed-to-seed range" means that the mean gain
is larger than the larger of the two models' ranges of PSNR over the seeds (max - min).

Writes ``tables.md``, ``summary.json`` and ``rd_*.png`` to the test directory.

    python -m stg.analyze_test
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = ("M0", "M1", "M2", "M3")
BUDGETS = (3000, 6000, 12000, 24000)
METRICS = {"psnr": "PSNR", "ssim": "SSIM", "vmaf": "VMAF", "psnr_in": "PSNR in object", "psnr_out": "PSNR outside object"}
THRESHOLD = 0.3


def load(root: Path):
    runs = {}
    for path in root.glob("*/metrics.json"):
        m = json.loads(path.read_text())
        row = {"psnr": m["psnr"], "ssim": m["ssim"], "vmaf": m.get("vmaf", np.nan),
               "psnr_in": float(np.nanmean(m["psnr_in_mask_per_frame"])),
               "psnr_out": float(np.nanmean(m["psnr_out_mask_per_frame"])),
               "mask_kind": m.get("mask_kind"), "seconds_per_iteration": m["seconds_per_iteration"],
               "render_seconds_per_frame": m["render_seconds_per_frame"]}
        runs[(m["sequence"], m["frames"], m["stride"], m["budget"], m["model"], m["seed"])] = row
    return runs


def cells(runs):
    """(sequence, frames, stride, budget) -> model -> seed -> row"""
    out = defaultdict(lambda: defaultdict(dict))
    for (seq, frames, stride, budget, model, seed), row in runs.items():
        out[(seq, frames, stride, budget)][model][seed] = row
    return out


def gain(cell, model, metric):
    """Mean, min and max over seeds of model - M0, paired by seed."""
    seeds = sorted(set(cell[model]) & set(cell["M0"]))
    g = np.array([cell[model][s][metric] - cell["M0"][s][metric] for s in seeds], float)
    return (float(np.nanmean(g)), float(np.nanmin(g)), float(np.nanmax(g))) if len(g) else (np.nan,) * 3


def seed_range(cell, model, metric="psnr"):
    v = np.array([r[metric] for r in cell[model].values()], float)
    return float(v.max() - v.min()) if len(v) else np.nan


def failure_test(table, model):
    """Pass/fail of the proposal's test for ``model`` against M0, on Jockey and ShakeNDry at stride 3."""
    result = {}
    for seq in ("Jockey", "ShakeNDry"):
        budgets = []
        for budget in BUDGETS:
            cell = table.get((seq, 10, 3, budget))
            if not cell or model not in cell:
                continue
            mean = gain(cell, model, "psnr")[0]
            rng = max(seed_range(cell, model), seed_range(cell, "M0"))
            budgets.append({"budget": budget, "gain": mean, "seed_range": rng,
                            "beats": bool(mean > THRESHOLD and mean > rng)})
        result[seq] = {"budgets": budgets, "passes": sum(b["beats"] for b in budgets) >= 3}
    result["passes"] = all(result[s]["passes"] for s in ("Jockey", "ShakeNDry"))
    return result


def fmt(g):
    return "—" if np.isnan(g[0]) else f"{g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}]"


def gain_tables(table, metric):
    lines = []
    keys = sorted({k[:3] for k in table}, key=lambda k: (k[1], ("Jockey", "ShakeNDry", "YachtRide", "Bosphorus").index(k[0]), k[2]))
    for seq, frames, stride in keys:
        lines += [f"\n**{seq}，{frames} 帧，stride {stride}**（{METRICS[metric]}）\n",
                  "| 存储量 | M0 | M1 比 M0 | M2 比 M0 | M3 比 M0 |", "|---|---|---|---|---|"]
        for budget in sorted({k[3] for k in table if k[:3] == (seq, frames, stride)}):
            cell = table[(seq, frames, stride, budget)]
            m0 = np.nanmean([r[metric] for r in cell["M0"].values()])
            lines.append(f"| {budget} | {m0:.2f} | " + " | ".join(fmt(gain(cell, m, metric)) for m in MODELS[1:]) + " |")
    return "\n".join(lines)


def overview(table, metric="psnr"):
    """Mean gain over all budgets, per sequence and stride (10-frame clips)."""
    lines = [f"| 视频 | stride | M1 比 M0 | M2 比 M0 | M3 比 M0 |", "|---|---|---|---|---|"]
    for seq in ("Jockey", "ShakeNDry", "YachtRide", "Bosphorus"):
        for stride in (1, 3):
            gs = [[gain(table[(seq, 10, stride, b)], m, metric)[0] for b in BUDGETS if (seq, 10, stride, b) in table]
                  for m in MODELS[1:]]
            if not gs[0]:
                continue
            lines.append(f"| {seq} | {stride} | " + " | ".join(f"{np.nanmean(g):+.2f}" for g in gs) + " |")
    return "\n".join(lines)


def plot(table, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted({k[:3] for k in table})
    fig, axes = plt.subplots(2, (len(keys) + 1) // 2, figsize=(4.2 * ((len(keys) + 1) // 2), 7), squeeze=False)
    for ax, (seq, frames, stride) in zip(axes.ravel(), keys):
        budgets = sorted({k[3] for k in table if k[:3] == (seq, frames, stride)})
        for model in MODELS:
            vals = [[r["psnr"] for r in table[(seq, frames, stride, b)][model].values()] for b in budgets]
            if not all(vals):           # a clip whose runs are not all done yet
                continue
            ax.plot(budgets, [np.mean(v) for v in vals], marker="o", label=model)
            ax.fill_between(budgets, [min(v) for v in vals], [max(v) for v in vals], alpha=0.2)
        ax.set_xscale("log")
        ax.set_xticks(budgets, [f"{b // 1000}k" for b in budgets])
        ax.set_title(f"{seq}, {frames} frames, stride {stride}")
        ax.set_xlabel("stored numbers")
        ax.set_ylabel("PSNR (dB)")
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(root / "rd_psnr.png", dpi=130)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=Path("runs/test"))
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore", RuntimeWarning)     # empty cells while runs are still missing
    runs = load(a.root)
    table = cells(runs)
    tests = {m: failure_test(table, m) for m in MODELS[1:]}
    kinds = defaultdict(int)
    for r in runs.values():
        kinds[r["mask_kind"]] += 1
    text = [f"# 正式实验结果（{len(runs)} 个实验；mask：{dict(kinds)}）",
            "\n## 各视频平均（4 个存储量的平均，PSNR，dB）\n", overview(table),
            "\n## 各视频平均：物体区域内的 PSNR\n", overview(table, "psnr_in"),
            "\n## 各视频平均：物体区域外的 PSNR\n", overview(table, "psnr_out"),
            "\n## 事先写好的失败标准（Jockey 和 ShakeNDry，stride 3）\n"]
    for model, t in tests.items():
        text.append(f"**{model}：{'通过（有效）' if t['passes'] else '未通过（按标准算失败）'}**\n")
        for seq in ("Jockey", "ShakeNDry"):
            text.append(f"- {seq}：" + "，".join(
                f"{b['budget']}：{b['gain']:+.2f}（种子波动 {b['seed_range']:.2f}）{'✓' if b['beats'] else '✗'}"
                for b in t[seq]["budgets"]))
        text.append("")
    for metric in METRICS:
        text += [f"\n## 详细：{METRICS[metric]}（3 个种子的平均 [最小, 最大]）", gain_tables(table, metric)]
    (a.root / "tables.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (a.root / "summary.json").write_text(json.dumps({"failure_tests": tests, "runs": len(runs)}, indent=1))
    plot(table, a.root)
    print("\n".join(text[:6 + 3 * len(tests) + 6]))


if __name__ == "__main__":
    main()
