"""Collect ``runs/*/metrics.json`` into one CSV, and a summary over seeds (mean, min, max).

    python -m stg.aggregate runs            # writes runs/results.csv and runs/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = ("sequence", "start", "frames", "stride", "model", "budget", "s_min", "acc_ratio")
VALUES = ("psnr", "ssim", "psnr_in_mask", "psnr_out_mask", "gaussians", "stored_numbers", "bytes",
          "seconds_per_iteration", "render_seconds_per_frame")


def load(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        m["psnr_in_mask"] = float(np.nanmean(m["psnr_in_mask_per_frame"]))
        m["psnr_out_mask"] = float(np.nanmean(m["psnr_out_mask_per_frame"]))
        rows.append({k: m[k] for k in KEYS + ("seed",) + VALUES} | {"run": path.parent.name})
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in KEYS)].append(r)
    out = []
    for key, group in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        row = dict(zip(KEYS, key), seeds=len(group))
        for v in VALUES:
            values = np.array([g[v] for g in group], float)
            row[f"{v}_mean"], row[f"{v}_min"], row[f"{v}_max"] = values.mean(), values.min(), values.max()
        out.append(row)
    return out


def write(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path, nargs="?", default=Path("runs"))
    a = p.parse_args(argv)
    rows = load(a.root)
    write(a.root / "results.csv", rows)
    write(a.root / "summary.csv", summarise(rows))
    print(f"{len(rows)} runs -> {a.root / 'results.csv'}, {a.root / 'summary.csv'}")


if __name__ == "__main__":
    main()
