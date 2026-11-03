"""Figures and tables of the project page and the report (PROGRESS.md, step 33), from ``docs/data/runs.json``
(``stg.export_site``). Trains nothing. Every gain over M0 is paired by seed (same clip, budget and seed)
and given as the mean over the seeds with the min and max, as in the analysis scripts.

Writes ``docs/figures/<name>.png`` and ``.pdf``, and ``docs/data/tables.json``: for every table a head
and rows of formatted cells.

    python -m stg.figures
    python -m stg.figures --crops        # also the crops of rendered frames (needs the GPU and data/)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEQUENCES = ("Jockey", "ShakeNDry", "YachtRide", "Bosphorus")
BUDGETS = (3000, 6000, 12000, 24000)
MODELS = ("M0", "M1", "M2", "M3")
LABEL = {"M0": "M0 straight (baseline)", "M1": "M1 curved", "M2": "M2 shrinking", "M3": "M3 both"}
# M0 is the reference and wears neutral ink; M1-M3 take the first three slots of the validated
# categorical palette (dataviz skill, references/palette.md), in a fixed order.
COLOUR = {"M0": "#52514e", "M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a"}
SEQ_COLOUR = {"Jockey": "#2a78d6", "ShakeNDry": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
THRESHOLD = 0.3
MINUS = "−"


# --- data ----------------------------------------------------------------------------------------

class Runs:
    def __init__(self, records):
        self.records = records

    def where(self, **kw):
        return Runs([r for r in self.records if all(r.get(k) == v for k, v in kw.items())])

    def seeds(self, model, value):
        return {r["seed"]: r[value] for r in self.records if r["model"] == model and r.get(value) is not None}

    def mean(self, model, value="psnr"):
        v = list(self.seeds(model, value).values())
        return float(np.mean(v)) if v else np.nan

    def gain(self, model, value="psnr", base="M0"):
        a, b = self.seeds(model, value), self.seeds(base, value)
        g = np.array([a[s] - b[s] for s in sorted(set(a) & set(b))], float)
        return (float(g.mean()), float(g.min()), float(g.max())) if len(g) else (np.nan,) * 3

    def seed_range(self, model, value="psnr"):
        v = np.array(list(self.seeds(model, value).values()), float)
        return float(v.max() - v.min()) if len(v) else np.nan


def signed(x, digits=2):
    if x is None or np.isnan(x):
        return "—"
    s = f"{x:+.{digits}f}"
    return s.replace("-", MINUS)


def ranged(g, digits=2):
    if np.isnan(g[0]):
        return "—"
    return f"{signed(g[0], digits)} [{signed(g[1], digits)}, {signed(g[2], digits)}]"


def plain(x, digits=2):
    return "—" if x is None or np.isnan(x) else f"{x:.{digits}f}"


# --- tables --------------------------------------------------------------------------------------

def tables(runs: Runs) -> dict:
    t = {}
    test, long = runs.where(stage="test", frames=10, start=0), runs.where(stage="long_test")

    # Step 14: the gains at 3000 iterations, averaged over the budgets, by sequence and stride.
    rows = []
    for seq in SEQUENCES:
        for stride in (1, 3):
            cell = test.where(sequence=seq, stride=stride)
            rows.append([seq, str(stride)] + [signed(np.mean([cell.where(budget=b).gain(m)[0] for b in BUDGETS]))
                                              for m in MODELS[1:]])
    t["test_avg"] = {"head": ["Sequence", "Stride", "M1 − M0", "M2 − M0", "M3 − M0"], "rows": rows}

    # Step 14: inside and outside the object mask.
    rows = []
    for seq in SEQUENCES:
        for stride in (1, 3):
            cell = test.where(sequence=seq, stride=stride)
            rows.append([seq, str(stride)] + [
                f"{signed(np.mean([cell.where(budget=b).gain(m, 'psnr_in')[0] for b in BUDGETS]))} / "
                f"{signed(np.mean([cell.where(budget=b).gain(m, 'psnr_out')[0] for b in BUDGETS]))}" for m in MODELS[1:]])
    t["test_inout"] = {"head": ["Sequence", "Stride", "M1: object / background", "M2: object / background",
                                "M3: object / background"], "rows": rows}

    # Step 14: SSIM and VMAF.
    rows = []
    for seq in SEQUENCES:
        for stride in (1, 3):
            cell = test.where(sequence=seq, stride=stride)
            ssim = [signed(np.mean([cell.where(budget=b).gain(m, "ssim")[0] for b in BUDGETS])) for m in MODELS[1:]]
            vmaf = [signed(np.mean([cell.where(budget=b).gain(m, "vmaf")[0] for b in BUDGETS]), 1) for m in MODELS[1:]]
            rows.append([seq, str(stride), " / ".join(ssim), " / ".join(vmaf)])
    t["test_other"] = {"head": ["Sequence", "Stride", "SSIM: M1 / M2 / M3 − M0", "VMAF: M1 / M2 / M3 − M0"], "rows": rows}

    t["test_fail"] = failure_table(test)
    t["long_fail"] = failure_table(long)

    # Step 14: the 30-frame Jockey clip.
    rows = []
    clip = runs.where(stage="test", frames=30)
    for b in sorted({r["budget"] for r in clip.records}):
        cell = clip.where(budget=b)
        rows.append([f"{b:,}", plain(cell.mean("M0"))] + [signed(cell.gain(m)[0]) for m in MODELS[1:]])
    t["long_clip"] = {"head": ["Budget", "M0 PSNR", "M1 − M0", "M2 − M0", "M3 − M0"], "rows": rows}

    # Step 14: costs at budget 24000, both strides.
    rows = []
    cell = test.where(budget=24000)
    base = np.mean([r["seconds_per_iteration"] for r in cell.records if r["model"] == "M0"])
    for m in MODELS:
        s = np.mean([r["seconds_per_iteration"] for r in cell.records if r["model"] == m])
        f = np.mean([r["render_seconds_per_frame"] for r in cell.records if r["model"] == m])
        rows.append([m, f"{1000 * s:.1f} ms" + ("" if m == "M0" else f" ({signed(100 * (s / base - 1), 0)}%)"),
                     f"{1000 * f:.2f} ms"])
    t["cost"] = {"head": ["Model", "Time per training step", "Time to render a frame"], "rows": rows}

    # Steps 22 and 23: fully trained, per cell, with the 3000-iteration gain beside it.
    rows = []
    for seq in SEQUENCES:
        for b in BUDGETS:
            cell, old = long.where(sequence=seq, budget=b), test.where(sequence=seq, stride=3, budget=b)
            rows.append([seq, f"{b:,}", plain(cell.mean("M0"))]
                        + [x for m in MODELS[1:] for x in (ranged(cell.gain(m)), signed(old.gain(m)[0]))])
    t["long_cells"] = {"head": ["Sequence", "Budget", "M0 trained", "M1 trained", "M1 3k", "M2 trained", "M2 3k",
                                "M3 trained", "M3 3k"], "rows": rows}
    rows = []
    for seq in SEQUENCES:
        cell, old = long.where(sequence=seq), test.where(sequence=seq, stride=3)
        rows.append([seq] + [f"{signed(np.mean([cell.where(budget=b).gain(m)[0] for b in BUDGETS]))} / "
                             f"{signed(np.mean([old.where(budget=b).gain(m)[0] for b in BUDGETS]))}" for m in MODELS[1:]])
    t["long_avg"] = {"head": ["Sequence", "M1 − M0", "M2 − M0", "M3 − M0"], "rows": rows}
    rows = []
    for seq in SEQUENCES:
        cell = long.where(sequence=seq)
        rows.append([seq] + [f"{signed(np.mean([cell.where(budget=b).gain(m, 'psnr_in')[0] for b in BUDGETS]))} / "
                             f"{signed(np.mean([cell.where(budget=b).gain(m, 'psnr_out')[0] for b in BUDGETS]))}"
                             for m in MODELS[1:]])
    t["long_inout"] = {"head": ["Sequence", "M1: object / background", "M2: object / background",
                                "M3: object / background"], "rows": rows}

    # Step 16: still photographs.
    rows = []
    still = runs.where(stage="still")
    for seq in SEQUENCES:
        for b in (3000, 12000):
            cell, moving = still.where(sequence=f"still_{seq}", budget=b), test.where(sequence=seq, stride=1, budget=b)
            rows.append([seq, f"{b:,}", plain(cell.mean("M0"))] + [ranged(cell.gain(m)) for m in MODELS[1:]]
                        + [signed(moving.gain("M2")[0])])
    t["still"] = {"head": ["Photograph", "Budget", "M0", "M1 − M0", "M2 − M0", "M3 − M0", "M2 − M0 on the moving clip"],
                  "rows": rows}

    # Step 25: fitted frames and frames between, M0.
    rows = []
    for seq in SEQUENCES:
        for b in BUDGETS:
            cell = long.where(sequence=seq, budget=b)
            rows.append([seq, f"{b:,}"] + [plain(cell.mean("M0", v)) for v in ("psnr", "unseen", "psnr_in", "unseen_in")])
    t["unseen_m0"] = {"head": ["Sequence", "Budget", "Fitted frames", "Frames between", "Object, fitted", "Object, between"],
                      "rows": rows}
    rows = []
    for seq in SEQUENCES:
        cell = long.where(sequence=seq)
        rows.append([seq] + [f"{signed(np.mean([cell.where(budget=b).gain(m)[0] for b in BUDGETS]))} → "
                             f"{signed(np.mean([cell.where(budget=b).gain(m, 'unseen')[0] for b in BUDGETS]))}"
                             for m in MODELS[1:]])
    t["unseen_gain"] = {"head": ["Sequence", "M1 − M0", "M2 − M0", "M3 − M0"], "rows": rows}
    rows = []
    for seq in SEQUENCES:
        for b in BUDGETS:
            cell = long.where(sequence=seq, budget=b)
            rows.append([seq, f"{b:,}"] + [ranged(cell.gain(m, "unseen_in")) for m in MODELS[1:]]
                        + [signed(cell.gain("M1", "psnr_in")[0])])
    t["unseen_obj"] = {"head": ["Sequence", "Budget", "M1 − M0", "M2 − M0", "M3 − M0", "M1 − M0, object on fitted frames"],
                       "rows": rows}

    # Step 26: degree of the path.
    rows = []
    poly = Runs(long.where(stride=3).records + runs.where(stage="poly").records)
    for value, where in (("psnr", "fitted frames"), ("unseen", "frames between"), ("unseen_in", "frames between, object")):
        for seq in ("Jockey", "ShakeNDry"):
            for b in (6000, 24000):
                cell = poly.where(sequence=seq, budget=b)
                rows.append([seq, f"{b:,}", where, plain(cell.mean("M0", value))]
                            + [ranged(cell.gain(m, value)) for m in ("M1", "D3", "D7")])
    t["poly"] = {"head": ["Sequence", "Budget", "Scored on", "M0 (degree 1)", "M1 (degree 2)", "D3 (degree 3)", "D7 (degree 7)"],
                 "rows": rows}

    # Step 30: arcs.
    rows = []
    arcs = runs.where(stage="arc_slow")
    for angle in (0, 90, 180, 270, 360):
        cell = arcs.where(sequence=f"arcs{angle}")
        row = [f"{angle}°"]
        for value in ("psnr_in", "unseen_in"):
            for b in (6000, 24000):
                c = cell.where(budget=b)
                row.append(f"{plain(c.mean('M0', value))}; {ranged(c.gain('M1', value))}")
        rows.append(row)
    t["arc"] = {"head": ["Turn of the path", "Fitted frames, 6,000", "Fitted frames, 24,000", "Frames between, 6,000",
                         "Frames between, 24,000"], "rows": rows}

    # Step 32: the confirmatory test.
    rows, passed = [], 0
    conf = runs.where(stage="confirm")
    for seq, start, label in (("ShakeNDry", 30, "ShakeNDry from frame 30 (dog)"), ("Jockey", 120, "Jockey from frame 120"),
                              ("Jockey", 200, "Jockey from frame 200")):
        row = [label]
        for stride in (3, 5):
            for b in (6000, 24000):
                g = conf.where(sequence=seq, start=start, stride=stride, budget=b).gain("M1", "unseen_in")
                ok = g[1] > 0
                passed += ok
                row.append(f"{ranged(g)} {'✓' if ok else '✗'}")
        rows.append(row)
    t["confirm"] = {"head": ["Clip", "Stride 3, 6,000", "Stride 3, 24,000", "Stride 5, 6,000", "Stride 5, 24,000"],
                    "rows": rows, "passed": passed}
    rows = []
    for value, label in (("psnr_in", "Fitted frames, object"), ("unseen", "Frames between, whole frame"),
                         ("psnr", "Fitted frames, whole frame")):
        g = [conf.where(sequence=s, start=k, stride=st, budget=b).gain("M1", value)[0]
             for s, k in (("ShakeNDry", 30), ("Jockey", 120), ("Jockey", 200)) for st in (3, 5) for b in (6000, 24000)]
        rows.append([label, f"{signed(min(g))} to {signed(max(g))}"])
    for stride in (3, 5):
        g = [conf.where(sequence=s, start=k, stride=stride, budget=b).gain("M1", "unseen_in")[0]
             for s, k in (("ShakeNDry", 30), ("Jockey", 120), ("Jockey", 200)) for b in (6000, 24000)]
        rows.append([f"Frames between, object: mean over stride {stride}", signed(np.mean(g))])
    t["confirm_other"] = {"head": ["M1 − M0 on", "Range over the 12 cells"], "rows": rows}
    return t


def failure_table(runs: Runs) -> dict:
    """The failure test of the proposal: > 0.3 dB and > the seed-to-seed range at >= 3 of 4 budgets,
    on both Jockey and ShakeNDry at stride 3."""
    rows = []
    for m in MODELS[1:]:
        row, passes = [m], []
        for seq in ("Jockey", "ShakeNDry"):
            marks, beats = [], 0
            for b in BUDGETS:
                cell = runs.where(sequence=seq, stride=3, budget=b)
                g = cell.gain(m)[0]
                ok = g > THRESHOLD and g > max(cell.seed_range(m), cell.seed_range("M0"))
                beats += ok
                marks.append(f"{signed(g)} {'✓' if ok else '✗'}")
            row.append(", ".join(marks) + f" → {beats}/4")
            passes.append(beats >= 3)
        row.append("passes" if all(passes) else "fails")
        rows.append(row)
    return {"head": ["Model", "Jockey, stride 3 (budgets 3k, 6k, 12k, 24k)", "ShakeNDry, stride 3", "Verdict"], "rows": rows}


# --- figures -------------------------------------------------------------------------------------

def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.8, "lines.markersize": 4.5,
        "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 100, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04, "pdf.fonttype": 42})
    return plt


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=220, facecolor="white")
    fig.savefig(out / f"{name}.pdf", facecolor="white")
    print(f"{name}.png", flush=True)


def budget_axis(ax, budgets=BUDGETS):
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b // 1000}k" for b in budgets])
    ax.minorticks_off()
    ax.set_xlabel("storage budget (stored numbers)")


def errorline(ax, x, gains, **kw):
    g = np.array(gains)
    ax.errorbar(x, g[:, 0], yerr=[g[:, 0] - g[:, 1], g[:, 2] - g[:, 0]], capsize=2.5, elinewidth=1, **kw)


def fig_rd(runs: Runs, out: Path, plt):
    """PSNR against storage, after 3000 iterations and fully trained, stride 3."""
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.6), sharex=True)
    for row, (stage, title) in enumerate((("test", "3,000 training steps"), ("long_test", "fully trained (30,000 steps)"))):
        r = runs.where(stage=stage, stride=3, frames=10, start=0)
        for ax, seq in zip(axes[row], SEQUENCES):
            for m in MODELS:
                ax.plot(BUDGETS, [r.where(sequence=seq, budget=b).mean(m) for b in BUDGETS], marker="o",
                        color=COLOUR[m], label=LABEL[m], zorder=3 if m == "M0" else 2)
            budget_axis(ax)
            ax.set_title(f"{seq}, {title}", loc="left")
            if row == 0:
                ax.set_xlabel("")
        axes[row][0].set_ylabel("PSNR on fitted frames (dB)")
    axes[0][0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, out, "rd")
    plt.close(fig)


def fig_gain_budget(runs: Runs, out: Path, plt):
    """Gain over M0 against storage: 3000 iterations (dashed) against fully trained (solid)."""
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.9), sharey=True)
    test, long = runs.where(stage="test", stride=3, frames=10, start=0), runs.where(stage="long_test")
    for ax, seq in zip(axes, SEQUENCES):
        ax.axhspan(-THRESHOLD, THRESHOLD, color="#f1f0ec", zorder=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        for m in MODELS[1:]:
            ax.plot(BUDGETS, [test.where(sequence=seq, budget=b).gain(m)[0] for b in BUDGETS], ls="--", marker="o",
                    mfc="white", color=COLOUR[m], lw=1.3)
            errorline(ax, BUDGETS, [long.where(sequence=seq, budget=b).gain(m) for b in BUDGETS], marker="o",
                      color=COLOUR[m], label=LABEL[m])
        budget_axis(ax)
        ax.set_title(seq, loc="left")
    axes[0].set_ylabel("PSNR gain over M0 (dB)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLOUR[m], marker="o", label=LABEL[m]) for m in MODELS[1:]]
    handles += [Line2D([], [], color=MUTED, ls="--", marker="o", mfc="white", label="3,000 steps"),
                Line2D([], [], color=MUTED, marker="o", label="fully trained (bars: min–max over 3 seeds)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "gain_budget")
    plt.close(fig)


def trace_mean(records):
    """Mean over the seeds of the PSNR trace, by step, and the mean seconds at each step. The first
    point of a trace is the random initialisation, before any step, and is left out."""
    steps = [p[0] for p in records[0]["trace"][1:]]
    psnr = np.mean([[p[1] for p in r["trace"][1:]] for r in records], axis=0)
    secs = np.mean([[p[2] for p in r["trace"][1:]] for r in records], axis=0)
    return np.array(steps), psnr, secs


def fig_training(runs: Runs, out: Path, plt):
    """PSNR against training step and training time, fully trained runs, and the gain over M0 by step."""
    long = runs.where(stage="long_test")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    cell = long.where(sequence="Jockey", budget=6000)
    for m in MODELS:
        steps, psnr, secs = trace_mean([r for r in cell.records if r["model"] == m])
        axes[0].plot(steps / 1000, psnr, color=COLOUR[m], label=LABEL[m])
        axes[1].plot(secs / 60, psnr, color=COLOUR[m], label=LABEL[m])
    test = runs.where(stage="test", stride=3, sequence="Jockey", budget=6000, frames=10, start=0)
    for m in MODELS:
        axes[0].plot(3, test.mean(m), marker="D", color=COLOUR[m], mec="white", ms=6, ls="none")
    axes[0].set_xlabel("training step (thousands)")
    axes[0].set_ylabel("PSNR on fitted frames (dB)")
    axes[0].set_title("Jockey, budget 6,000: by step", loc="left")
    axes[1].set_xlabel("training time (minutes, RTX 4060 laptop)")
    axes[1].set_title("by training time", loc="left")
    axes[0].set_ylim(20, None)
    axes[1].set_ylim(20, None)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLOUR[m], label=LABEL[m]) for m in MODELS]
    handles.append(Line2D([], [], color=MUTED, marker="D", ls="none", mec="white", ms=6, label="separate 3,000-step runs"))
    axes[0].legend(handles=handles, loc="lower right")
    # gain over M0 at each step, mean over the 4 sequences, 4 budgets and 3 seeds
    ax = axes[2]
    ax.axhspan(-THRESHOLD, THRESHOLD, color="#f1f0ec", zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    for m in MODELS[1:]:
        gains = []
        for seq in SEQUENCES:
            for b in BUDGETS:
                c = long.where(sequence=seq, budget=b)
                base = {r["seed"]: np.array([p[1] for p in r["trace"][1:]]) for r in c.records if r["model"] == "M0"}
                for r in c.records:
                    if r["model"] == m:
                        gains.append(np.array([p[1] for p in r["trace"][1:]]) - base[r["seed"]])
        steps = np.array([p[0] for p in long.records[0]["trace"][1:]])
        ax.plot(steps / 1000, np.mean(gains, axis=0), color=COLOUR[m], label=LABEL[m])
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("PSNR gain over M0 (dB)")
    ax.set_title("all 4 clips × 4 budgets × 3 seeds", loc="left")
    fig.tight_layout()
    save(fig, out, "training")
    plt.close(fig)


def fig_still(runs: Runs, out: Path, plt):
    """Gains on still photographs against the same clips in motion (3000 iterations, stride 1)."""
    still, test = runs.where(stage="still"), runs.where(stage="test", stride=1, frames=10, start=0)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.8), sharey=True)
    width = 0.2
    for ax, b in zip(axes, (3000, 12000)):
        x = np.arange(len(SEQUENCES))
        ax.axhline(0, color=MUTED, lw=0.8)
        for i, m in enumerate(MODELS[1:]):
            g = np.array([still.where(sequence=f"still_{s}", budget=b).gain(m) for s in SEQUENCES])
            ax.bar(x + (i - 1) * width, g[:, 0], width * 0.9, color=COLOUR[m], label=f"{m} on the still photo")
            ax.errorbar(x + (i - 1) * width, g[:, 0], yerr=[g[:, 0] - g[:, 1], g[:, 2] - g[:, 0]], fmt="none",
                        ecolor=INK, elinewidth=0.8, capsize=2)
        moving = [test.where(sequence=s, budget=b).gain("M2")[0] for s in SEQUENCES]
        ax.plot(x, moving, ls="none", marker="D", ms=6, color="white", mec=INK, mew=1.4, label="M2 on the moving clip")
        ax.set_xticks(x)
        ax.set_xticklabels(SEQUENCES)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"budget {b:,}", loc="left")
    axes[0].set_ylabel("PSNR gain over M0 (dB)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, out, "still")
    plt.close(fig)


def fig_degree(runs: Runs, out: Path, plt):
    """Gain over M0 against the degree of the path, fitted frames and frames between."""
    poly = Runs(runs.where(stage="long_test").records + runs.where(stage="poly").records)
    degrees = {"M0": 1, "M1": 2, "D3": 3, "D7": 7}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    for ax, (value, title) in zip(axes, (("psnr", "fitted frames"), ("unseen", "frames between (never fitted)"))):
        ax.axhline(0, color=MUTED, lw=0.8)
        for seq in ("Jockey", "ShakeNDry"):
            for b, ls in ((6000, "--"), (24000, "-")):
                cell = poly.where(sequence=seq, budget=b)
                g = [(0.0, 0.0, 0.0) if m == "M0" else cell.gain(m, value) for m in degrees]
                errorline(ax, list(degrees.values()), g, marker="o", ls=ls, color=SEQ_COLOUR[seq],
                          mfc="white" if b == 6000 else SEQ_COLOUR[seq], label=f"{seq}, budget {b:,}")
        ax.set_xticks([1, 2, 3, 7])
        ax.set_xticklabels(["1\nM0", "2\nM1", "3\nD3", "7\nD7"])
        ax.set_xlabel("degree of the path")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("PSNR gain over M0 (dB)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, out, "degree")
    plt.close(fig)


def fig_arc(runs: Runs, out: Path, plt):
    arcs = runs.where(stage="arc_slow")
    angles = (0, 90, 180, 270, 360)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    for ax, (value, title) in zip(axes, (("psnr_in", "ball, fitted frames"), ("unseen_in", "ball, frames between"))):
        ax.axhline(0, color=MUTED, lw=0.8)
        for b, ls, colour in ((6000, "--", "#86b6ef"), (24000, "-", "#2a78d6")):
            errorline(ax, angles, [arcs.where(sequence=f"arcs{a}", budget=b).gain("M1", value) for a in angles],
                      marker="o", ls=ls, color=colour, label=f"budget {b:,}")
        ax.set_xticks(angles)
        ax.set_xticklabels([f"{a}°" for a in angles])
        ax.set_xlabel("how far the path turns (0° = straight line)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("PSNR gain of M1 over M0 (dB)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, out, "arc")
    plt.close(fig)


def fig_unseen(runs: Runs, out: Path, plt):
    long = runs.where(stage="long_test")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
    ax = axes[0]
    for seq, colour in zip(SEQUENCES, ("#2a78d6", "#eb6834", "#1baf7a", "#52514e")):
        gap = [long.where(sequence=seq, budget=b).mean("M0", "unseen") - long.where(sequence=seq, budget=b).mean("M0")
               for b in BUDGETS]
        ax.plot(BUDGETS, gap, marker="o", color=colour, label=seq)
    ax.axhline(0, color=MUTED, lw=0.8)
    budget_axis(ax)
    ax.set_ylabel("frames between − fitted frames (dB)")
    ax.set_title("M0: how much worse the frames between are", loc="left")
    ax.legend(loc="lower left")
    ax = axes[1]
    ax.axhline(0, color=MUTED, lw=0.8)
    for seq, colour in zip(SEQUENCES, ("#2a78d6", "#eb6834", "#1baf7a", "#52514e")):
        errorline(ax, BUDGETS, [long.where(sequence=seq, budget=b).gain("M1", "unseen_in") for b in BUDGETS],
                  marker="o", color=colour, label=seq)
    budget_axis(ax)
    ax.set_ylabel("M1 − M0 (dB)")
    ax.set_title("M1 on the moving object, frames between", loc="left")
    fig.tight_layout()
    save(fig, out, "unseen")
    plt.close(fig)


def fig_confirm(runs: Runs, out: Path, plt):
    conf = runs.where(stage="confirm")
    clips = (("ShakeNDry", 30, "dog, from 30"), ("Jockey", 120, "horse, from 120"), ("Jockey", 200, "horse, from 200"))
    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    ax.axhline(0, color=MUTED, lw=0.8)
    x, labels = 0, []
    for i, (seq, start, name) in enumerate(clips):
        ax.text(4 * i + 1.5, -0.2, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color=INK)
        if i:
            ax.axvline(4 * i - 0.5, color=GRID, lw=1)
        for stride in (3, 5):
            for b in (6000, 24000):
                cell = conf.where(sequence=seq, start=start, stride=stride, budget=b)
                g = cell.gain("M1", "unseen_in")
                ok = g[1] > 0
                ax.errorbar([x], [g[0]], yerr=[[g[0] - g[1]], [g[2] - g[0]]], marker="o", ms=6, capsize=3,
                            color="#2a78d6" if ok else MUTED, mfc="#2a78d6" if ok else "white")
                w = cell.gain("M1", "unseen")[0]
                ax.plot([x + 0.22], [w], marker="s", ms=4.5, ls="none", color="#eb6834")
                labels.append(f"stride {stride}\n{b // 1000}k")
                x += 1
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("M1 − M0 on frames between (dB)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="#2a78d6", marker="o", ls="none", label="object: passes (all 3 seeds > 0)"),
                       Line2D([], [], color=MUTED, marker="o", mfc="white", ls="none", label="object: does not pass"),
                       Line2D([], [], color="#eb6834", marker="s", ls="none", label="whole frame (not part of the test)")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    save(fig, out, "confirm")
    plt.close(fig)


# --- crops of rendered frames --------------------------------------------------------------------

CROPS = {"Jockey": (255, 95, 140, 110), "ShakeNDry": (135, 90, 190, 150)}


def fig_crops(out: Path, plt, root=Path("runs"), data_root=Path("data/uvg"), device="cuda"):
    """The horse's legs and the dog: the frame, then M0 to M3 fully trained (budget 24000, seed 0),
    at a fitted frame and at the frame after it, which no model saw."""
    import warp as wp
    from stg import data
    from stg.model import SpaceTimeGaussians
    from stg.render import ClipRenderer
    wp.init()
    names = {"M0": "M0_B24000_seed0_smin0.1_acc1.0_lrv0.2", "M1": "M1_B24000_seed0_smin0.1_acc0.3",
             "M2": "M2_B24000_seed0_smin0.1_acc1.0_lrv0.1", "M3": "M3_B24000_seed0_smin0.1_acc0.3_lrv0.2"}
    k = 4                                    # clip time 4 = frame 12; the frame after it is 13
    fig, axes = plt.subplots(4, 5, figsize=(9.6, 6.0))
    for r, seq in enumerate(("Jockey", "ShakeNDry")):
        x, y, w, h = CROPS[seq]
        frames = data.load_clip(data_root, seq, 3 * k, 2, 1)
        for c, (t, frame, kind) in enumerate(((k, frames[0], "fitted"), (k + 1 / 3, frames[1], "between"))):
            row = 2 * r + c
            axes[row][0].imshow(frame[y:y + h, x:x + w])
            axes[row][0].set_ylabel(f"{seq}\nframe {3 * k + c} ({kind})", fontsize=8)
            for i, m in enumerate(MODELS):
                g = SpaceTimeGaussians.load(root / "long_test" / f"{seq}_k0_n10_s3_{names[m]}" / "params.npz")
                image = np.clip(ClipRenderer(1, *data.SIZE[::-1], device, times=[t]).render(g, 0.1)[0], 0, 1)
                crop = image[y:y + h, x:x + w]
                p = -10 * np.log10(np.mean((crop - frame[y:y + h, x:x + w]) ** 2))
                axes[row][i + 1].imshow(crop)
                axes[row][i + 1].set_xlabel(f"{p:.2f} dB in this crop", fontsize=7)
                if row == 0:
                    axes[row][i + 1].set_title(LABEL[m], fontsize=8)
            if row == 0:
                axes[row][0].set_title("frame", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    save(fig, out, "crops")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, default=Path("docs/data"))
    p.add_argument("--out", type=Path, default=Path("docs/figures"))
    p.add_argument("--crops", action="store_true")
    a = p.parse_args(argv)
    runs = Runs(json.loads((a.data / "runs.json").read_text()))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.data / "tables.json").write_text(json.dumps(tables(runs), ensure_ascii=False, indent=0), encoding="utf-8")
    plt = style()
    for fig in (fig_rd, fig_gain_budget, fig_training, fig_still, fig_degree, fig_arc, fig_unseen, fig_confirm):
        fig(runs, a.out, plt)
    if a.crops:
        fig_crops(a.out, plt)


if __name__ == "__main__":
    main()
