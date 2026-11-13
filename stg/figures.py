"""Figures and tables of the project page and the report (PROGRESS.md, step 33), from ``docs/data/runs.json``
(``stg.export_site``). Trains nothing. Every gain over M0 is paired by seed (same clip, budget and seed)
and given as the mean over the seeds with the min and max, as in the analysis scripts.

Writes ``docs/figures/<name>.png`` and ``.pdf``, and ``docs/data/tables.json``: for every table a head
and rows of formatted cells.

    python -m stg.figures
    python -m stg.figures --crops        # also the figures made from video frames (needs data/; crops need the GPU)
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
PASS, FAIL = "#1a7f4b", "#b3261e"     # the green and red of the page (style.css --good, --bad)
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
    t["test_fail_short"] = failure_summary(test)
    t["long_fail_short"] = failure_summary(long)

    # Storage: Gaussians per budget.
    t["budget"] = {"head": ["Model", "Numbers per Gaussian", "Gaussians at 3,000", "6,000", "12,000", "24,000"],
                   "rows": [[m, str(n)] + [f"{b // n:,}" for b in BUDGETS]
                            for m, n in (("M0, M2", 12), ("M1, M3", 14), ("D3", 16), ("D7", 24))]}

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
    # The same, one condition per table, for the main text.
    for name, source in (("long_avg_trained", long), ("test_avg_s3", test.where(stride=3))):
        t[name] = {"head": ["Sequence", "M1 curved", "M2 shrinking", "M3 both"],
                   "rows": [[seq] + [signed(np.mean([source.where(sequence=seq, budget=b).gain(m)[0] for b in BUDGETS]))
                                     for m in MODELS[1:]] for seq in SEQUENCES]}
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
    # The mean and the pass mark only, for the main text.
    t["confirm_short"] = {"head": t["confirm"]["head"],
                          "rows": [[r[0]] + [c.split(" [")[0] + " " + c[-1] for c in r[1:]] for r in rows]}
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


def failure_summary(runs: Runs) -> dict:
    """The failure test in short: at how many of the 4 budgets each model beats M0 by the margin."""
    rows = []
    for m in MODELS[1:]:
        row, passes = [LABEL[m]], []
        for seq in ("Jockey", "ShakeNDry"):
            beats = 0
            for b in BUDGETS:
                cell = runs.where(sequence=seq, stride=3, budget=b)
                g = cell.gain(m)[0]
                beats += g > THRESHOLD and g > max(cell.seed_range(m), cell.seed_range("M0"))
            row.append(f"{beats} of 4")
            passes.append(beats >= 3)
        row.append("passes" if all(passes) else "fails")
        rows.append(row)
    return {"head": ["Model", "Jockey", "ShakeNDry", "Verdict"], "rows": rows}


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
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.4), sharey=True)
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
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
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
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
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
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
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
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "training")
    plt.close(fig)


def fig_still(runs: Runs, out: Path, plt):
    """Gains of M1 and M2 over M0 on still photographs and on the same clips in motion.

    Both use 3000 iterations, the same budget and the same seeds; the moving clip is the 10-frame
    test clip at stride 1. Each gain is model - M0 paired by seed: mean, lowest and highest of 3 seeds.
    """
    iters = 3000
    still = runs.where(stage="still", iterations=iters)
    test = runs.where(stage="test", stride=1, frames=10, start=0, iterations=iters)
    budgets, models = (3000, 12000), ("M1", "M2")
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.0), sharey=True)
    width, x = 0.36, np.arange(len(SEQUENCES))
    for row, m in enumerate(models):
        for col, b in enumerate(budgets):
            ax = axes[row, col]
            ax.axhline(0, color=MUTED, lw=0.8)
            for k, (cells, label, solid) in enumerate((
                    ([still.where(sequence=f"still_{s}", budget=b) for s in SEQUENCES], "still photograph", False),
                    ([test.where(sequence=s, budget=b) for s in SEQUENCES], "moving clip (stride 1)", True))):
                for c in cells:     # three seeds of both models, paired by seed
                    if sorted(c.seeds(m, "psnr")) != sorted(c.seeds("M0", "psnr")) or len(c.seeds(m, "psnr")) != 3:
                        raise ValueError(f"unpaired seeds for {m} at budget {b}")
                g = np.array([c.gain(m) for c in cells])
                pos = x + (k - 0.5) * width
                ax.bar(pos, g[:, 0], width * 0.9, color=COLOUR[m] if solid else "white", edgecolor=COLOUR[m],
                       hatch=None if solid else "////", lw=1.0, label=label)
                ax.errorbar(pos, g[:, 0], yerr=[g[:, 0] - g[:, 1], g[:, 2] - g[:, 0]], fmt="none",
                            ecolor=INK, elinewidth=0.8, capsize=2)
            ax.set_xticks(x)
            ax.set_xticklabels(SEQUENCES)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{LABEL[m]}, budget {b:,}", loc="left")
            if col == 0:
                ax.set_ylabel(f"{m} {MINUS} M0 PSNR (dB)")
            else:
                ax.legend(loc="upper right")
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
    """The confirmatory test (step 32): M1 - M0 on the frames between, the moving object above and the
    whole frame below, on the same scale. A dot is the mean of the 3 seeds, the bar their range."""
    conf = runs.where(stage="confirm")
    clips = (("ShakeNDry", 30, "dog, from frame 30"), ("Jockey", 120, "horse, from frame 120"), ("Jockey", 200, "horse, from frame 200"))
    cells = [(seq, start, stride, b) for seq, start, _ in clips for stride in (3, 5) for b in (6000, 24000)]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.8), sharex=True, sharey=True)
    for ax, (value, title) in zip(axes, (("unseen_in", "moving object"), ("unseen", "whole frame"))):
        ax.axhline(0, color=INK, lw=0.9)
        gains = [conf.where(sequence=seq, start=start, stride=stride, budget=b).gain("M1", value) for seq, start, stride, b in cells]
        for x, g in enumerate(gains):
            if value == "unseen_in":        # the scored region: does the cell pass (all 3 seeds above zero)?
                colour = PASS if g[1] > 0 else FAIL
                face = colour if g[1] > 0 else "white"
            else:                           # the whole frame is not part of the test
                colour, face = MUTED, MUTED
            ax.errorbar([x], [g[0]], yerr=[[g[0] - g[1]], [g[2] - g[0]]], marker="o", ms=5.5, ls="none", capsize=2.5,
                        elinewidth=1, color=colour, mfc=face, mew=1.4)
        for i in (1, 2):
            ax.axvline(4 * i - 0.5, color=GRID, lw=1)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{title}: M1 − M0 on the frames between (dB)", loc="left")
        if value == "unseen_in":
            n = sum(g[1] > 0 for g in gains)
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([], [], color=PASS, marker="o", ls="none", label=f"all 3 seeds above zero ({n} of {len(cells)} cells)"),
                               Line2D([], [], color=FAIL, marker="o", mfc="white", mew=1.4, ls="none", label="not all 3 seeds above zero")],
                      loc="upper right", ncol=2)
    axes[1].set_xticks(range(len(cells)))
    axes[1].set_xticklabels([f"stride {stride}\n{b // 1000}k" for _, _, stride, b in cells], fontsize=7)
    for i, (_, _, name) in enumerate(clips):
        axes[1].text(4 * i + 1.5, -0.32, name, transform=axes[1].get_xaxis_transform(), ha="center", va="top", fontsize=8, color=INK)
    fig.tight_layout()
    save(fig, out, "confirm")
    plt.close(fig)


def fig_arc_motion(out: Path, plt):
    """The five arc clips played frame by frame, side by side and in step (``arc_motion.gif``), with a
    still of frame 0 for the report (``arc_paths.png/.pdf``). The ball and its stripes are the pixels
    that ``stg.data.arc_clip`` draws (style ``arcs``); for display they are put over a light background
    instead of the clip's own. The grey line is the true path of the centre, the blue dots the fitted
    frames of stride 3 (0, 3, ..., 27)."""
    import io
    from PIL import Image
    from stg.data import ARC_ANGLES, ARC_FRAMES, ARC_STYLES, arc_centres, arc_clip
    style = ARC_STYLES["arcs"]
    times = np.arange(ARC_FRAMES + 1)
    light = np.array([0.984, 0.980, 0.973])
    balls, paths = {}, {}
    for angle in ARC_ANGLES:
        frames, masks = arc_clip(f"arcs{angle}", 0, len(times), 1)
        # the clip's background: at each pixel, the first frame in which the ball is not there
        first_free = np.argmax(~masks, axis=0)
        background = np.take_along_axis(frames, first_free[None, ..., None].repeat(3, -1), axis=0)[0]
        centres = arc_centres(angle, times, speed=style["speed"])
        yy, xx = np.mgrid[0:frames.shape[1], 0:frames.shape[2]] + 0.5
        shown = []
        for t, (cx, cy) in enumerate(centres):
            # the same soft edge as arc_clip: recover the ball's colour, then lay it over the light background
            alpha = np.clip(style["radius"] + 1.0 - np.hypot(xx - cx, yy - cy), 0, 1)[..., None]
            ball = np.where(alpha > 0, (frames[t] - (1 - alpha) * background) / np.maximum(alpha, 1e-6), 0)
            shown.append(np.clip((1 - alpha) * light + alpha * ball, 0, 1))
        balls[angle] = shown
        paths[angle] = arc_centres(angle, np.linspace(0, ARC_FRAMES, 400), speed=style["speed"])
    pad = style["radius"] + 5
    half_x = max(np.ptp(p[:, 0]) for p in paths.values()) / 2 + pad
    half_y = max(np.ptp(p[:, 1]) for p in paths.values()) / 2 + pad + 6

    def draw(t):
        fig, axes = plt.subplots(1, len(ARC_ANGLES), figsize=(8.4, 2.4))
        for ax, angle in zip(axes, ARC_ANGLES):
            path = paths[angle]
            cx, cy = (path.max(0) + path.min(0)) / 2
            ax.imshow(balls[angle][t], interpolation="bilinear")
            ax.plot(path[:, 0], path[:, 1], color="#b9b7b0", lw=1.0, zorder=2)
            fitted = arc_centres(angle, times[::3], speed=style["speed"])
            ax.plot(fitted[:, 0], fitted[:, 1], "o", ms=2.6, color="#2a78d6", zorder=3)
            if t % 3 == 0:                  # the ball is at a fitted frame: ring its dot
                here = arc_centres(angle, np.array([t]), speed=style["speed"])[0]
                ax.plot(*here, "o", ms=7, mfc="none", mec="#2a78d6", mew=1.3, zorder=4)
            ax.set_xlim(cx - half_x, cx + half_x)
            ax.set_ylim(cy + half_y, cy - half_y)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(True); side.set_color(GRID)
            ax.set_title(f"turn {angle}°", loc="center")
        fig.tight_layout(rect=(0, 0.24, 1, 1))
        # a timeline under the panels: a tick per frame, the fitted ones tall and blue, and a pointer at t
        tl = fig.add_axes((0.06, 0.07, 0.6, 0.1))
        for k in times:
            fit = k % 3 == 0
            tl.plot([k, k], [0, 1 if fit else 0.45], color="#2a78d6" if fit else "#c9c7c0", lw=1.6 if fit else 1)
        tl.plot([t], [-0.35], marker="^", ms=6, color=INK, clip_on=False)
        tl.set_xlim(-0.6, ARC_FRAMES + 0.6)
        tl.set_ylim(-0.6, 1.1)
        tl.axis("off")
        tl.text(-1.4, 0.3, "frame", ha="right", va="center", fontsize=8, color=MUTED)
        tl.text(0, -0.75, "0", ha="center", va="top", fontsize=7.5, color=MUTED)
        tl.text(ARC_FRAMES, -0.75, str(ARC_FRAMES), ha="center", va="top", fontsize=7.5, color=MUTED)
        fig.text(0.70, 0.13, "blue: fitted frames (0, 3, …, 27)", fontsize=8, color="#2a78d6", va="center")
        fig.text(0.70, 0.05, "grey line: true path of the centre", fontsize=8, color=MUTED, va="center")
        return fig

    pictures = []
    for t in times:
        fig = draw(t)
        if t == 0:
            save(fig, out, "arc_paths")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor="white", bbox_inches=None)
        plt.close(fig)
        buf.seek(0)
        pictures.append(Image.open(buf).convert("RGB"))
    palette = pictures[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    gif = [p.quantize(palette=palette, dither=Image.Dither.NONE) for p in pictures]
    durations = [180] * len(gif)
    durations[-1] = 900                       # a pause at the end of the path before it starts again
    gif[0].save(out / "arc_motion.gif", save_all=True, append_images=gif[1:], duration=durations, loop=0, optimize=True)
    print("arc_motion.gif", flush=True)


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


def fig_motion(out: Path, plt, data_root=Path("data/uvg"), mask_root=Path("data/masks")):
    """Motion that a straight line cannot follow: the horse's legs and the shaking dog in frames
    0, 3, ..., 15 of the test clips (stride 3). The crop follows the object outline of each frame."""
    from PIL import Image
    from stg import data
    frames_k = list(range(0, 16, 3))
    size = {"Jockey": (150, 100), "ShakeNDry": (210, 140)}
    fig, axes = plt.subplots(2, len(frames_k), figsize=(9.6, 3.3))
    for r, seq in enumerate(("Jockey", "ShakeNDry")):
        frames = data.load_clip(data_root, seq, 0, 16, 1)
        w, h = size[seq]
        for c, k in enumerate(frames_k):
            mask = np.asarray(Image.open(mask_root / seq / f"{k:04d}.png").convert("L").resize(data.SIZE, Image.NEAREST)) > 127
            ys, xs = np.nonzero(mask)
            # the legs are the lower part of the horse and rider; the dog is centred
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + 0.72 * (ys.max() - ys.min()) if seq == "Jockey" else (ys.min() + ys.max()) / 2)
            x0 = int(np.clip(cx - w / 2, 0, data.SIZE[0] - w))
            y0 = int(np.clip(cy - h / 2, 0, data.SIZE[1] - h))
            ax = axes[r][c]
            ax.imshow(frames[k][y0:y0 + h, x0:x0 + w])
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"frame {k}", fontsize=8.5)
        axes[r][0].set_ylabel({"Jockey": "Jockey:\nthe legs", "ShakeNDry": "ShakeNDry:\nthe dog"}[seq], fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "motion")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, default=Path("docs/data"))
    p.add_argument("--out", type=Path, default=Path("docs/figures"))
    p.add_argument("--crops", action="store_true", help="also the figures made from video frames: motion and crops")
    a = p.parse_args(argv)
    runs = Runs(json.loads((a.data / "runs.json").read_text()))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.data / "tables.json").write_text(json.dumps(tables(runs), ensure_ascii=False, indent=0), encoding="utf-8")
    plt = style()
    for fig in (fig_rd, fig_gain_budget, fig_training, fig_still, fig_degree, fig_arc, fig_unseen, fig_confirm):
        fig(runs, a.out, plt)
    fig_arc_motion(a.out, plt)
    if a.crops:
        fig_motion(a.out, plt)
        fig_crops(a.out, plt)


if __name__ == "__main__":
    main()
