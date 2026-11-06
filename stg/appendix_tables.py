"""The data tables of the appendix of the project page, one statistic per cell, and the downloadable CSV files,
from ``docs/data/runs.json`` (PROGRESS.md, step 57). Trains nothing and changes no number: the gains are
paired by seed and the test of Section 3.3 is applied exactly as in ``stg.figures`` (``Runs``, ``THRESHOLD``).

Writes ``docs/data/appendix_tables.json`` (read by docs/assets/js/main.js, ``data-atable``) and
``docs/data/csv/*.csv``. Leaves ``docs/data/tables.json`` alone.

    python -m stg.appendix_tables
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from stg.figures import BUDGETS, MODELS, SEQUENCES, THRESHOLD, Runs, plain, signed

NAME = {"M1": "M1 curved", "M2": "M2 shrinking", "M3": "M3 both"}
DATA = Path("docs/data")


# --- table structure --------------------------------------------------------------------------------
# head: rows of header cells {"t", "cs", "rs", "num"}; num: per column, right-aligned numbers;
# rows: {"g": label} for a group row, or {"c": [cells], "ok": [column indices meeting the criterion]}

def H(t, cs=1, rs=1, num=False):
    return {"t": t, "cs": cs, "rs": rs, "num": num}


def mean_low_high(g):
    return [signed(g[0]), signed(g[1]), signed(g[2])]


def meets(cell, m):
    """The test of Section 3.3 for one budget, as in stg.figures.failure_table."""
    g = cell.gain(m)[0]
    return bool(g > THRESHOLD and g > max(cell.seed_range(m), cell.seed_range("M0")))


def test_table(runs):
    rows = []
    for seq in ("Jockey", "ShakeNDry"):
        rows.append({"g": seq})
        for m in MODELS[1:]:
            cells, ok, n = [NAME[m]], [], 0
            for i, b in enumerate(BUDGETS):
                cell = runs.where(sequence=seq, stride=3, budget=b)
                cells.append(signed(cell.gain(m)[0]))
                if meets(cell, m):
                    ok.append(i + 1)
                    n += 1
            rows.append({"c": cells + [f"{n} of 4"], "ok": ok})
    head = [[H("Model", rs=2), H("Gain over M0 at each budget (dB)", cs=4, num=True), H("Budgets meeting the criterion", rs=2, num=True)],
            [H("3,000", num=True), H("6,000", num=True), H("12,000", num=True), H("24,000", num=True)]]
    return {"head": head, "num": [False, True, True, True, True, True], "rows": rows}


def model_triplet_head(first, label="− M0 (dB)", models=MODELS[1:], extra=()):
    top = [H(t, rs=2, num=n) for t, n in first]
    top += [H(f"{m} {label}" if m != "M0" else m, cs=3, num=True) for m in models]
    top += [H(t, rs=2, num=True) for t in extra]
    sub = [H(x, num=True) for _ in models for x in ("Mean", "Lowest", "Highest")]
    return [top, sub]


def tables(runs: Runs) -> dict:
    t = {}
    test, long = runs.where(stage="test", frames=10, start=0), runs.where(stage="long_test")
    avg = lambda src, m, value="psnr": np.mean([src.where(budget=b).gain(m, value)[0] for b in BUDGETS])

    # B.1: gain after 3,000 steps by clip and stride, mean over budgets and seeds
    rows = []
    for seq in SEQUENCES:
        rows.append({"g": seq})
        for stride in (1, 3):
            cell = test.where(sequence=seq, stride=stride)
            rows.append({"c": [str(stride)] + [signed(avg(cell, m)) for m in MODELS[1:]]})
    t["t3k_avg"] = {"head": [[H("Stride", rs=2, num=True), H("Gain over M0 (dB)", cs=3, num=True)],
                             [H(NAME[m], num=True) for m in MODELS[1:]]],
                    "num": [True] * 4, "rows": rows}
    t["t3k_test"] = test_table(test)

    rows = []
    for seq in SEQUENCES:
        rows.append({"g": seq})
        for stride in (1, 3):
            cell = test.where(sequence=seq, stride=stride)
            rows.append({"c": [str(stride)] + [signed(avg(cell, m, "ssim")) for m in MODELS[1:]]
                         + [signed(avg(cell, m, "vmaf"), 1) for m in MODELS[1:]]})
    t["t3k_other"] = {"head": [[H("Stride", rs=2, num=True), H("SSIM gain over M0", cs=3, num=True), H("VMAF gain over M0", cs=3, num=True)],
                               [H(m, num=True) for m in MODELS[1:]] * 2],
                      "num": [True] * 7, "rows": rows}

    # B.2: fully trained
    t["full_test"] = test_table(long)
    rows = []
    for seq in SEQUENCES:
        rows.append({"g": seq})
        for b in BUDGETS:
            cell = long.where(sequence=seq, budget=b)
            rows.append({"c": [f"{b:,}", plain(cell.mean("M0"))] + [x for m in MODELS[1:] for x in mean_low_high(cell.gain(m))]})
    t["full_cells"] = {"head": model_triplet_head([("Budget", True), ("M0 PSNR (dB)", True)]),
                       "num": [True] * 11, "rows": rows}

    # B.4: degree of the path
    poly = Runs(long.where(stride=3).records + runs.where(stage="poly").records)
    rows = []
    for value, where in (("psnr", "Fitted frames"), ("unseen", "Frames between"), ("unseen_in", "Frames between, object")):
        rows.append({"g": where})
        for seq in ("Jockey", "ShakeNDry"):
            for b in (6000, 24000):
                cell = poly.where(sequence=seq, budget=b)
                rows.append({"c": [seq, f"{b:,}", plain(cell.mean("M0", value))]
                             + [x for m in ("M1", "D3", "D7") for x in mean_low_high(cell.gain(m, value))]})
    t["poly"] = {"head": model_triplet_head([("Clip", False), ("Budget", True), ("M0 PSNR (dB)", True)], models=("M1", "D3", "D7")),
                 "num": [False] + [True] * 11, "rows": rows}

    # B.5: the 30-frame clip, and the cost
    clip = runs.where(stage="test", frames=30)
    t["long_clip"] = {"head": [[H("Budget", rs=2, num=True), H("M0 PSNR (dB)", rs=2, num=True), H("Gain over M0 (dB)", cs=3, num=True)],
                               [H(NAME[m], num=True) for m in MODELS[1:]]],
                      "num": [True] * 5,
                      "rows": [{"c": [f"{b:,}", plain(clip.where(budget=b).mean("M0"))] + [signed(clip.where(budget=b).gain(m)[0]) for m in MODELS[1:]]}
                               for b in sorted({r["budget"] for r in clip.records})]}
    cell = test.where(budget=24000)
    mean = lambda m, k: np.mean([r[k] for r in cell.records if r["model"] == m])
    base = mean("M0", "seconds_per_iteration")
    t["cost"] = {"head": [[H("Model"), H("Training step (ms)", num=True), H("Change from M0 (%)", num=True), H("Rendering a frame (ms)", num=True)]],
                 "num": [False, True, True, True],
                 "rows": [{"c": [m, f"{1000 * mean(m, 'seconds_per_iteration'):.1f}",
                                 "—" if m == "M0" else signed(100 * (mean(m, "seconds_per_iteration") / base - 1), 0),
                                 f"{1000 * mean(m, 'render_seconds_per_frame'):.2f}"]} for m in MODELS]}

    # C.1: object and background; still photographs
    sub = [H(x, num=True) for _ in MODELS[1:] for x in ("Object", "Background")]
    rows = []
    for seq in SEQUENCES:
        rows.append({"g": seq})
        for stride in (1, 3):
            c = test.where(sequence=seq, stride=stride)
            rows.append({"c": [str(stride)] + [signed(avg(c, m, v)) for m in MODELS[1:] for v in ("psnr_in", "psnr_out")]})
    t["t3k_inout"] = {"head": [[H("Stride", rs=2, num=True)] + [H(f"{NAME[m]} − M0 (dB)", cs=2, num=True) for m in MODELS[1:]], sub],
                      "num": [True] * 7, "rows": rows}
    t["full_inout"] = {"head": [[H("Clip", rs=2)] + [H(f"{NAME[m]} − M0 (dB)", cs=2, num=True) for m in MODELS[1:]], sub],
                       "num": [False] + [True] * 6,
                       "rows": [{"c": [seq] + [signed(avg(long.where(sequence=seq), m, v)) for m in MODELS[1:] for v in ("psnr_in", "psnr_out")]}
                                for seq in SEQUENCES]}
    still = runs.where(stage="still")
    rows = []
    for seq in SEQUENCES:
        rows.append({"g": seq})
        for b in (3000, 12000):
            c, moving = still.where(sequence=f"still_{seq}", budget=b), test.where(sequence=seq, stride=1, budget=b)
            rows.append({"c": [f"{b:,}", plain(c.mean("M0"))] + [x for m in MODELS[1:] for x in mean_low_high(c.gain(m))]
                         + [signed(moving.gain("M2")[0])]})
    t["still"] = {"head": model_triplet_head([("Budget", True), ("M0 PSNR (dB)", True)], extra=("M2 − M0 on the moving clip (dB)",)),
                  "num": [True] * 12, "rows": rows}

    # D.1: the confirmatory test, one row per cell
    conf = runs.where(stage="confirm")
    clips = (("ShakeNDry", 30, "ShakeNDry from frame 30 (dog)"), ("Jockey", 120, "Jockey from frame 120"),
             ("Jockey", 200, "Jockey from frame 200"))
    rows, passed = [], 0
    for seq, start, label in clips:
        rows.append({"g": label})
        for stride in (3, 5):
            for b in (6000, 24000):
                g = conf.where(sequence=seq, start=start, stride=stride, budget=b).gain("M1", "unseen_in")
                ok = g[1] > 0
                passed += ok
                rows.append({"c": [str(stride), f"{b:,}"] + mean_low_high(g) + ["Yes" if ok else "No"], "ok": [5] if ok else []})
    t["confirm"] = {"head": [[H("Stride", rs=2, num=True), H("Budget", rs=2, num=True), H("M1 − M0 (dB)", cs=3, num=True), H("All 3 seeds above zero", rs=2)],
                             [H("Mean", num=True), H("Lowest", num=True), H("Highest", num=True)]],
                    "num": [True] * 5 + [False], "rows": rows, "passed": int(passed)}
    rows = []
    for value, label in (("psnr_in", "Fitted frames, object"), ("unseen", "Frames between, whole frame"),
                         ("psnr", "Fitted frames, whole frame")):
        g = [conf.where(sequence=s, start=k, stride=st, budget=b).gain("M1", value)[0]
             for s, k, _ in clips for st in (3, 5) for b in (6000, 24000)]
        rows.append({"c": [label, signed(min(g)), signed(max(g))]})
    t["confirm_other"] = {"head": [[H("M1 − M0 on"), H("Lowest of the 12 cells (dB)", num=True), H("Highest of the 12 cells (dB)", num=True)]],
                          "num": [False, True, True], "rows": rows}
    t["confirm_stride_means"] = {
        s: signed(np.mean([conf.where(sequence=q, start=k, stride=s, budget=b).gain("M1", "unseen_in")[0]
                           for q, k, _ in clips for b in (6000, 24000)])) for s in (3, 5)}

    # D.2: the arcs, second version
    arcs = runs.where(stage="arc_slow")
    rows = []
    for angle in (0, 90, 180, 270, 360):
        rows.append({"g": f"{angle}°" + (" (straight)" if angle == 0 else " (full turn)" if angle == 360 else "")})
        for b in (6000, 24000):
            c = arcs.where(sequence=f"arcs{angle}", budget=b)
            rows.append({"c": [f"{b:,}"] + [x for v in ("psnr_in", "unseen_in") for x in [plain(c.mean("M0", v))] + mean_low_high(c.gain("M1", v))]})
    t["arc"] = {"head": [[H("Budget", rs=2, num=True), H("Fitted frames, on the ball (dB)", cs=4, num=True), H("Frames between, on the ball (dB)", cs=4, num=True)],
                         [H(x, num=True) for _ in range(2) for x in ("M0 PSNR", "M1 − M0 mean", "Lowest", "Highest")]],
                "num": [True] * 9, "rows": rows}

    # D.3: fitted frames and frames between
    rows = []
    for seq in SEQUENCES:
        rows.append({"g": seq})
        for b in BUDGETS:
            c = long.where(sequence=seq, budget=b)
            rows.append({"c": [f"{b:,}"] + [plain(c.mean("M0", v)) for v in ("psnr", "unseen", "psnr_in", "unseen_in")]})
    t["unseen_m0"] = {"head": [[H("Budget", rs=2, num=True), H("Whole frame (dB)", cs=2, num=True), H("Object (dB)", cs=2, num=True)],
                               [H(x, num=True) for x in ("Fitted frames", "Frames between") * 2]],
                      "num": [True] * 5, "rows": rows}
    t["unseen_gain"] = {"head": [[H("Clip", rs=2)] + [H(f"{NAME[m]} − M0 (dB)", cs=2, num=True) for m in MODELS[1:]],
                                 [H(x, num=True) for _ in MODELS[1:] for x in ("Fitted frames", "Frames between")]],
                        "num": [False] + [True] * 6,
                        "rows": [{"c": [seq] + [signed(avg(long.where(sequence=seq), m, v)) for m in MODELS[1:] for v in ("psnr", "unseen")]}
                                 for seq in SEQUENCES]}
    rows = []
    for seq in SEQUENCES:
        rows.append({"g": seq})
        for b in BUDGETS:
            c = long.where(sequence=seq, budget=b)
            rows.append({"c": [f"{b:,}"] + [x for m in MODELS[1:] for x in mean_low_high(c.gain(m, "unseen_in"))]
                         + [signed(c.gain("M1", "psnr_in")[0])]})
    t["unseen_obj"] = {"head": model_triplet_head([("Budget", True)], extra=("M1 − M0, object on the fitted frames (dB)",)),
                       "num": [True] * 11, "rows": rows}
    return t


# --- check against the published tables (docs/data/tables.json) ------------------------------------

def check(t, old):
    """Rebuild the strings of the old tables from the new cells, so that no value or mark has changed."""
    cells = lambda key: [r["c"] for r in t[key]["rows"] if "c" in r]
    rng = lambda c: f"{c[0]} [{c[1]}, {c[2]}]"
    n = 0
    for key, okey in (("t3k_test", "test_fail"), ("full_test", "long_fail")):
        new = cells(key)
        for mi, row in enumerate(old[okey]["rows"]):
            for si in range(2):
                r = t[key]["rows"][1 + si * 4 + mi]
                s = ", ".join(f"{r['c'][i + 1]} {'✓' if i + 1 in r['ok'] else '✗'}" for i in range(4)) + f" → {r['c'][5].split()[0]}/4"
                assert s == row[1 + si], (key, s, row[1 + si]); n += 1
    for new, orow in zip(cells("t3k_avg"), old["test_avg"]["rows"]):
        assert new == orow[1:], (new, orow); n += 1
    for new, orow in zip(cells("t3k_other"), old["test_other"]["rows"]):
        assert " / ".join(new[1:4]) == orow[2] and " / ".join(new[4:7]) == orow[3]; n += 1
    for new, orow in zip(cells("full_cells"), old["long_cells"]["rows"]):
        assert new[:2] == orow[1:3] and [rng(new[2:5]), rng(new[5:8]), rng(new[8:11])] == [orow[3], orow[5], orow[7]]; n += 1
    for new, orow in zip(cells("poly"), old["poly"]["rows"]):
        assert new[:3] == [orow[0], orow[1], orow[3]] and [rng(new[3:6]), rng(new[6:9]), rng(new[9:12])] == orow[4:]; n += 1
    for new, orow in zip(cells("long_clip"), old["long_clip"]["rows"]):
        assert new == orow; n += 1
    for new, orow in zip(cells("cost"), old["cost"]["rows"]):
        assert orow[1].startswith(new[1] + " ms") and orow[2] == new[3] + " ms" and (new[2] == "—" or f"({new[2]}%)" in orow[1]); n += 1
    for new, orow in zip(cells("t3k_inout"), old["test_inout"]["rows"]):
        assert [f"{new[1 + 2 * i]} / {new[2 + 2 * i]}" for i in range(3)] == orow[2:]; n += 1
    for new, orow in zip(cells("full_inout"), old["long_inout"]["rows"]):
        assert [f"{new[1 + 2 * i]} / {new[2 + 2 * i]}" for i in range(3)] == orow[1:]; n += 1
    for new, orow in zip(cells("still"), old["still"]["rows"]):
        assert new[:2] == orow[1:3] and [rng(new[2:5]), rng(new[5:8]), rng(new[8:11])] == orow[3:6] and new[11] == orow[6]; n += 1
    old_conf = [c for r in old["confirm"]["rows"] for c in r[1:]]
    for new, oc in zip(cells("confirm"), old_conf):
        assert f"{rng(new[2:5])} {'✓' if new[5] == 'Yes' else '✗'}" == oc; n += 1
    assert t["confirm"]["passed"] == old["confirm"]["passed"]
    for new, orow in zip(cells("confirm_other"), old["confirm_other"]["rows"][:3]):
        assert f"{new[1]} to {new[2]}" == orow[1]; n += 1
    assert [t["confirm_stride_means"][s] for s in (3, 5)] == [r[1] for r in old["confirm_other"]["rows"][3:]]
    old_arc = [r[1:] for r in old["arc"]["rows"]]
    new_arc = cells("arc")
    for i, orow in enumerate(old_arc):
        for j, b in enumerate((0, 1)):
            r = new_arc[2 * i + b]
            assert f"{r[1]}; {rng(r[2:5])}" == orow[j] and f"{r[5]}; {rng(r[6:9])}" == orow[2 + j]; n += 1
    for new, orow in zip(cells("unseen_m0"), old["unseen_m0"]["rows"]):
        assert new == orow[1:]; n += 1
    for new, orow in zip(cells("unseen_gain"), old["unseen_gain"]["rows"]):
        assert [f"{new[1 + 2 * i]} → {new[2 + 2 * i]}" for i in range(3)] == orow[1:]; n += 1
    for new, orow in zip(cells("unseen_obj"), old["unseen_obj"]["rows"]):
        assert new[0] == orow[1] and [rng(new[1:4]), rng(new[4:7]), rng(new[7:10])] == orow[2:5] and new[10] == orow[5]; n += 1
    return n


# --- CSV ---------------------------------------------------------------------------------------------

FIT_FIELDS = [("stage", "stage"), ("sequence", "clip"), ("start", "start_frame"), ("frames", "frames"),
              ("stride", "stride"), ("model", "model"), ("budget", "budget_numbers"), ("seed", "seed"),
              ("iterations", "training_steps"), ("lr_vel", "velocity_learning_rate"), ("s_min", "s_min"),
              ("acc_ratio", "acceleration_ratio"), ("gaussians", "gaussians"), ("psnr", "psnr_fitted_db"),
              ("ssim", "ssim_fitted"), ("vmaf", "vmaf_fitted"), ("psnr_in", "psnr_fitted_object_db"),
              ("psnr_out", "psnr_fitted_background_db"), ("unseen", "psnr_between_db"),
              ("unseen_in", "psnr_between_object_db"), ("unseen_out", "psnr_between_background_db"),
              ("train_seconds", "training_seconds"), ("seconds_per_iteration", "seconds_per_training_step"),
              ("render_seconds_per_frame", "seconds_per_rendered_frame")]


def num(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.6f}" if isinstance(x, float) else str(x))


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[num(v) if not isinstance(v, str) else v for v in r] for r in rows])
    return len(rows)


def gain_fields(cell, m, value):
    g = cell.gain(m, value)
    return list(g) if not np.isnan(g[0]) else [None] * 3


def csvs(runs: Runs, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    counts["fits.csv"] = write_csv(out / "fits.csv", [b for _, b in FIT_FIELDS],
                                   [[r.get(a) for a, _ in FIT_FIELDS] for r in runs.records])
    test, long = runs.where(stage="test", frames=10, start=0), runs.where(stage="long_test")
    G = lambda name: [f"{name}_mean_db", f"{name}_lowest_db", f"{name}_highest_db"]
    for name, src, strides, extra in (("gains_3000_steps.csv", test, (1, 3), ()), ("gains_full_training.csv", long, (3,), ("unseen", "unseen_in"))):
        rows = []
        for seq in SEQUENCES:
            for stride in strides:
                for b in BUDGETS:
                    cell = src.where(sequence=seq, stride=stride, budget=b)
                    for m in MODELS[1:]:
                        crit = "yes" if meets(cell, m) else "no"
                        crit = crit if stride == 3 and seq in ("Jockey", "ShakeNDry") else ""
                        rows.append([seq, stride, b, m, len(cell.seeds(m, "psnr")), cell.mean("M0")]
                                    + gain_fields(cell, m, "psnr") + [max(cell.seed_range(m), cell.seed_range("M0")), crit]
                                    + [cell.gain(m, v)[0] for v in ("psnr_in", "psnr_out", "ssim", "vmaf")]
                                    + [x for v in extra for x in gain_fields(cell, m, v)])
        head = (["clip", "stride", "budget_numbers", "model", "seeds", "m0_psnr_fitted_mean_db"] + G("gain_psnr_fitted")
                + ["seed_spread_db", "meets_criterion_of_section_3_3", "gain_psnr_fitted_object_mean_db",
                   "gain_psnr_fitted_background_mean_db", "gain_ssim_mean", "gain_vmaf_mean"]
                + [x for v in extra for x in G({"unseen": "gain_psnr_between", "unseen_in": "gain_psnr_between_object"}[v])])
        counts[name] = write_csv(out / name, head, rows)
    poly = Runs(long.where(stride=3).records + runs.where(stage="poly").records)
    rows = []
    for seq in ("Jockey", "ShakeNDry"):
        for b in (6000, 24000):
            cell = poly.where(sequence=seq, budget=b)
            for m in ("M1", "D3", "D7"):
                rows.append([seq, b, m] + [x for v in ("psnr", "unseen", "unseen_in") for x in gain_fields(cell, m, v)])
    counts["degree.csv"] = write_csv(out / "degree.csv", ["clip", "budget_numbers", "model"] + G("gain_psnr_fitted")
                                     + G("gain_psnr_between") + G("gain_psnr_between_object"), rows)
    still = runs.where(stage="still")
    rows = [[seq, b, m, still.where(sequence=f"still_{seq}", budget=b).mean("M0")] + gain_fields(still.where(sequence=f"still_{seq}", budget=b), m, "psnr")
            for seq in SEQUENCES for b in (3000, 12000) for m in MODELS[1:]]
    counts["still_photographs.csv"] = write_csv(out / "still_photographs.csv",
                                                ["photograph_from_clip", "budget_numbers", "model", "m0_psnr_mean_db"] + G("gain_psnr"), rows)
    conf = runs.where(stage="confirm")
    rows = []
    for seq, start in (("ShakeNDry", 30), ("Jockey", 120), ("Jockey", 200)):
        for stride in (3, 5):
            for b in (6000, 24000):
                cell = conf.where(sequence=seq, start=start, stride=stride, budget=b)
                g = cell.gain("M1", "unseen_in")
                rows.append([seq, start, stride, b] + list(g) + ["yes" if g[1] > 0 else "no"]
                            + [cell.gain("M1", v)[0] for v in ("unseen", "psnr_in", "psnr")])
    counts["confirmatory_test.csv"] = write_csv(out / "confirmatory_test.csv",
                                                ["clip", "start_frame", "stride", "budget_numbers"] + G("gain_psnr_between_object")
                                                + ["all_3_seeds_above_zero", "gain_psnr_between_mean_db", "gain_psnr_fitted_object_mean_db",
                                                   "gain_psnr_fitted_mean_db"], rows)
    arcs = runs.where(stage="arc_slow")
    rows = []
    for angle in (0, 90, 180, 270, 360):
        for b in (6000, 24000):
            c = arcs.where(sequence=f"arcs{angle}", budget=b)
            rows.append([angle, b] + [x for v in ("psnr_in", "unseen_in") for x in [c.mean("M0", v)] + gain_fields(c, "M1", v)])
    counts["synthetic_arcs.csv"] = write_csv(out / "synthetic_arcs.csv",
                                             ["turn_degrees", "budget_numbers", "m0_psnr_fitted_ball_mean_db"] + G("gain_psnr_fitted_ball")
                                             + ["m0_psnr_between_ball_mean_db"] + G("gain_psnr_between_ball"), rows)
    return counts


def main():
    runs = Runs(json.loads((DATA / "runs.json").read_text(encoding="utf-8")))
    t = tables(runs)
    n = check(t, json.loads((DATA / "tables.json").read_text(encoding="utf-8")))
    (DATA / "appendix_tables.json").write_text(json.dumps(t, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    counts = csvs(runs, DATA / "csv")
    print(f"{len(t)} tables, {n} rows checked against tables.json; CSV rows: {counts}")


if __name__ == "__main__":
    main()
