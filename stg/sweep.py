"""The experiment design of the proposal as lists of runs, one list per stage.

    validation   the swept hyperparameters (s_min, acc_ratio) on the Jockey validation clip
    test         the main experiment (288 fits) and the 30-frame Jockey clip (48 fits)
    synthetic    the four models on the synthetic clip with known paths

Each run is a list of arguments for ``stg.run``, and writes to ``runs/<stage>/`` by default. A Slurm
array job runs one of them per task; ``--all`` runs the whole stage in one process and skips the
runs that already have a metrics.json, so it can be stopped and restarted:

    python -m stg.sweep --stage test --count              # number of runs
    python -m stg.sweep --stage test --list               # the commands
    python -m stg.sweep --stage test --index $SLURM_ARRAY_TASK_ID --device cuda
    python -m stg.sweep --stage synthetic --all --device cuda
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from stg.data import SEQUENCES

BUDGETS = (3000, 6000, 12000, 24000)
SEEDS = (0, 1, 2)
STRIDES = (1, 3)
S_MINS = (0.05, 0.1, 0.2)
ACC_RATIOS = (0.3, 1.0, 3.0)
# The synthetic stage showed that the gain of M1 depends on the learning rate of v (step 7 of
# PROGRESS.md), so every model, M0 included, also gets its own rate of v. The rate of a is acc_ratio
# times the rate of v.
LR_VELS = (0.02, 0.05, 0.1)
MODELS = ("M0", "M1", "M2", "M3")

# A Jockey segment that does not overlap frames 0-29 and shows the galloping horse with the camera
# panning: frames 60-69 at stride 1 and 60-87 at stride 3. The camera zooms in over the sequence, and
# from frame 60 the whole horse is still in view at about the scale of the test clips.
VALIDATION_START = 60

# The settings before tuning, used by the synthetic stage.
DEFAULT = {"lr_vel": 0.05, "s_min": 0.1, "acc_ratio": 1.0}

# Chosen per model on the validation clip by ``stg.select``, then used for every test run; a model's
# s_min or acc_ratio is ignored when it has no shrinking or acceleration. Until the validation stage
# is done, the defaults.
CHOSEN_FILE = Path("runs/validation/chosen.json")
CHOSEN = {model: dict(DEFAULT) for model in MODELS}
if CHOSEN_FILE.exists():
    CHOSEN.update({model: {k: v[k] for k in DEFAULT} for model, v in json.loads(CHOSEN_FILE.read_text()).items()})


def _args(model, budget, sequence, stride, seed, start=0, frames=10, lr_vel=None, s_min=None, acc_ratio=None,
          settings=None):
    settings = settings or CHOSEN[model]
    lr_vel = settings["lr_vel"] if lr_vel is None else lr_vel
    s_min = settings["s_min"] if s_min is None else s_min
    acc_ratio = settings["acc_ratio"] if acc_ratio is None else acc_ratio
    return ["--model", model, "--budget", str(budget), "--sequence", sequence, "--start", str(start),
            "--frames", str(frames), "--stride", str(stride), "--seed", str(seed),
            "--lr-vel", str(lr_vel), "--s-min", str(s_min), "--acc-ratio", str(acc_ratio)]


def validation_runs():
    """For each budget and stride: M0 x 3 rates of v, M1 x 3 x 3 acc_ratio, M2 x 3 x 3 s_min,
    M3 x 3 x 3 x 3: 48 runs, so 384 in all."""
    if VALIDATION_START is None:
        raise SystemExit("set VALIDATION_START in stg/sweep.py to the first frame of the validation clip")
    runs = []
    for budget, stride, lr_vel in itertools.product(BUDGETS, STRIDES, LR_VELS):
        common = dict(budget=budget, sequence="Jockey", stride=stride, seed=0, start=VALIDATION_START,
                      lr_vel=lr_vel, settings=DEFAULT)
        runs.append(_args("M0", **common))
        runs += [_args("M1", acc_ratio=r, **common) for r in ACC_RATIOS]
        runs += [_args("M2", s_min=s, **common) for s in S_MINS]
        runs += [_args("M3", s_min=s, acc_ratio=r, **common) for s, r in itertools.product(S_MINS, ACC_RATIOS)]
    return runs


def validation_extension_runs():
    """The validation stage chose a rate of v of 0.1 for every model and acc_ratio 0.3 for M1 and M3,
    both at the edge of their grids. This adds the next values around those choices: rate of v
    0.2, and acc_ratio 0.1. 8 settings x 4 budgets x 2 strides = 64 runs, written with the validation
    runs (``--out runs/validation``) so that ``stg.select`` compares them all."""
    settings = [("M0", 0.2, 0.1, 1.0),
                ("M1", 0.2, 0.1, 0.3), ("M1", 0.1, 0.1, 0.1), ("M1", 0.2, 0.1, 0.1),
                ("M2", 0.2, 0.1, 1.0),
                ("M3", 0.2, 0.1, 0.3), ("M3", 0.1, 0.1, 0.1), ("M3", 0.2, 0.1, 0.1)]
    return [_args(model, budget, "Jockey", stride, 0, start=VALIDATION_START, lr_vel=lr_vel, s_min=s_min,
                  acc_ratio=acc_ratio)
            for budget, stride in itertools.product(BUDGETS, STRIDES)
            for model, lr_vel, s_min, acc_ratio in settings]


def test_runs():
    runs = [_args(model, budget, sequence, stride, seed)
            for sequence, stride, budget, model, seed
            in itertools.product(SEQUENCES, STRIDES, BUDGETS, MODELS, SEEDS)]
    # 30-frame Jockey clip at stride 1, budgets tripled so that each frame gets the same budget
    runs += [_args(model, 3 * budget, "Jockey", 1, seed, frames=30)
             for budget, model, seed in itertools.product(BUDGETS, MODELS, SEEDS)]
    return runs


def still_runs():
    """Plan A (PROGRESS.md): frame 0 of each sequence repeated 10 times, so the clip has no motion.
    4 images x 4 models x 2 budgets x 3 seeds = 96 runs, with the settings chosen for the test runs."""
    return [_args(model, budget, f"still_{sequence}", 1, seed)
            for sequence, budget, model, seed in itertools.product(SEQUENCES, (3000, 12000), MODELS, SEEDS)]


def long_runs():
    """Does the gain over M0 survive longer training? (PROGRESS.md, step 18.) Jockey, stride 3, seed 0,
    10000 iterations instead of 3000: all models at budgets 6000 and 24000 of the test runs, and M0
    and M3 at 192000 (16000 Gaussians of M0). The learning rates fall to a quarter over the run, as in
    every fit, so the schedule is stretched rather than continued."""
    runs = [_args(model, budget, "Jockey", 3, 0) + ["--iterations", "10000"]
            for budget, model in itertools.product((6000, 24000), MODELS)]
    runs += [_args(model, 192000, "Jockey", 3, 0) + ["--iterations", "10000"] for model in ("M0", "M3")]
    return runs


def synthetic_runs():
    return [_args(model, budget, "synthetic", 1, seed, settings=DEFAULT)
            for budget, model, seed in itertools.product(BUDGETS, MODELS, SEEDS)]


STAGES = {"validation": validation_runs, "validation_extension": validation_extension_runs, "test": test_runs,
          "synthetic": synthetic_runs, "still": still_runs, "long": long_runs}


def main(argv=None):
    p = argparse.ArgumentParser(description="Runs of one stage of the experiment.")
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--count", action="store_true")
    group.add_argument("--list", action="store_true")
    group.add_argument("--index", type=int)
    group.add_argument("--all", action="store_true")
    a, extra = p.parse_known_args(argv)         # extra arguments go to every run, e.g. --device cuda
    if "--out" not in extra:
        extra = extra + ["--out", f"runs/{a.stage}"]
    runs = [r + extra for r in STAGES[a.stage]()]
    if a.count:
        print(len(runs))
    elif a.list:
        for i, r in enumerate(runs):
            print(i, "python -m stg.run", " ".join(r))
    elif a.index is not None:
        from stg import run
        run.main(runs[a.index])
    else:
        import time
        from pathlib import Path
        from stg import run
        start = time.perf_counter()
        for i, r in enumerate(runs):
            args = run.parser().parse_args(r)
            if (Path(args.out) / (args.name or run.run_name(args)) / "metrics.json").exists():
                continue
            print(f"[{i + 1}/{len(runs)}] {time.perf_counter() - start:.0f}s", flush=True)
            run.main(r)


if __name__ == "__main__":
    main()
