"""Finite-difference checks of the hand-written gradients, for all four models.

1. The slicing step alone, in double precision: for a random upstream gradient G on the sliced
   centre, Cholesky factor and colour, compare d(G . slice)/d(parameter) with central differences.
   The test points have w above and below s_min, so both branches of the shrinking are checked.
2. The whole clip loss through the course renderer, as in the course's ``gradient_check``: a cut of
   6 standard deviations in space and in time, so that the support does not change under the steps.

    python -m stg.check --device cpu
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

from stg.model import MODELS, SpaceTimeGaussians
from stg.render import loss_and_gradients, render_clip
from stg.slicing import slice_at, slice_backward

S_MIN = 0.3


def _random_gaussians(spec, n, width, height, n_frames, rng, dtype):
    arrays = {
        "mu": rng.uniform(0.35, 0.65, (n, 2)) * (width, height),
        "vel": rng.uniform(-1.5, 1.5, (n, 2)),
        "acc": rng.uniform(-0.3, 0.3, (n, 2)),
        "t0": rng.uniform(0.5, n_frames - 1.5, n),
        "sigma_t": rng.uniform(0.8, 1.6, n),
        "chol": np.column_stack([rng.uniform(1.5, 3, n), rng.uniform(-1, 1, n), rng.uniform(1.5, 3, n)]),
        "colour": rng.uniform(0.2, 0.8, (n, 3)),
    }
    if not spec.accel:
        del arrays["acc"]
    return SpaceTimeGaussians(spec, **{k: v.astype(dtype) for k, v in arrays.items()})


def check_slicing(step=1e-6):
    rng = np.random.default_rng(0)
    worst = 0.0
    for spec in MODELS.values():
        g = _random_gaussians(spec, 6, 48, 48, 6, rng, np.float64)
        for t in (0.0, 2.0, 2.7, 5.0):
            sl = slice_at(g, t, S_MIN, time_cut=1e9)
            upstream = [rng.normal(size=a.shape) for a in (sl.mu, sl.chol, sl.colour)]

            def objective(h):
                s = slice_at(h, t, S_MIN, time_cut=1e9)
                return sum(float(np.sum(u * a)) for u, a in zip(upstream, (s.mu, s.chol, s.colour)))

            grads = {n: np.zeros_like(a) for n, a in g.params.items()}
            slice_backward(g, sl, S_MIN, *upstream, grads)
            for name, grad in grads.items():
                for index in np.ndindex(g.params[name].shape):
                    w = sl.w[index[0]]
                    if spec.shrink and abs(w - S_MIN) < 1e-3:
                        continue          # the kink of max(w, s_min)
                    values = []
                    for sign in (1.0, -1.0):
                        h = g.copy()
                        h.params[name][index] += sign * step
                        values.append(objective(h))
                    numeric = (values[0] - values[1]) / (2 * step)
                    worst = max(worst, abs(numeric - grad[index]) / max(1.0, abs(numeric)))
    print(f"slicing: largest difference between the hand-written and numerical gradients: {worst:.2e}")
    assert worst < 1e-5


def check_clip(device, step=1e-2):
    """Relative to the largest gradient of each attribute, as in the course's check."""
    rng = np.random.default_rng(1)
    n_frames, size = 4, 40
    frames = rng.uniform(0, 1, (n_frames, size, size, 3)).astype(np.float32)
    for spec in MODELS.values():
        g = _random_gaussians(spec, 5, size, size, n_frames, rng, np.float32)
        _, grads, _ = loss_and_gradients(g, frames, device, S_MIN, cut=6.0, time_cut=6.0)

        def loss(h):
            images = render_clip(h, n_frames, size, size, device, S_MIN, cut=6.0, time_cut=6.0)
            return float(np.mean((images.astype(np.float64) - frames) ** 2))

        worst = 0.0
        for name, grad in grads.items():
            scale = np.abs(grad).max()
            for index in np.ndindex(g.params[name].shape):
                values = []
                for sign in (1.0, -1.0):
                    h = g.copy()
                    h.params[name][index] += sign * step
                    values.append(loss(h))
                numeric = (values[0] - values[1]) / (2 * step)
                worst = max(worst, abs(numeric - grad[index]) / scale)
        print(f"clip, {spec.name}: largest difference relative to the largest gradient of each attribute: {worst:.4f}")
        assert worst < 0.02


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    wp.init()
    check_slicing()
    check_clip(a.device)


if __name__ == "__main__":
    main()
