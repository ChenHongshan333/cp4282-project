"""Adam over all frames of a clip, as in the course's ``fit`` but with one group per attribute."""

from __future__ import annotations

import time

import numpy as np

from gaussian_image_video import MIN_SCALE
from stg.metrics import psnr_per_frame
from stg.model import SpaceTimeGaussians
from stg.render import ClipRenderer

# Learning rates per step. mu, chol and colour are the course's; the others are starting points to
# be checked on the validation clip. The rate of ``acc`` is ``acc_ratio`` times the rate of ``vel``.
RATES = {"mu": 0.25, "vel": 0.05, "t0": 0.05, "sigma_t": 0.05, "chol": 0.05, "colour": 0.01}
SIGMA_T_MIN = 0.3      # lower bound on the duration, in frames


def fit(g: SpaceTimeGaussians, frames: np.ndarray, iterations: int, device, s_min: float,
        rates: dict | None = None, acc_ratio: float = 1.0, trace_every: int = 0, log=print):
    """Fit ``g`` in place. Returns a trace of (step, loss, mean PSNR, seconds since start)."""
    rates = dict(RATES, **(rates or {}))
    rates["acc"] = acc_ratio * rates["vel"]
    names = list(g.params)
    first = {n: np.zeros_like(g.params[n]) for n in names}
    second = {n: np.zeros_like(g.params[n]) for n in names}
    renderer = ClipRenderer(*frames.shape[:3], device, targets=frames)
    trace = []
    start = time.perf_counter()
    for step in range(1, iterations + 1):
        tracing = trace_every and (step == 1 or step % trace_every == 0)
        loss, grads, images = renderer.loss_and_gradients(g, s_min, want_images=tracing)
        if tracing:
            value = float(np.mean(psnr_per_frame(images, frames)))
            trace.append((step - 1, loss, value, time.perf_counter() - start))
            log(f"  step {step - 1:5d}  loss {loss:.6f}  PSNR {value:.2f}  {trace[-1][3]:.1f}s")
        decay = 0.5 ** (step / max(iterations, 1) * 2.0)       # the rates fall to a quarter
        for n in names:
            m, v, grad = first[n], second[n], grads[n]
            m[:] = 0.9 * m + 0.1 * grad
            v[:] = 0.999 * v + 0.001 * grad * grad
            g.params[n] -= rates[n] * decay * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        g.chol[:, 0] = np.maximum(g.chol[:, 0], MIN_SCALE)
        g.chol[:, 2] = np.maximum(g.chol[:, 2], MIN_SCALE)
        g.sigma_t[:] = np.maximum(g.sigma_t, SIGMA_T_MIN)
    return trace
