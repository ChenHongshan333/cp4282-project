r"""Slicing a space-time Gaussian at frame t, for the four models, and its backward pass.

Writing tau = t - t0, the slice at frame t is the 2D Gaussian of the course renderer with

    centre      mu(t)  = mu + v tau            (+ a tau^2 for M1 and M3)
    Cholesky    L(t)   = sqrt(s) L,            s = max(w, s_min) for M2 and M3, s = 1 otherwise
    colour      c(t)   = w c,                  w = exp(-tau^2 / (2 sigma_t^2))

Shrinking scales the covariance L L^T by s, so it scales the Cholesky factor by sqrt(s).

The backward pass takes the gradients of the loss with respect to the sliced 2D attributes, which
the course's gradient kernel returns, and adds the gradients with respect to the stored attributes:

    d/d mu  = g_mu                      d/d v = tau g_mu              d/d a = tau^2 g_mu
    d/d c   = w g_c                     d/d L = sqrt(s) g_L
    dL/dtau = (v + 2 a tau) . g_mu  +  dL/dw * dw/dtau,        dw/dtau   = -w tau / sigma_t^2
    dL/dw   = c . g_c  (+ (L . g_L) / (2 sqrt(s))  for shrinking models where w > s_min)
    d/d t0  = -dL/dtau                  d/d sigma_t = dL/dw * w tau^2 / sigma_t^3
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaussian_image_video import CUT
from stg.model import SpaceTimeGaussians

# A Gaussian is skipped in a frame where |tau| >= TIME_CUT sigma_t, where w < exp(-TIME_CUT^2 / 2) =
# 0.002: the same rule in time as the course renderer's cut in space.
TIME_CUT = CUT


@dataclass
class Slice:
    index: np.ndarray       # the Gaussians that are active in this frame
    mu: np.ndarray          # (n, 2) sliced centres
    chol: np.ndarray        # (n, 3) sliced Cholesky factors
    colour: np.ndarray      # (n, 3) time-weighted colours
    tau: np.ndarray
    w: np.ndarray
    s: np.ndarray


def slice_at(g: SpaceTimeGaussians, t: float, s_min: float, time_cut: float = TIME_CUT) -> Slice:
    spec = g.spec
    tau_all = t - g.t0
    index = np.nonzero(np.abs(tau_all) < time_cut * g.sigma_t)[0]
    tau = tau_all[index]
    sigma = g.sigma_t[index]
    w = np.exp(-0.5 * (tau / sigma) ** 2)
    mu = g.mu[index] + g.vel[index] * tau[:, None]
    if spec.accel:
        mu = mu + g.acc[index] * (tau ** 2)[:, None]
    s = np.maximum(w, s_min) if spec.shrink else np.ones_like(w)
    chol = g.chol[index] * np.sqrt(s)[:, None]
    colour = g.colour[index] * w[:, None]
    return Slice(index, mu, chol, colour, tau, w, s)


def slice_backward(g: SpaceTimeGaussians, sl: Slice, s_min: float, g_mu, g_chol, g_colour, grads: dict):
    """Add the gradients of one frame, with respect to the stored attributes, to ``grads``."""
    spec = g.spec
    i, tau, w, s = sl.index, sl.tau, sl.w, sl.s
    sigma = g.sigma_t[i]
    r = np.sqrt(s)
    # ``i`` holds each Gaussian at most once, so indexed += does not drop repeated entries.
    grads["mu"][i] += g_mu
    grads["vel"][i] += tau[:, None] * g_mu
    d_tau = np.sum(g.vel[i] * g_mu, axis=1)
    if spec.accel:
        grads["acc"][i] += (tau ** 2)[:, None] * g_mu
        d_tau += 2.0 * tau * np.sum(g.acc[i] * g_mu, axis=1)
    grads["colour"][i] += w[:, None] * g_colour
    grads["chol"][i] += r[:, None] * g_chol
    d_w = np.sum(g.colour[i] * g_colour, axis=1)
    if spec.shrink:
        d_s = np.sum(g.chol[i] * g_chol, axis=1) / (2.0 * r)
        d_w += np.where(w > s_min, d_s, 0.0)
    d_tau += d_w * (-w * tau / sigma ** 2)
    grads["t0"][i] -= d_tau
    grads["sigma_t"][i] += d_w * w * tau ** 2 / sigma ** 3
