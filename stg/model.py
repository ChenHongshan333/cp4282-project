"""The stored parameters of space-time Gaussians, the four models, and the storage budget.

Every Gaussian stores its centre mu (2 numbers), peak time t0 (1), duration sigma_t (1), velocity
v (2), the Cholesky factor of its slice covariance (3) and its colour (3): 12 numbers. The models
with acceleration (M1, M3) also store a (2), so 14. All numbers are float32, so a budget of B
stored numbers is 4B bytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    accel: bool       # centre mu + v tau + a tau^2
    shrink: bool      # slice covariance scaled by max(w, s_min)

    @property
    def numbers_per_gaussian(self) -> int:
        return 12 + (2 if self.accel else 0)

    def count(self, budget: int) -> int:
        """Number of Gaussians that fit in a budget of stored numbers."""
        return budget // self.numbers_per_gaussian


MODELS = {s.name: s for s in (
    ModelSpec("M0", accel=False, shrink=False),     # baseline, Unit 16
    ModelSpec("M1", accel=True, shrink=False),
    ModelSpec("M2", accel=False, shrink=True),
    ModelSpec("M3", accel=True, shrink=True),
)}

# Shapes per Gaussian of every stored attribute. ``acc`` exists only for models with acceleration.
SHAPES = {"mu": (2,), "vel": (2,), "acc": (2,), "t0": (), "sigma_t": (), "chol": (3,), "colour": (3,)}


class SpaceTimeGaussians:
    """Parameters of N space-time Gaussians as NumPy arrays, keyed by attribute name."""

    def __init__(self, spec: ModelSpec, **arrays):
        self.spec = spec
        names = [n for n in SHAPES if n != "acc" or spec.accel]
        missing = set(names) - set(arrays)
        if missing:
            raise ValueError(f"missing attributes {sorted(missing)} for {spec.name}")
        self.params = {n: np.array(arrays[n]) for n in names}

    def __getattr__(self, name):
        params = self.__dict__.get("params", {})
        if name in params:
            return params[name]
        raise AttributeError(name)

    def __len__(self):
        return len(self.params["mu"])

    def copy(self):
        return SpaceTimeGaussians(self.spec, **{n: a.copy() for n, a in self.params.items()})

    def astype(self, dtype):
        return SpaceTimeGaussians(self.spec, **{n: a.astype(dtype) for n, a in self.params.items()})

    def stored_numbers(self) -> int:
        return int(sum(a.size for a in self.params.values()))

    def save(self, path: Path):
        np.savez(path, model=self.spec.name, **{n: a.astype(np.float32) for n, a in self.params.items()})

    @staticmethod
    def load(path: Path) -> "SpaceTimeGaussians":
        data = np.load(path)
        spec = MODELS[str(data["model"])]
        return SpaceTimeGaussians(spec, **{n: data[n] for n in data.files if n != "model"})


def initial(spec: ModelSpec, budget: int, frames: np.ndarray, seed: int, sigma_t_init: float = 2.0,
            scale_factor: float = 0.8) -> SpaceTimeGaussians:
    """Random Gaussians for a clip ``frames`` of shape (T, H, W, 3), with velocity and acceleration 0.

    The shared parameters are drawn for the largest count of the budget, floor(B / 12), and every
    model takes the first ``spec.count(budget)`` of them. So for one seed, M1 starts from a prefix of
    the Gaussians of M0, and the initialisation differs between models only in the count.
    """
    n_frames, height, width, _ = frames.shape
    n_max = budget // 12
    n = spec.count(budget)
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0, 1, (n_max, 2)) * (width, height)
    t0 = rng.uniform(0, n_frames - 1, n_max)
    # The course sets the scale so that the Gaussians of one image tile it. A space-time Gaussian is
    # visible in about sqrt(2 pi) sigma_t frames, so a frame sees that share of them.
    per_frame = n_max * min(1.0, math.sqrt(2 * math.pi) * sigma_t_init / n_frames)
    scale = scale_factor * math.sqrt(width * height / per_frame)
    k = np.clip(np.round(t0).astype(int), 0, n_frames - 1)
    y = np.minimum(mu[:, 1].astype(int), height - 1)
    x = np.minimum(mu[:, 0].astype(int), width - 1)
    arrays = {
        "mu": mu,
        "vel": np.zeros((n_max, 2)),
        "acc": np.zeros((n_max, 2)),
        "t0": t0,
        "sigma_t": np.full(n_max, sigma_t_init),
        "chol": np.tile((scale, 0.0, scale), (n_max, 1)),
        "colour": 0.5 * frames[k, y, x],
    }
    if not spec.accel:
        del arrays["acc"]
    return SpaceTimeGaussians(spec, **{name: a[:n].astype(np.float32) for name, a in arrays.items()})
