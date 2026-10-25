"""Rendering a clip and the gradient of the joint loss, with the course's 2D kernels unchanged.

Every frame is sliced (``slicing.py``), rendered and differentiated by ``render`` and
``loss_and_gradients`` of ``gaussian_image_video.py``, and the 2D gradients are carried back to the
space-time parameters. The loss is the mean squared error over all frames of the clip, so each
frame's gradient is divided by the number of frames.
"""

from __future__ import annotations

import numpy as np

import gaussian_image_video as course
from stg.model import SpaceTimeGaussians
from stg.slicing import Slice, slice_at, slice_backward, TIME_CUT


def _gaussians_2d(sl: Slice):
    return course.Gaussians(sl.mu, sl.chol, sl.colour)


def render_frame(g: SpaceTimeGaussians, t: float, width: int, height: int, device, s_min: float,
                 cut=course.CUT, time_cut=TIME_CUT) -> np.ndarray:
    sl = slice_at(g, t, s_min, time_cut)
    if len(sl.index) == 0:
        return np.zeros((height, width, 3), np.float32)
    return course.render(_gaussians_2d(sl), width, height, device, cut)


def render_clip(g, n_frames, width, height, device, s_min, cut=course.CUT, time_cut=TIME_CUT) -> np.ndarray:
    return np.stack([render_frame(g, t, width, height, device, s_min, cut, time_cut) for t in range(n_frames)])


def loss_and_gradients(g: SpaceTimeGaussians, frames: np.ndarray, device, s_min: float,
                       cut=course.CUT, time_cut=TIME_CUT):
    """Mean squared error over all frames of ``frames`` (T, H, W, 3), its gradient for every stored
    attribute, and the rendered frames. Returns (loss, grads, images)."""
    n_frames, height, width, _ = frames.shape
    grads = {n: np.zeros_like(a) for n, a in g.params.items()}
    images = np.empty_like(frames)
    total = 0.0
    for t in range(n_frames):
        sl = slice_at(g, t, s_min, time_cut)
        if len(sl.index) == 0:
            images[t] = 0.0
            total += float(np.mean(frames[t] ** 2))
            continue
        loss, (g_mu, g_chol, g_colour), image = course.loss_and_gradients(_gaussians_2d(sl), frames[t], device, cut)
        images[t] = image
        total += loss
        slice_backward(g, sl, s_min, g_mu / n_frames, g_chol / n_frames, g_colour / n_frames, grads)
    return total / n_frames, grads, images
