"""Rendering a clip and the gradient of the joint loss.

``ClipRenderer`` slices every frame (``slicing.py``), renders and differentiates all frames in one
launch each with the row-parallel kernels of ``kernels.py``, keeps the frames on the GPU, and carries
the 2D gradients back to the space-time parameters. The loss is the mean squared error over all
frames of the clip.

``render_clip_reference`` and ``loss_and_gradients_reference`` do the same frame by frame with the
course's ``render`` and ``loss_and_gradients``, unchanged; ``stg.check`` compares the two.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import gaussian_image_video as course
from stg.kernels import gradient_rows_kernel, mse_gradient_kernel, render_rows_kernel
from stg.model import SpaceTimeGaussians
from stg.slicing import Slice, slice_at, slice_backward, TIME_CUT


class ClipRenderer:
    """Renders ``n_frames`` frames of ``height`` x ``width``; with ``targets`` (T, H, W, 3) it also
    gives the loss and its gradients. Buffers are allocated once and reused. The frames are at the
    clip times 0, 1, ..., or at ``times``, which may lie between them (``stg.unseen``)."""

    def __init__(self, n_frames, height, width, device, targets=None, cut=course.CUT, time_cut=TIME_CUT,
                 times=None):
        self.n_frames, self.height, self.width = n_frames, height, width
        self.times = list(range(n_frames)) if times is None else list(times)
        assert len(self.times) == n_frames
        self.device, self.cut, self.time_cut = device, cut, time_cut
        pixels = n_frames * height * width
        self.image = wp.zeros(pixels, dtype=wp.vec3, device=device)
        self.targets = None
        if targets is not None:
            assert targets.shape == (n_frames, height, width, 3)
            self.targets = targets
            self.target = wp.array(targets.reshape(-1, 3), dtype=wp.vec3, device=device)
            self.image_grad = wp.zeros(pixels, dtype=wp.vec3, device=device)

    def _slices(self, g, s_min):
        slices = [slice_at(g, t, s_min, self.time_cut) for t in self.times]
        mu = np.concatenate([s.mu for s in slices]).astype(np.float32)
        chol = np.concatenate([s.chol for s in slices]).astype(np.float32)
        colour = np.concatenate([s.colour for s in slices]).astype(np.float32)
        frame = np.concatenate([np.full(len(s.index), t, np.int32) for t, s in enumerate(slices)])
        return slices, mu, chol, colour, frame

    def _rows(self, chol):
        """Threads per Gaussian: an upper bound on the rows of the largest box."""
        ry = self.cut * np.sqrt(chol[:, 1] ** 2 + chol[:, 2] ** 2).max()
        return min(int(np.ceil(2 * ry)) + 2, self.height)

    def _render(self, mu, chol, colour, frame):
        self.image.zero_()
        if len(mu) == 0:
            return None
        arrays = (wp.array(mu, dtype=wp.vec2, device=self.device), wp.array(chol, dtype=wp.vec3, device=self.device),
                  wp.array(colour, dtype=wp.vec3, device=self.device), wp.array(frame, dtype=int, device=self.device))
        wp.launch(render_rows_kernel, (len(mu), self._rows(chol)), device=self.device,
                  inputs=[*arrays, self.width, self.height, self.cut], outputs=[self.image])
        return arrays

    def images(self) -> np.ndarray:
        return self.image.numpy().reshape(self.n_frames, self.height, self.width, 3)

    def render(self, g: SpaceTimeGaussians, s_min: float) -> np.ndarray:
        _, mu, chol, colour, frame = self._slices(g, s_min)
        self._render(mu, chol, colour, frame)
        return self.images()

    def loss_and_gradients(self, g: SpaceTimeGaussians, s_min: float, want_images=False):
        """Gradients of the mean squared error over all frames for every stored attribute.
        Returns (loss, grads, images); loss and images are computed only with ``want_images``,
        since they need the frames copied back from the GPU."""
        slices, mu, chol, colour, frame = self._slices(g, s_min)
        grads = {n: np.zeros_like(a) for n, a in g.params.items()}
        arrays = self._render(mu, chol, colour, frame)
        if arrays is not None:
            wp.launch(mse_gradient_kernel, self.image.shape[0], device=self.device,
                      inputs=[self.image, self.target, 2.0 / (self.image.shape[0] * 3)], outputs=[self.image_grad])
            outs = [wp.zeros(len(mu), dtype=t, device=self.device) for t in (wp.vec2, wp.vec3, wp.vec3)]
            wp.launch(gradient_rows_kernel, (len(mu), self._rows(chol)), device=self.device,
                      inputs=[*arrays, self.width, self.height, self.cut, self.image_grad], outputs=outs)
            g_mu, g_chol, g_colour = (o.numpy() for o in outs)
            start = 0
            for s in slices:
                end = start + len(s.index)
                slice_backward(g, s, s_min, g_mu[start:end], g_chol[start:end], g_colour[start:end], grads)
                start = end
        if not want_images:
            return None, grads, None
        images = self.images()
        return float(np.mean((images - self.targets) ** 2)), grads, images


# --- reference: frame by frame with the course's functions ----------------------------------

def _gaussians_2d(sl: Slice):
    return course.Gaussians(sl.mu, sl.chol, sl.colour)


def render_clip_reference(g, n_frames, width, height, device, s_min, cut=course.CUT, time_cut=TIME_CUT):
    frames = []
    for t in range(n_frames):
        sl = slice_at(g, t, s_min, time_cut)
        frames.append(np.zeros((height, width, 3), np.float32) if len(sl.index) == 0
                      else course.render(_gaussians_2d(sl), width, height, device, cut))
    return np.stack(frames)


def loss_and_gradients_reference(g: SpaceTimeGaussians, frames: np.ndarray, device, s_min: float,
                                 cut=course.CUT, time_cut=TIME_CUT):
    n_frames = len(frames)
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
