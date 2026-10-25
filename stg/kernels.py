r"""The course's render and gradient kernels, with more threads and all frames of a clip in one launch.

The course kernels (``gaussian_image_video.py``) give one thread to each Gaussian, which loops over
every pixel of its box. With a thousand Gaussians of about 90 x 90 pixels, a few thousand threads
each do thousands of steps and most of the GPU is idle. Here thread (i, r) handles row r of the box
of Gaussian i, and every 2D Gaussian carries the index of its frame, so one launch covers the clip.

The arithmetic per pixel is the course's, line for line; ``stg.check`` compares the two.
"""

from __future__ import annotations

import warp as wp


@wp.func
def _box(m: wp.vec2, c: wp.vec3, width: int, height: int, cut: float):
    rx = cut * wp.abs(c[0])
    ry = cut * wp.sqrt(c[1] * c[1] + c[2] * c[2])
    x0 = wp.max(int(wp.floor(m[0] - rx)), 0)
    x1 = wp.min(int(wp.floor(m[0] + rx)), width - 1)
    y0 = wp.max(int(wp.floor(m[1] - ry)), 0)
    y1 = wp.min(int(wp.floor(m[1] + ry)), height - 1)
    return wp.vec4i(x0, x1, y0, y1)


@wp.kernel(enable_backward=False)
def render_rows_kernel(
    mu: wp.array(dtype=wp.vec2),
    chol: wp.array(dtype=wp.vec3),
    colour: wp.array(dtype=wp.vec3),
    frame: wp.array(dtype=int),
    width: int,
    height: int,
    cut: float,
    image: wp.array(dtype=wp.vec3),         # all frames, frame-major
):
    i, r = wp.tid()
    l11 = chol[i][0]
    l21 = chol[i][1]
    l22 = chol[i][2]
    box = _box(mu[i], chol[i], width, height, cut)
    y = box[2] + r
    if y > box[3]:
        return
    base = frame[i] * width * height + y * width
    for x in range(box[0], box[1] + 1):
        z1 = (float(x) + 0.5 - mu[i][0]) / l11
        z2 = (float(y) + 0.5 - mu[i][1] - l21 * z1) / l22
        q = z1 * z1 + z2 * z2
        if q < cut * cut:
            wp.atomic_add(image, base + x, wp.exp(-0.5 * q) * colour[i])


@wp.kernel(enable_backward=False)
def mse_gradient_kernel(
    image: wp.array(dtype=wp.vec3),
    target: wp.array(dtype=wp.vec3),
    scale: float,
    image_grad: wp.array(dtype=wp.vec3),
):
    k = wp.tid()
    image_grad[k] = scale * (image[k] - target[k])


@wp.kernel(enable_backward=False)
def gradient_rows_kernel(
    mu: wp.array(dtype=wp.vec2),
    chol: wp.array(dtype=wp.vec3),
    colour: wp.array(dtype=wp.vec3),
    frame: wp.array(dtype=int),
    width: int,
    height: int,
    cut: float,
    image_grad: wp.array(dtype=wp.vec3),     # dL/dC at every pixel of every frame
    mu_grad: wp.array(dtype=wp.vec2),        # accumulated over rows: zero before the launch
    chol_grad: wp.array(dtype=wp.vec3),
    colour_grad: wp.array(dtype=wp.vec3),
):
    i, r = wp.tid()
    l11 = chol[i][0]
    l21 = chol[i][1]
    l22 = chol[i][2]
    box = _box(mu[i], chol[i], width, height, cut)
    y = box[2] + r
    if y > box[3]:
        return
    base = frame[i] * width * height + y * width
    d_mu = wp.vec2(0.0, 0.0)
    d_chol = wp.vec3(0.0, 0.0, 0.0)
    d_colour = wp.vec3(0.0, 0.0, 0.0)
    for x in range(box[0], box[1] + 1):
        z1 = (float(x) + 0.5 - mu[i][0]) / l11
        z2 = (float(y) + 0.5 - mu[i][1] - l21 * z1) / l22
        q = z1 * z1 + z2 * z2
        if q < cut * cut:
            w = wp.exp(-0.5 * q)
            g = image_grad[base + x]
            d_colour += w * g
            d_w = wp.dot(g, colour[i])          # dL/dw
            d_z1 = -w * z1 * d_w                # dL/dz1 through w only
            d_z2 = -w * z2 * d_w
            # z2 depends on z1, so collect the total derivative with respect to z1 first.
            d_z1_total = d_z1 - d_z2 * l21 / l22
            d_mu += wp.vec2(-d_z1_total / l11, -d_z2 / l22)
            d_chol += wp.vec3(-d_z1_total * z1 / l11, -d_z2 * z1 / l22, -d_z2 * z2 / l22)
    wp.atomic_add(mu_grad, i, d_mu)
    wp.atomic_add(chol_grad, i, d_chol)
    wp.atomic_add(colour_grad, i, d_colour)
