r"""Fit images and a short video with 2D Gaussians.

Runnable example for Unit 16. An image is represented as a sum of 2D Gaussians, each with a centre
(2 numbers), a Cholesky factor of its covariance (3 numbers), and a colour (3 numbers). The
program fits the Gaussians by gradient descent with a hand-written backward pass, quantizes them,
compares the result with JPEG, and then fits consecutive frames of a video, either each frame from
random Gaussians or each frame starting from the Gaussians of the previous frame.

Usage (the photographs are from the Kodak image set; the video frames are the test views of the
Lego scene, which form a camera orbit):

    python gaussian_image_video.py out --images data/kodak/kodim13.png data/kodak/kodim19.png \
        data/kodak/kodim23.png --video data/lego          # about one hour on a laptop CPU
    python gaussian_image_video.py out --images data/kodak/kodim23.png --video data/lego --quick
    python gaussian_image_video.py --check                # finite-difference gradient check

Output in ``out/``: ``results.json`` with every number the unit reports, and PNG renders.

Rendering model. With pixel centre p, centre mu, and Cholesky factor L = [[l11, 0], [l21, l22]]:

    z = L^-1 (p - mu),    w = exp(-|z|^2 / 2),    C(p) = sum_i colour_i * w_i(p).

A Gaussian is evaluated only where |z| < CUT, which lies inside a box of CUT standard deviations.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
import warp as wp

CUT = 3.5          # a Gaussian is evaluated where |z| < CUT; exp(-CUT^2 / 2) = 0.002
MIN_SCALE = 0.3    # lower bound on l11 and l22, in pixels


@wp.kernel(enable_backward=False)
def render_kernel(
    mu: wp.array(dtype=wp.vec2),
    chol: wp.array(dtype=wp.vec3),
    colour: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    cut: float,
    image: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    l11 = chol[i][0]
    l21 = chol[i][1]
    l22 = chol[i][2]
    rx = cut * wp.abs(l11)
    ry = cut * wp.sqrt(l21 * l21 + l22 * l22)
    x0 = wp.max(int(wp.floor(mu[i][0] - rx)), 0)
    x1 = wp.min(int(wp.floor(mu[i][0] + rx)), width - 1)
    y0 = wp.max(int(wp.floor(mu[i][1] - ry)), 0)
    y1 = wp.min(int(wp.floor(mu[i][1] + ry)), height - 1)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            z1 = (float(x) + 0.5 - mu[i][0]) / l11
            z2 = (float(y) + 0.5 - mu[i][1] - l21 * z1) / l22
            q = z1 * z1 + z2 * z2
            if q < cut * cut:
                wp.atomic_add(image, y * width + x, wp.exp(-0.5 * q) * colour[i])


@wp.kernel(enable_backward=False)
def gradient_kernel(
    mu: wp.array(dtype=wp.vec2),
    chol: wp.array(dtype=wp.vec3),
    colour: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    cut: float,
    image_grad: wp.array(dtype=wp.vec3),      # dL/dC at every pixel
    mu_grad: wp.array(dtype=wp.vec2),
    chol_grad: wp.array(dtype=wp.vec3),
    colour_grad: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    l11 = chol[i][0]
    l21 = chol[i][1]
    l22 = chol[i][2]
    rx = cut * wp.abs(l11)
    ry = cut * wp.sqrt(l21 * l21 + l22 * l22)
    x0 = wp.max(int(wp.floor(mu[i][0] - rx)), 0)
    x1 = wp.min(int(wp.floor(mu[i][0] + rx)), width - 1)
    y0 = wp.max(int(wp.floor(mu[i][1] - ry)), 0)
    y1 = wp.min(int(wp.floor(mu[i][1] + ry)), height - 1)
    d_mu = wp.vec2(0.0, 0.0)
    d_chol = wp.vec3(0.0, 0.0, 0.0)
    d_colour = wp.vec3(0.0, 0.0, 0.0)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            z1 = (float(x) + 0.5 - mu[i][0]) / l11
            z2 = (float(y) + 0.5 - mu[i][1] - l21 * z1) / l22
            q = z1 * z1 + z2 * z2
            if q < cut * cut:
                w = wp.exp(-0.5 * q)
                g = image_grad[y * width + x]
                d_colour += w * g
                d_w = wp.dot(g, colour[i])          # dL/dw
                d_z1 = -w * z1 * d_w                # dL/dz1 through w only
                d_z2 = -w * z2 * d_w
                # z2 depends on z1, so collect the total derivative with respect to z1 first.
                d_z1_total = d_z1 - d_z2 * l21 / l22
                d_mu += wp.vec2(-d_z1_total / l11, -d_z2 / l22)
                d_chol += wp.vec3(-d_z1_total * z1 / l11, -d_z2 * z1 / l22, -d_z2 * z2 / l22)
    mu_grad[i] = d_mu
    chol_grad[i] = d_chol
    colour_grad[i] = d_colour


class Gaussians:
    """Parameters of M Gaussians as NumPy arrays: mu (M, 2), chol (M, 3), colour (M, 3)."""

    def __init__(self, mu, chol, colour):
        self.mu, self.chol, self.colour = (np.asarray(a, np.float32).copy() for a in (mu, chol, colour))

    def copy(self):
        return Gaussians(self.mu, self.chol, self.colour)

    def __len__(self):
        return len(self.mu)

    @staticmethod
    def random(count, target, rng):
        height, width, _ = target.shape
        mu = rng.uniform(0, 1, (count, 2)) * (width, height)
        scale = 0.8 * math.sqrt(width * height / count)
        chol = np.tile((scale, 0.0, scale), (count, 1))
        at = target[np.minimum(mu[:, 1].astype(int), height - 1), np.minimum(mu[:, 0].astype(int), width - 1)]
        return Gaussians(mu, chol, 0.5 * at)


def render(g, width, height, device, cut=CUT):
    image = wp.zeros(width * height, dtype=wp.vec3, device=device)
    wp.launch(render_kernel, len(g), device=device, inputs=[
        wp.array(g.mu, dtype=wp.vec2, device=device), wp.array(g.chol, dtype=wp.vec3, device=device),
        wp.array(g.colour, dtype=wp.vec3, device=device), width, height, cut], outputs=[image])
    return image.numpy().reshape(height, width, 3)


def loss_and_gradients(g, target, device, cut=CUT):
    """Mean squared error over the 3WH colour components, and its gradient for every parameter."""
    height, width, _ = target.shape
    image = render(g, width, height, device, cut)
    diff = image - target
    image_grad = (2.0 / diff.size) * diff
    outs = [wp.zeros(len(g), dtype=t, device=device) for t in (wp.vec2, wp.vec3, wp.vec3)]
    wp.launch(gradient_kernel, len(g), device=device, inputs=[
        wp.array(g.mu, dtype=wp.vec2, device=device), wp.array(g.chol, dtype=wp.vec3, device=device),
        wp.array(g.colour, dtype=wp.vec3, device=device), width, height, cut,
        wp.array(image_grad.reshape(-1, 3), dtype=wp.vec3, device=device)], outputs=outs)
    return float(np.mean(diff ** 2)), [o.numpy() for o in outs], image


def psnr(image, target):
    return float(-10.0 * np.log10(np.mean((np.clip(image, 0, 1) - target) ** 2)))


# --- quantization: uniform steps on a fixed range, the same for every frame -------------------

BITS = {"mu": 12, "chol": 10, "colour": 6}
BITS_PER_GAUSSIAN = 2 * BITS["mu"] + 3 * BITS["chol"] + 3 * BITS["colour"]


def ranges(width, height):
    """Range of each attribute channel. Positions cover the image; the Cholesky entries and the
    colours use fixed ranges, so that the integers of two frames can be subtracted."""
    return {"mu": (np.zeros(2), np.array([width, height], float)),
            "chol": (np.array([MIN_SCALE, -8.0, MIN_SCALE]), np.array([16.0, 8.0, 16.0])),
            "colour": (np.full(3, -0.5), np.full(3, 1.5))}


def to_integers(g, rng_):
    out = {}
    for name in BITS:
        lo, hi = rng_[name]
        levels = 2 ** BITS[name] - 1
        out[name] = np.clip(np.round((getattr(g, name) - lo) / (hi - lo) * levels), 0, levels).astype(np.int64)
    return out


def from_integers(k, rng_):
    vals = {}
    for name in BITS:
        lo, hi = rng_[name]
        vals[name] = lo + k[name] / (2 ** BITS[name] - 1) * (hi - lo)
    return Gaussians(vals["mu"], vals["chol"], vals["colour"])


def quantized(g, rng_):
    return from_integers(to_integers(g, rng_), rng_)


def fit(g, target, iterations, device, rates=(0.25, 0.05, 0.01), quantize=None, trace_every=0):
    """Adam on the three attribute groups. With ``quantize``, the forward pass uses the quantized
    values and the gradient is passed to the continuous values (straight-through estimator)."""
    params = [g.mu, g.chol, g.colour]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    trace = []
    for step in range(1, iterations + 1):
        used = quantized(g, quantize) if quantize else g
        loss, grads, image = loss_and_gradients(used, target, device)
        if trace_every and (step == 1 or step % trace_every == 0):
            trace.append((step - 1, psnr(image, target)))
        decay = 0.5 ** (step / max(iterations, 1) * 2.0)       # the rates fall to a quarter
        for p, grad, m, v, rate in zip(params, grads, first, second, rates):
            m[:] = 0.9 * m + 0.1 * grad
            v[:] = 0.999 * v + 0.001 * grad * grad
            p -= rate * decay * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        g.chol[:, 0] = np.maximum(g.chol[:, 0], MIN_SCALE)
        g.chol[:, 2] = np.maximum(g.chol[:, 2], MIN_SCALE)
    final = quantized(g, quantize) if quantize else g
    return psnr(render(final, target.shape[1], target.shape[0], device), target), trace


def entropy_bits(symbols):
    """Entropy of a set of integers in bits per value: the smallest average code length of a
    lossless code that uses one probability table for all of them (Unit 15)."""
    _, counts = np.unique(symbols, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def load_frames(data: Path, indices, size):
    from PIL import Image
    frames = []
    for k in indices:
        rgba = np.asarray(Image.open(data / "test" / f"r_{k}.png").convert("RGBA").resize((size, size), Image.LANCZOS),
                          np.float32) / 255.0
        frames.append(rgba[..., :3] * rgba[..., 3:] + (1.0 - rgba[..., 3:]))     # white background
    return frames


def save(path, image):
    from PIL import Image
    Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def jpeg_at_size(target, max_bytes):
    """The JPEG of highest quality that is not larger than max_bytes: (bytes, PSNR, quality)."""
    from PIL import Image
    image = Image.fromarray(np.round(target * 255).astype(np.uint8))
    best = None
    for quality in range(1, 96):
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=quality, optimize=True)
        if buffer.tell() <= max_bytes:
            decoded = np.asarray(Image.open(io.BytesIO(buffer.getvalue())), np.float32) / 255.0
            best = (buffer.tell(), psnr(decoded, target), quality)
    return best


def gradient_check(device):
    """Compare the hand-written gradients with central differences. The check uses a cut of 6
    standard deviations, where the Gaussian has fallen to 1.5e-8: with the cut of 3.5 used for
    fitting, a pixel that enters or leaves the support changes the loss by a small step that the
    numerical derivative sees and the analytic derivative does not."""
    rng = np.random.default_rng(1)
    target = rng.uniform(0, 1, (48, 48, 3)).astype(np.float32)
    g = Gaussians(rng.uniform(16, 32, (5, 2)),
                  np.column_stack([rng.uniform(1.5, 3, 5), rng.uniform(-1, 1, 5), rng.uniform(1.5, 3, 5)]),
                  rng.uniform(0.2, 0.8, (5, 3)))
    grads = loss_and_gradients(g, target, device, cut=6.0)[1]

    def loss(h):       # in double precision, so that the differences are not rounded away
        return float(np.mean((render(h, 48, 48, device, cut=6.0).astype(np.float64) - target) ** 2))

    worst = 0.0
    for name, grad in zip(("mu", "chol", "colour"), grads):
        scale = np.abs(grad).max()
        for index in np.ndindex(getattr(g, name).shape):
            values = []
            for sign in (1.0, -1.0):
                h = g.copy()
                getattr(h, name)[index] += sign * 1e-2
                values.append(loss(h))
            numeric = (values[0] - values[1]) / 2e-2
            worst = max(worst, abs(numeric - grad[index]) / scale)
    print(f"largest difference between the hand-written and numerical gradients, relative to the "
          f"largest gradient of each attribute: {worst:.4f}")
    assert worst < 0.01

def load_image(path: Path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def pan(image, frames, size, step):
    """A second shot for the scene-cut test: a window of size x size pixels that moves ``step``
    pixels per frame across a reduced copy of a photograph."""
    from PIL import Image
    height, width, _ = image.shape
    small = Image.fromarray(np.round(image * 255).astype(np.uint8)).resize(
        (round(width * size / height), size), Image.LANCZOS)
    small = np.asarray(small, np.float32) / 255.0
    return [small[:, k * step:k * step + size] for k in range(frames)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--images", type=Path, nargs="*", default=[], help="photographs to fit, at full resolution")
    parser.add_argument("--video", type=Path, help="Lego scene directory containing test/r_*.png")
    parser.add_argument("--size", type=int, default=256, help="video frames are resized to size x size pixels")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    wp.init()
    if args.check:
        return gradient_check(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4282)
    long_run, short_run = (150, 40) if args.quick else (3000, 300)
    results = {"bits_per_gaussian": BITS_PER_GAUSSIAN, "bits": BITS, "long": long_run, "short": short_run, "images": []}

    # 1. Photographs: fit each with two numbers of Gaussians, then quantize the larger model.
    counts = [300, 600] if args.quick else [2500, 10000]
    for path in args.images:
        target = load_image(path)
        height, width, _ = target.shape
        if args.quick:
            target = target[:96, :128]
            height, width = 96, 128
        entry = {"name": path.stem, "width": width, "height": height, "fits": []}
        for count in counts:
            g = Gaussians.random(count, target, rng)
            value, trace = fit(g, target, long_run, args.device, trace_every=short_run)
            entry["fits"].append({"gaussians": count, "psnr": value, "bpp_float32": count * 256 / (width * height),
                                  "trace": trace})
            print(path.stem, count, round(value, 2), flush=True)
            save(args.out / f"{path.stem}_{count}.png", render(g, width, height, args.device))
        rng_ = ranges(width, height)
        rounded = psnr(render(quantized(g, rng_), width, height, args.device), target)
        tuned, _ = fit(g, target, long_run // 4, args.device, rates=(0.05, 0.01, 0.002), quantize=rng_)
        save(args.out / f"{path.stem}_{count}_quantized.png", render(quantized(g, rng_), width, height, args.device))
        nbytes = math.ceil(count * BITS_PER_GAUSSIAN / 8)
        jpeg = jpeg_at_size(target, nbytes)
        entry["quantization"] = {"gaussians": count, "psnr_float32": value, "psnr_rounded": rounded,
                                 "psnr_fine_tuned": tuned, "bytes": nbytes, "bpp": nbytes * 8 / (width * height),
                                 "jpeg": None if jpeg is None else {"bytes": jpeg[0], "psnr": jpeg[1], "quality": jpeg[2]}}
        print(entry["quantization"], flush=True)
        results["images"].append(entry)
        (args.out / "results.json").write_text(json.dumps(results, indent=1))
    if args.video is None:
        return

    # 2. A video: every frame from random Gaussians (short and long), or from the previous frame.
    size = 64 if args.quick else args.size
    count = 400 if args.quick else 2000
    n_frames = 4 if args.quick else 16
    rng_ = ranges(size, size)
    video = load_frames(args.video, range(n_frames), size)
    rows, previous, previous_k = [], None, None
    for t, frame in enumerate(video):
        fresh = Gaussians.random(count, frame, rng)
        scratch_short, _ = fit(fresh.copy(), frame, short_run, args.device)
        scratch_long, _ = fit(fresh, frame, long_run, args.device)
        row = {"frame": t, "scratch_short": scratch_short, "scratch_long": scratch_long}
        if previous is None:
            current = fresh.copy()
            row["predicted"] = scratch_long
        else:
            current = previous.copy()
            row["start"] = psnr(render(current, size, size, args.device), frame)
            row["predicted"], _ = fit(current, frame, short_run, args.device, rates=(0.1, 0.02, 0.005))
        k = to_integers(current, rng_)
        row["bits_values"] = sum(entropy_bits(k[name][:, c]) for name in BITS for c in range(k[name].shape[1]))
        if previous_k is not None:
            row["bits_differences"] = sum(entropy_bits(k[name][:, c] - previous_k[name][:, c])
                                          for name in BITS for c in range(k[name].shape[1]))
            for name in BITS:
                row[f"bits_differences_{name}"] = sum(entropy_bits(k[name][:, c] - previous_k[name][:, c])
                                                      for c in range(k[name].shape[1]))
                row[f"bits_values_{name}"] = sum(entropy_bits(k[name][:, c]) for c in range(k[name].shape[1]))
            row["unchanged_fraction"] = float(np.mean(np.concatenate(
                [(k[name] == previous_k[name]).ravel() for name in BITS])))
        rows.append(row)
        previous, previous_k = current, k
        print(row, flush=True)
        if t in (0, 1, n_frames - 1):
            save(args.out / f"video_{t}_predicted.png", render(current, size, size, args.device))
            save(args.out / f"video_{t}_target.png", frame)
    results["video"] = {"gaussians": count, "size": size, "frames": rows}
    (args.out / "results.json").write_text(json.dumps(results, indent=1))

    # 3. A scene cut: the last frames of the orbit, then a different shot. Every frame is fitted
    #    twice with the same number of iterations: from random Gaussians, and from the previous frame.
    before = 3 if args.quick else 5
    if args.images:
        second = pan(load_image(args.images[-1]), before, size, 3)
    else:
        second = load_frames(args.video, range(100, 100 + before), size)
    shots = video[-before:] + second
    rows, previous = [], None
    for t, frame in enumerate(shots):
        fresh = Gaussians.random(count, frame, rng)
        intra, _ = fit(fresh.copy(), frame, short_run, args.device)
        row = {"frame": t, "intra": intra}
        if previous is None:
            previous = fresh
            fit(previous, frame, long_run, args.device)
        else:
            row["start"] = psnr(render(previous, size, size, args.device), frame)
            row["predicted"], _ = fit(previous, frame, short_run, args.device, rates=(0.1, 0.02, 0.005))
            row["loss_difference"] = 10 ** (-row["predicted"] / 10) - 10 ** (-intra / 10)
        rows.append(row)
        print(row, flush=True)
        save(args.out / f"cut_{t}_target.png", frame)
    results["cut"] = {"cut_at": before, "frames": rows}
    (args.out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
