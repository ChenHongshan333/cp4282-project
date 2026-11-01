"""Clips for the experiments: UVG sequences with a frame stride, and a synthetic clip.

A UVG sequence is read from ``<data_root>/<Sequence>/``, which holds either the raw 8-bit 4:2:0 YUV
file of the UVG download (the size is read from its name, e.g. Jockey_1920x1080_120fps_420_8bit_YUV.yuv)
or the frames as numbered PNG/JPEG images. A clip with stride s holds frames k, k + s, ..., and is
re-indexed as t = 0, 1, ...
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SIZE = (480, 270)           # width, height, as in Unit 16
# Bosphorus was added after the motion measurement (PROGRESS.md, step 12): YachtRide's water makes
# its motion less straight than planned, and Bosphorus has the smallest acceleration of all clips.
SEQUENCES = ("Jockey", "ShakeNDry", "YachtRide", "Bosphorus")


def frame_indices(start: int, n_frames: int, stride: int) -> list[int]:
    return [start + k * stride for k in range(n_frames)]


def _resize(rgb_uint8: np.ndarray, size) -> np.ndarray:
    from PIL import Image
    image = Image.fromarray(rgb_uint8)
    if image.size != tuple(size):
        image = image.resize(tuple(size), Image.LANCZOS)
    return np.asarray(image, np.float32) / 255.0


def _yuv420_to_rgb(raw: np.ndarray, width: int, height: int) -> np.ndarray:
    """One frame of 8-bit limited-range BT.709 4:2:0 YUV to 8-bit RGB."""
    n = width * height
    y = raw[:n].reshape(height, width).astype(np.float32)
    u = raw[n:n + n // 4].reshape(height // 2, width // 2).astype(np.float32)
    v = raw[n + n // 4:].reshape(height // 2, width // 2).astype(np.float32)
    u = u.repeat(2, 0).repeat(2, 1)
    v = v.repeat(2, 0).repeat(2, 1)
    y = (y - 16.0) / 219.0
    cb = (u - 128.0) / 224.0
    cr = (v - 128.0) / 224.0
    rgb = np.stack([y + 1.5748 * cr, y - 0.1873 * cb - 0.4681 * cr, y + 1.8556 * cb], axis=-1)
    return np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _read_yuv(path: Path, indices, size) -> np.ndarray:
    match = re.search(r"(\d+)x(\d+)", path.name)
    if match is None:
        raise ValueError(f"cannot read the frame size from the name {path.name}")
    width, height = int(match[1]), int(match[2])
    frame_bytes = width * height * 3 // 2
    frames = []
    with open(path, "rb") as f:
        for k in indices:
            f.seek(k * frame_bytes)
            raw = np.frombuffer(f.read(frame_bytes), np.uint8)
            if raw.size != frame_bytes:
                raise ValueError(f"{path.name} has no frame {k}")
            frames.append(_resize(_yuv420_to_rgb(raw, width, height), size))
    return np.stack(frames)


def _read_images(paths: list[Path], indices, size) -> np.ndarray:
    from PIL import Image
    return np.stack([_resize(np.asarray(Image.open(paths[k]).convert("RGB")), size) for k in indices])


def load_clip(data_root: Path, sequence: str, start: int, n_frames: int, stride: int, size=SIZE) -> np.ndarray:
    """Frames start, start + stride, ... of a sequence, as float32 (T, H, W, 3) in [0, 1]."""
    if sequence == "synthetic":
        return synthetic_clip(n_frames, stride, size)[0]
    if sequence.startswith("arc"):
        return arc_clip(sequence, start, n_frames, stride, size)[0]
    if sequence.startswith("still_"):
        # A clip with no motion at all: frame ``start`` of the sequence, repeated (PROGRESS.md, plan A).
        frame = load_clip(data_root, sequence[len("still_"):], start, 1, 1, size)
        return np.repeat(frame, n_frames, axis=0)
    folder = Path(data_root) / sequence
    indices = frame_indices(start, n_frames, stride)
    yuv = sorted(folder.glob("*.yuv"))
    if yuv:
        return _read_yuv(yuv[0], indices, size)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not images:
        raise FileNotFoundError(f"no .yuv file or images in {folder}")
    return _read_images(images, indices, size)


# --- synthetic clip: blobs on known paths over a static background ------------------------------

def blob_paths(t: np.ndarray, width: int, height: int) -> dict[str, np.ndarray]:
    """Centres (len(t), 2) of the blobs at (fractional) frame times t, in pixels."""
    t = np.asarray(t, np.float64)
    return {
        # thrown ball: constant acceleration downward
        "parabola": np.stack([0.15 * width + 9.0 * t, 0.25 * height + -12.0 * t + 1.6 * t * t], axis=1),
        # swinging back and forth, like a leg
        "swing": np.stack([0.55 * width + 40.0 * np.sin(2 * np.pi * t / 12.0), 0.65 * height + 0 * t], axis=1),
        # control: constant velocity
        "straight": np.stack([0.75 * width - 6.0 * t, 0.3 * height + 3.0 * t], axis=1),
    }


def synthetic_clip(n_frames: int = 10, stride: int = 1, size=SIZE):
    """A clip of coloured soft discs moving along ``blob_paths`` over a smooth static background.
    Returns (frames, paths), where paths[name] holds the true centres at the clip's frames."""
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32) + 0.5
    background = np.stack([0.3 + 0.3 * xx / width,
                           0.4 + 0.1 * np.sin(2 * np.pi * yy / height * 2),
                           0.5 - 0.2 * yy / height], axis=-1)
    colours = {"parabola": (0.9, 0.2, 0.1), "swing": (0.1, 0.8, 0.2), "straight": (0.2, 0.3, 0.95)}
    radius = 0.05 * height
    t = np.arange(n_frames) * stride
    paths = blob_paths(t, width, height)
    frames = np.repeat(background[None], n_frames, axis=0)
    for name, centres in paths.items():
        for k, (cx, cy) in enumerate(centres):
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            alpha = np.clip(radius + 1.0 - d, 0, 1)[..., None]        # one-pixel soft edge
            frames[k] = (1 - alpha) * frames[k] + alpha * np.array(colours[name], np.float32)
    return frames.astype(np.float32), paths


# --- arc clips: one ball on a circular arc of chosen bend (PROGRESS.md, plan 2e) ----------------

# The ball moves at a fixed speed along an arc that turns by ``angle`` degrees over frames
# 0 .. ARC_FRAMES, the span of a 10-frame clip at stride 3; angle 0 is a straight line. Every arc of
# a style has the same length and speed, so its clips differ only in how much the path bends. A clip
# is named <style><angle>, e.g. arc180 or arcs180.
#   arc    plan 2e: 30 pixels between the fitted frames and stripes of period 9, too fast to learn
#          (PROGRESS.md, step 28)
#   arcs   plan 2e': slower, a smaller ball and wider stripes, so a step is under half a period
ARC_ANGLES = (0, 90, 180, 270, 360)
ARC_FRAMES = 27
ARC_STYLES = {"arc": {"speed": 10.0, "radius": 0.08 * SIZE[1], "period": 9.0},
              "arcs": {"speed": 3.0, "radius": 8.0, "period": 30.0}}


def arc_style(sequence: str) -> tuple[dict, float]:
    """The style and the angle of an arc clip's name."""
    style = sequence.rstrip("0123456789.")
    return ARC_STYLES[style], float(sequence[len(style):])


def arc_centres(angle: float, t: np.ndarray, size=SIZE, speed: float = ARC_STYLES["arc"]["speed"]) -> np.ndarray:
    """Centres (len(t), 2) of the ball at (fractional) frame times t, in pixels."""
    width, height = size
    t = np.asarray(t, np.float64)
    length = speed * ARC_FRAMES
    u = speed * t - length / 2                      # arc length from the middle of the path
    if angle == 0:
        x, y = u, 0 * u
    else:
        radius = length / np.radians(angle)
        x, y = radius * np.sin(u / radius), radius * (1 - np.cos(u / radius))
    # centre the box of the whole path in the frame
    span = np.linspace(-length / 2, length / 2, 200)
    ys = 0 * span if angle == 0 else radius * (1 - np.cos(span / radius))
    return np.stack([x + width / 2, y + height / 2 - (ys.max() + ys.min()) / 2], axis=1)


def arc_clip(sequence: str, start: int, n_frames: int, stride: int, size=SIZE):
    """Frames start, start + stride, ... of the arc clip, and the masks of the ball. The ball has
    stripes that move with it (it translates, without turning), over the smooth static background of
    ``synthetic_clip``."""
    style, angle = arc_style(sequence)
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32) + 0.5
    background = np.stack([0.3 + 0.3 * xx / width,
                           0.4 + 0.1 * np.sin(2 * np.pi * yy / height * 2),
                           0.5 - 0.2 * yy / height], axis=-1)
    radius = style["radius"]
    centres = arc_centres(angle, np.array(frame_indices(start, n_frames, stride)), size, style["speed"])
    frames = np.repeat(background[None], n_frames, axis=0)
    masks = np.zeros((n_frames, height, width), bool)
    for k, (cx, cy) in enumerate(centres):
        dx, dy = xx - cx, yy - cy
        d = np.sqrt(dx ** 2 + dy ** 2)
        alpha = np.clip(radius + 1.0 - d, 0, 1)[..., None]        # one-pixel soft edge
        stripes = 0.5 + 0.5 * np.sin(2 * np.pi * (dx + 0.5 * dy) / style["period"])
        ball = np.stack([0.9 - 0.4 * stripes, 0.2 + 0.6 * stripes, 0.15 + 0 * stripes], axis=-1)
        frames[k] = (1 - alpha) * frames[k] + alpha * ball
        masks[k] = d < radius + 1.0
    return frames.astype(np.float32), masks
