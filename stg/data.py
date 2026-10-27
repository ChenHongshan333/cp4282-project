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
