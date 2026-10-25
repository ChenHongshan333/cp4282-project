"""Motion statistics and motion masks from Farnebaeck optical flow (OpenCV).

Used only to measure how much (and how curved) the motion of a clip is, and to split the pixels
into moving regions and background for the per-region PSNR.
"""

from __future__ import annotations

import numpy as np


def _gray(frame):
    return np.round(np.clip(frame, 0, 1) @ np.array([0.299, 0.587, 0.114]) * 255).astype(np.uint8)


def flows(frames: np.ndarray) -> np.ndarray:
    """Forward flow from frame k to k + 1, (T - 1, H, W, 2) in pixels."""
    import cv2
    gray = [_gray(f) for f in frames]
    return np.stack([cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 5, 7, 1.5, 0)
                     for a, b in zip(gray[:-1], gray[1:])])


def motion_stats(frames: np.ndarray) -> dict:
    """Mean displacement per frame (speed) and mean change of displacement between consecutive
    frame pairs (acceleration), with and without the median (camera) flow of each frame.

    The change is taken at the same pixel, not along the motion, so it is a rough measure."""
    f = flows(frames)
    camera = np.median(f.reshape(len(f), -1, 2), axis=1)[:, None, None, :]
    rel = f - camera
    out = {
        "speed": float(np.linalg.norm(f, axis=-1).mean()),
        "speed_relative": float(np.linalg.norm(rel, axis=-1).mean()),
        "camera_speed": float(np.linalg.norm(camera, axis=-1).mean()),
    }
    if len(f) > 1:
        out["acceleration"] = float(np.linalg.norm(np.diff(f, axis=0), axis=-1).mean())
        out["acceleration_relative"] = float(np.linalg.norm(np.diff(rel, axis=0), axis=-1).mean())
    return out


def motion_masks(frames: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """(T, H, W) boolean: pixels whose flow differs from the frame's median flow by more than
    ``threshold`` pixels. The last frame uses the flow from the frame before it."""
    f = flows(frames)
    f = np.concatenate([f, f[-1:]])
    camera = np.median(f.reshape(len(f), -1, 2), axis=1)[:, None, None, :]
    return np.linalg.norm(f - camera, axis=-1) > threshold
