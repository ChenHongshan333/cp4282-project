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


def _change_along_motion(f: np.ndarray) -> list[np.ndarray]:
    """For each pixel p of frame k, |d_{k+1}(p + d_k(p)) - d_k(p)|: how much the displacement of the
    point that starts at p changes over the next frame. Pixels that leave the frame are NaN."""
    import cv2
    height, width = f.shape[1:3]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    out = []
    for d, d_next in zip(f[:-1], f[1:]):
        mx, my = xx + d[..., 0], yy + d[..., 1]
        sampled = cv2.remap(d_next, mx, my, cv2.INTER_LINEAR)
        change = np.linalg.norm(sampled - d, axis=-1)
        change[(mx < 0) | (mx > width - 1) | (my < 0) | (my > height - 1)] = np.nan
        out.append(change)
    return out


def motion_stats(frames: np.ndarray) -> dict:
    """How fast and how curved the motion of a clip is, from the flow between consecutive frames.

    speed           mean displacement per frame, in pixels
    acceleration    mean change of displacement between consecutive frames, followed along the
                    motion (``_change_along_motion``), in pixels per frame^2
    curvature       acceleration / speed: how far the motion is from straight and uniform
    *_p90           the 90th percentile over pixels, for the fastest-changing tenth of the frame
    The camera is included: a steady pan adds speed but little acceleration."""
    f = flows(frames)
    speed = np.linalg.norm(f, axis=-1)
    change = np.stack(_change_along_motion(f))
    out = {"speed": float(speed.mean()), "speed_p90": float(np.percentile(speed, 90)),
           "acceleration": float(np.nanmean(change)), "acceleration_p90": float(np.nanpercentile(change, 90))}
    out["curvature"] = out["acceleration"] / max(out["speed"], 1e-6)
    return out


def motion_masks(frames: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """(T, H, W) boolean: pixels whose flow differs from the frame's median flow by more than
    ``threshold`` pixels. The last frame uses the flow from the frame before it."""
    f = flows(frames)
    f = np.concatenate([f, f[-1:]])
    camera = np.median(f.reshape(len(f), -1, 2), axis=1)[:, None, None, :]
    return np.linalg.norm(f - camera, axis=-1) > threshold
