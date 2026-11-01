"""Object masks for the per-region PSNR, from a pretrained instance segmentation model.

Optical-flow masks find the moving objects only when the camera is still (PROGRESS.md, step 11), so
the objects are segmented instead: torchvision's Mask R-CNN (ResNet-50 FPN v2) trained on COCO, which
knows the classes of the four sequences. Each frame is segmented at 960x540 and the union of the
instances of the sequence's classes with score >= SCORE is kept, reduced to the 480x270 of the fits.
Writes ``<mask_root>/<Sequence>/<frame>.png`` (white = object), which ``stg.run`` reads, and an
overlay image per sequence for checking.

    python -m stg.masks                        # frames 0-29 of every sequence, on the CPU
    python -m stg.masks --sequences Jockey --start 120 --frames 46
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from stg import data

# COCO category ids in torchvision's numbering
PERSON, BOAT, DOG, HORSE = 1, 9, 18, 19
CLASSES = {"Jockey": (HORSE, PERSON), "ShakeNDry": (DOG,), "YachtRide": (BOAT, PERSON), "Bosphorus": (BOAT, PERSON)}
SCORE = 0.5
# Classes tried when a frame has none of the sequence's classes: in a few blurred frames of the shaking
# dog the model sees a horse instead (there is none in the sequence).
FALLBACK = {"ShakeNDry": (HORSE,)}
SEGMENT_SIZE = (960, 540)


def load_model():
    os.environ.setdefault("TORCH_HOME", str(Path("data/torch").resolve()))     # weights on D:, not in ~/.cache
    import torch
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    torch.set_num_threads(4)            # leave CPU for the fits running on the GPU
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval()
    return model


def segment(model, frame: np.ndarray, classes) -> tuple[np.ndarray, list]:
    """Boolean mask (H, W) of ``frame`` (H, W, 3) in [0, 1], and the (class, score) of what it kept."""
    import torch
    with torch.no_grad():
        out = model([torch.from_numpy(frame).permute(2, 0, 1).float()])[0]
    keep = [(int(l), float(s)) for l, s in zip(out["labels"], out["scores"]) if int(l) in classes and s >= SCORE]
    mask = np.zeros(frame.shape[:2], bool)
    for i, (l, s) in enumerate(zip(out["labels"], out["scores"])):
        if int(l) in classes and s >= SCORE:
            mask |= out["masks"][i, 0].numpy() > 0.5
    return mask, keep


def largest_component(mask: np.ndarray) -> np.ndarray:
    """The largest 8-connected region of ``mask``. The main object and what touches it (the jockey
    on the horse, the people in the boat) form one region; distant boats and the cars on the bridge
    in Bosphorus are separate small regions and are dropped."""
    import cv2
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def overlay(frame, mask):
    tinted = frame.copy()
    tinted[mask] = 0.5 * tinted[mask] + 0.5 * np.array([1.0, 0.1, 0.1])
    return tinted


def overlay_name(a):
    return "overlay.png" if a.start == 0 else f"overlay_{a.start:04d}.png"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sequences", nargs="*", default=list(data.SEQUENCES))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--frames", type=int, default=30, help="frames start .. start+frames-1 of each sequence")
    p.add_argument("--data-root", type=Path, default=Path("data/uvg"))
    p.add_argument("--mask-root", type=Path, default=Path("data/masks"))
    p.add_argument("--clean-only", action="store_true",
                   help="only reduce the saved masks to their largest region, without segmenting again")
    a = p.parse_args(argv)
    from PIL import Image
    width, height = data.SIZE
    if a.clean_only:
        for sequence in a.sequences:
            folder = a.mask_root / sequence
            small = data.load_clip(a.data_root, sequence, a.start, a.frames, 1)
            shares, tiles = [], []
            for i, k in enumerate(range(a.start, a.start + a.frames)):
                mask = largest_component(np.asarray(Image.open(folder / f"{k:04d}.png")) > 127)
                Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"{k:04d}.png")
                shares.append(mask.mean())
                if i % 9 == 0:
                    tiles.append(overlay(small[i], mask))
            Image.fromarray(np.round(np.concatenate(tiles, axis=1) * 255).astype(np.uint8)).save(folder / overlay_name(a))
            print(f"{sequence}: object share of the frame {np.mean(shares):.1%} "
                  f"(min {np.min(shares):.1%}, max {np.max(shares):.1%})", flush=True)
        return
    model = load_model()
    for sequence in a.sequences:
        frames = data.load_clip(a.data_root, sequence, a.start, a.frames, 1, size=SEGMENT_SIZE)
        small = data.load_clip(a.data_root, sequence, a.start, a.frames, 1)
        folder = a.mask_root / sequence
        folder.mkdir(parents=True, exist_ok=True)
        shares, tiles = [], []
        for i, (k, frame) in enumerate(zip(range(a.start, a.start + a.frames), frames)):
            mask, kept = segment(model, frame, CLASSES[sequence])
            if not kept and sequence in FALLBACK:
                mask, kept = segment(model, frame, FALLBACK[sequence])
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.BILINEAR)) > 127
            mask = largest_component(mask)
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"{k:04d}.png")
            shares.append(mask.mean())
            if i % 9 == 0:
                tiles.append(overlay(small[i], mask))
            print(f"{sequence} frame {k}: {len(kept)} objects {[(l, round(s, 2)) for l, s in kept]}, "
                  f"{mask.mean():.1%} of the frame", flush=True)
        sheet = np.concatenate(tiles, axis=1)
        Image.fromarray(np.round(sheet * 255).astype(np.uint8)).save(folder / overlay_name(a))
        print(f"{sequence}: object share of the frame {np.mean(shares):.1%} "
              f"(min {np.min(shares):.1%}, max {np.max(shares):.1%})", flush=True)


if __name__ == "__main__":
    main()
