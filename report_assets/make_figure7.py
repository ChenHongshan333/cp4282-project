"""Figure 7 of report.md: the five synthetic arc clips at frames 0, 13 and 27.

Rows are the five turns (0, 90, 180, 270, 360 degrees); columns are frame 0 (fitted), frame 13 (never
fitted at stride 3) and frame 27 (fitted). The ball and its stripes are the pixels that
``stg.data.arc_clip`` draws (style ``arcs``, the clips of Section 5.3.2), recovered and laid over a
light background exactly as in ``stg.figures.fig_arc_motion`` (which made docs/figures/arc_motion.gif).
The grey line is the true path of the ball's centre over frames 0 to 27 (``stg.data.arc_centres``),
and the arrows show the direction of motion along it. Nothing is fitted or trained here.

Run from the repository root:  .venv/Scripts/python report_assets/make_figure7.py
Writes report_assets/figure7_motion_sequence.png only.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from stg.data import ARC_ANGLES, ARC_FRAMES, ARC_STYLES, arc_centres, arc_clip  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"      # the colours of stg/figures.py
BLUE, PATH = "#2a78d6", "#b9b7b0"
SHOWN = (0, 13, 27)                                     # frames in the three columns
STRIDE = 3                                              # fitted frames are 0, 3, ..., 27
ARROWS_AT = (6.5, 20.5)                                 # frame times where direction arrows sit
LIGHT = np.array([0.984, 0.980, 0.973])


def ball_layers(angle, style):
    """For each frame 0..27: the ball's colour and its alpha, recovered from the clip's pixels."""
    times = np.arange(ARC_FRAMES + 1)
    frames, masks = arc_clip(f"arcs{angle}", 0, len(times), 1)
    first_free = np.argmax(~masks, axis=0)
    background = np.take_along_axis(frames, first_free[None, ..., None].repeat(3, -1), axis=0)[0]
    centres = arc_centres(angle, times, speed=style["speed"])
    yy, xx = np.mgrid[0:frames.shape[1], 0:frames.shape[2]] + 0.5
    layers = []
    for t, (cx, cy) in enumerate(centres):
        alpha = np.clip(style["radius"] + 1.0 - np.hypot(xx - cx, yy - cy), 0, 1)[..., None]
        ball = np.where(alpha > 0, (frames[t] - (1 - alpha) * background) / np.maximum(alpha, 1e-6), 0)
        layers.append(np.concatenate([np.clip(ball, 0, 1), alpha], axis=-1))
    return layers, centres


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.bbox": "tight",
                         "savefig.pad_inches": 0.05})
    style = ARC_STYLES["arcs"]
    data = {}
    for angle in ARC_ANGLES:
        layers, centres = ball_layers(angle, style)
        path = arc_centres(angle, np.linspace(0, ARC_FRAMES, 400), speed=style["speed"])
        data[angle] = (layers, centres, path)
    # one window size for every panel, so all panels share the same pixel scale
    pad = style["radius"] + 6
    half_x = max(np.ptp(d[2][:, 0]) for d in data.values()) / 2 + pad
    half_y = max(np.ptp(d[2][:, 1]) for d in data.values()) / 2 + pad
    h, w = data[0][0][0].shape[:2]

    fig, axes = plt.subplots(len(ARC_ANGLES), len(SHOWN), figsize=(6.6, 6.4))
    fig.subplots_adjust(left=0.1, right=0.995, top=0.9, bottom=0.07, wspace=0.05, hspace=0.08)
    for r, angle in enumerate(ARC_ANGLES):
        layers, centres, path = data[angle]
        cx, cy = (path.max(0) + path.min(0)) / 2
        for c, t in enumerate(SHOWN):
            ax = axes[r, c]
            ax.imshow(np.broadcast_to(LIGHT, (h, w, 3)), interpolation="nearest", zorder=0)
            ax.plot(path[:, 0], path[:, 1], color=PATH, lw=1.3, zorder=1, solid_capstyle="round")
            for ta in ARROWS_AT:
                p0, p1 = arc_centres(angle, np.array([ta - 0.6, ta + 0.6]), speed=style["speed"])
                ax.annotate("", xy=p1, xytext=p0, zorder=2,
                            arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.3",
                                            color=MUTED, lw=1.1, shrinkA=0, shrinkB=0))
            ax.imshow(layers[t], interpolation="bilinear", zorder=3)
            ax.set_xlim(cx - half_x, cx + half_x)
            ax.set_ylim(cy + half_y, cy - half_y)
            ax.set_xticks([]); ax.set_yticks([])
            fitted = t % STRIDE == 0
            for side in ax.spines.values():
                side.set_color(BLUE if fitted else "#8a8984")
                side.set_linewidth(1.4 if fitted else 1.0)
                side.set_linestyle("-" if fitted else (0, (3, 2)))
            if c == 0:
                ax.set_ylabel(f"turn {angle}°", rotation=0, ha="right", va="center", fontsize=9.5,
                              color=INK, labelpad=6)
    heads = [("frame 0", "fitted", BLUE), ("frame 13", "unseen (between fitted frames)", MUTED),
             ("frame 27", "fitted", BLUE)]
    for c, (frame, kind, colour) in enumerate(heads):
        ax = axes[0, c]
        ax.text(0.5, 1.30, frame, transform=ax.transAxes, ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=INK)
        ax.text(0.5, 1.07, kind, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                color=colour)
    legend = [Line2D([], [], color=PATH, lw=1.3, label="true path of the ball's centre, frames 0 to 27"),
              Line2D([], [], color=MUTED, lw=0, marker=">", ms=6, label="direction of motion"),
              Line2D([], [], color=BLUE, lw=1.4, label="fitted frame (border)"),
              Line2D([], [], color="#8a8984", lw=1.0, ls=(0, (3, 2)), label="unseen frame (border)")]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=8.3,
               bbox_to_anchor=(0.55, -0.02), handlelength=2.2, columnspacing=1.6)
    out = ROOT / "report_assets" / "figure7_motion_sequence.png"
    fig.savefig(out, dpi=250, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
