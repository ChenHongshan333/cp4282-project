# Is Folding Worth the Storage?

CP4282 project: do the curved (acceleration) and shrinking extensions of space-time Gaussians
(VeGaS) improve video fitting once the number of stored parameters is held equal?

| Model | Slice centre μ(t) | Slice covariance Σ(t) | Numbers per Gaussian |
|---|---|---|---|
| M0 (Unit 16) | μ + vτ | L Lᵀ | 12 |
| M1 | μ + vτ + aτ² | L Lᵀ | 14 |
| M2 | μ + vτ | max(w, s_min) · L Lᵀ | 12 |
| M3 | μ + vτ + aτ² | max(w, s_min) · L Lᵀ | 14 |

## Layout

```
gaussian_image_video.py   course program (Unit 16): 2D renderer and gradient kernels, used unchanged
stg/                      this project (see stg/__init__.py for one line per module)
slurm/array.sh            one fit per Slurm array task
data/uvg/<Sequence>/      UVG .yuv file or frame images (not in the repository)
data/masks/<Sequence>/    hand-drawn masks, <frame>.png (optional; flow masks otherwise)
runs/<run>/               config.json, metrics.json, params.npz, PNGs of the worst frame
```

## Setup

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
```

## Commands

```
python -m stg.check --device cpu                                     # gradient checks, all models
python -m stg.run --model M1 --budget 6000 --sequence synthetic --iterations 300 --device cuda
python -m stg.sweep --stage test --list                              # every run of the main experiment
sbatch --array=0-335 slurm/array.sh test                             # on the cluster
python -m stg.aggregate runs                                         # runs/results.csv, runs/summary.csv
```

## Decisions to make before the main runs

- `stg/sweep.py`: `VALIDATION_START` (the Jockey validation segment), then `CHOSEN` after the
  validation stage.
- `stg/fit.py`: learning rates of `vel`, `t0`, `sigma_t` (the course gives only `mu`, `chol`,
  `colour`), and `SIGMA_T_MIN`.
- Shrinking and the course's lower bound of 0.3 px: the bound is applied to the stored Cholesky
  factor, so a sliced factor can be √s_min · 0.3 px (0.07 px for s_min = 0.05), smaller than a pixel.
- The loss is on the unclamped render, as in the course; PSNR and SSIM are on the render clamped to [0, 1].
