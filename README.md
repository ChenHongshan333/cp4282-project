# Is Folding Worth the Storage?

[![Website](https://img.shields.io/badge/Website-project%20page-1c5cab?logo=githubpages&logoColor=white)](https://chenhongshan333.github.io/cp4282-project/)
[![Report](https://img.shields.io/badge/Report-PDF-b3261e?logo=readme&logoColor=white)](https://github.com/ChenHongshan333/cp4282-project/blob/master/cp4282_final_report.pdf)
[![Slides](https://img.shields.io/badge/Slides-Canva-00c4cc)](https://canva.link/tcalod79alsm9z4)
![Python](https://img.shields.io/badge/Python-3.13-3776ab?logo=python&logoColor=white)
![NVIDIA Warp](https://img.shields.io/badge/NVIDIA-Warp-76b900?logo=nvidia&logoColor=white)
![Course](https://img.shields.io/badge/NUS-CP4282-ef7c00)
![Fits](https://img.shields.io/badge/fits-1%2C496-6b6a65)

CP4282 project: do the curved (acceleration) and shrinking extensions of space-time Gaussians
(VeGaS) improve video fitting once the number of stored parameters is held equal?

**[Website](https://chenhongshan333.github.io/cp4282-project/)** | 
**[Report (PDF)](https://github.com/ChenHongshan333/cp4282-project/blob/master/cp4282_final_report.pdf)** | 
**[Slides](https://canva.link/tcalod79alsm9z4)**

The website has the full report with interactive demos, reconstruction videos, every table as CSV
and the record of all fits.

## Key findings

- **Reconstruction at equal storage (Q1).** Folding gives no lasting gain. After the planned 3,000
  steps, shrinking (M2) and both abilities together (M3) beat the straight-line baseline by 0.5 to
  2.3 dB, but with full training the baseline catches up and all three fail the test fixed in the
  proposal.
- **Where the early gains come from (Q2).** They are more consistent with faster optimization than
  with better motion modelling: similar gains appear on still photographs, and the learned
  accelerations do not consistently follow the real motion.
- **Frames between the fitted ones (Q3).** A curved path draws the unseen frames better: up to about
  9 dB on synthetic arcs, growing with the bend, and about 0.5 dB on the moving object in new real
  clips, with no consistent gain over the whole frame.

## Models

| Model | Slice centre μ(t) | Slice covariance Σ(t) | Numbers per Gaussian |
|---|---|---|---|
| M0 (Unit 16) | μ + vτ | L Lᵀ | 12 |
| M1 | μ + vτ + aτ² | L Lᵀ | 14 |
| M2 | μ + vτ | max(w, s_min) · L Lᵀ | 12 |
| M3 | μ + vτ + aτ² | max(w, s_min) · L Lᵀ | 14 |

τ = t − t₀ and w is the temporal weight. At a given budget of stored numbers, a model with more
numbers per Gaussian has fewer Gaussians.

## Repository layout

```
gaussian_image_video.py   course program (Unit 16): 2D renderer and gradient kernels, unchanged
stg/                      this project; stg/__init__.py gives one line per module
  model.py, slicing.py    the four models and slicing a space-time Gaussian at time t
  kernels.py, render.py   the course kernels with all frames in one launch; `stg.check` compares them
  fit.py, run.py          one fit
  sweep.py                every experiment of the project, by stage
  unseen.py               PSNR on the frames between the fitted ones
  analyze_*.py            analyses of each stage
  figures.py, appendix_tables.py, export_site.py, export_videos.py
                          figures, tables, data and videos of the website and the report
docs/                     the website (GitHub Pages): index.html, figures/, data/, media/recon/
report.md                 the report text; cp4282_final_report.pdf is the final version
PROGRESS.md               working log of every step (in Chinese)
slurm/array.sh            one fit per Slurm array task
data/uvg/<Sequence>/      UVG .yuv file or frame images (not in the repository)
data/masks/<Sequence>/    hand-drawn object masks, <frame>.png (not in the repository)
runs/<stage>/<run>/       config.json, metrics.json, params.npz (not in the repository)
```

## Setup

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
```

A CUDA GPU is used for fitting (`--device cuda`); the UVG sequences go in `data/uvg/`.

## Reproduce

Each experiment resumes where it stopped. The section of the report each command belongs to is on
the right.

```
python -m stg.check --device cuda                          # gradient checks                (4.1)
python -m stg.sweep --stage test --all --device cuda       # main experiment, 3,000 steps   (5.1.1)
python -m stg.sweep --stage still --all --device cuda      # still photographs              (5.2.2)
python -m stg.sweep --stage long_test --all --device cuda  # main experiment, fully trained (5.1.2)
python -m stg.sweep --stage poly --all --device cuda       # degree 3 and 7                 (5.1.2)
python -m stg.unseen runs/long_test                        # frames between                 (5.3.1)
python -m stg.sweep --stage arc_slow --all --device cuda   # synthetic arcs                 (5.3.2)
python -m stg.sweep --stage confirm --all --device cuda    # confirmatory test              (5.3.1)
python -m stg.export_site                                  # data of the website
python -m stg.appendix_tables                              # appendix tables and CSV files
python -m stg.figures --crops                              # figures and tables
python -m stg.export_videos                                # reconstruction videos of the website
```

`python -m stg.sweep --stage <stage> --list` lists the runs of a stage. On a Slurm cluster, one fit
per array task:

```
sbatch --array=0-$(( $(python -m stg.sweep --stage test --count) - 1 )) slurm/array.sh test
```
