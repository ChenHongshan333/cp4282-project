"""Space-time Gaussians for video, with the curved (acceleration) and shrinking extensions of VeGaS.

The 2D renderer and its hand-written gradients are the course's (``gaussian_image_video.py``, Unit
16) and are used unchanged. This package adds what the four models M0-M3 need on top of them:

    model.py      the stored parameters of a space-time Gaussian, the models, the storage budget
    slicing.py    slicing a space-time Gaussian at frame t, and its backward pass
    kernels.py    the course's render and gradient kernels, one thread per row, all frames in one launch
    render.py     rendering a clip and the gradient of the joint loss (and a frame-by-frame reference
                  that calls the course's functions unchanged)
    fit.py        Adam over all frames of a clip
    data.py       UVG loader with frame stride, and the synthetic clip
    metrics.py    PSNR, SSIM, and PSNR inside and outside a mask
    flow.py       optical-flow motion statistics and motion masks
    run.py        one fit:                     python -m stg.run --help
    sweep.py      the experiment design:       python -m stg.sweep --help
    check.py      finite-difference checks:    python -m stg.check
    aggregate.py  runs/*/metrics.json -> CSV:  python -m stg.aggregate runs
"""
