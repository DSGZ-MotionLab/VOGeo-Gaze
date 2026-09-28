# VOGeo-Gaze

Official inference code and weights for

> **VOGeo-Gaze: Calibration-Free, Geometry-Aware Deep Learning for Real-Time Gaze Tracking in Clinical Video-Oculography**
> Jingkang Zhao, Seyed-Ahmad Ahmadi, Julian Decker, Peter zu Eulenburg, Andreas Zwergal, Virginia L. Flanagin, Max Wuehr.
> MICCAI 2026.

VOGeo-Gaze estimates gaze from monocular eye video without subject calibration or camera intrinsics.
It does not regress gaze angles.  Instead it reconstructs the eye parameters: the eyeball centre
**c**<sub>eye</sub>, the optic axis **n** and the pupil radius r<sub>pup</sub>.  It takes them from the
projected pupil and iris, corrects them for corneal refraction and constrains them with a two-sphere
anatomical eye model.  Gaze then follows from the reconstructed optic axis.  On 116 clinical
EyeSeeCam recordings the paper reports median absolute errors of 0.33° horizontally and 0.35°
vertically, with more than 90 % of recordings below 1°, at more than 300 frames/s.

## Pipeline

| Step (paper, Fig. 2) | Code |
|---|---|
| (a) ResUNet segmentation of pupil, iris and open eye; ellipse fits Ê<sub>pup</sub>, E<sub>iris</sub> | `VOGeoGaze.resunet`, `models/ellipse_fit.py` |
| (b) geometric tokens **Z** = [**Z**<sub>spt</sub> \| **Z**<sub>tmp</sub>]; prior gaze direction F<sub>dir</sub> signs the pupil minor axis (Eq. 1) | `VOGeoGaze.z_spt`, `z_tmp`, `f_dir` |
| (c) refraction-aware ellipse correction F<sub>ref</sub> (Eq. 2) | `models/refraction.py` |
| (e) geometry-constrained eye reconstruction Π<sub>f,r</sub> (Eq. 3) | `models/reconstruction.py`, `geometry/unprojection.py` |
| (f) temporal anatomical stabilisation F<sub>tem</sub> (Eq. 4) | `models/temporal.py` |
| anatomical priors 𝒜 (Table 1) | `geometry/anatomy.py` |

The whole network is `vogeo_gaze/models/vogeo_gaze.py`.  The weights (8 MB) ship with the code in
`vogeo_gaze/weights/`.

## Installation

Python 3.10 or newer.  Tested with Python 3.11, PyTorch 2.5 and CUDA 12.1.

```bash
git clone https://github.com/DSGZ-MotionLab/VOGeo-Gaze.git
cd VOGeo-Gaze
pip install .          # or: pip install -r requirements.txt
```

A CUDA GPU is recommended.  The CPU also works, at about 25 frames/s.

## Usage

```bash
python -m vogeo_gaze data/video.mp4               # -> data/vogeo_results/video/
python -m vogeo_gaze a.mp4 b.avi --out-dir out    # -> out/a/, out/b/
```

After `pip install .`, `vogeo-gaze` is the same command.  Progress bars show frames/s for the
network pass and the overlay.  Each video produces the following in `<out-dir>/<video name>/`;
by default `<out-dir>` is `vogeo_results/` next to the video.

| File | Content |
|---|---|
| `gaze.csv` | per-frame gaze, eye parameters, ellipses and confidences |
| `overlay.mp4` | input beside the fitted eye (legend below) |
| `run_info.json` | settings, eyeball centre and throughput |
| `segmentation.mp4` | segmentation and detected ellipses (with `--seg-video`) |

Overlay legend: white, detected entrance pupil Ê<sub>pup</sub>; yellow, its rim refracted through the
fitted cornea onto the pupil plane; cyan, limbus of the fitted eye; pink and blue, eyeball and cornea
wireframes; red dot, eyeball centre; green, pupil centre and gaze.  The fitted-eye elements are dimmed
on frames where `gaze_ok` is 0.

F<sub>tem</sub> estimates one eyeball centre per clip of 16 frames (Eq. 4).  Assuming the eye does
not move relative to the camera during a recording, the median of these clip centres is used as the
eyeball centre of the whole recording.  Every frame's refraction-corrected pupil ray is then lifted
onto the eyeball sphere about that centre, which gives its gaze.

### Options

| Option | Meaning |
|---|---|
| `--source-fpx F` | native focal length of the camera in pixels, if known (see below) |
| `--start N`, `--frames N` | process a frame range |
| `--batch N` | videos per forward pass |
| `--no-video`, `--no-mesh`, `--overlay-only` | overlay output |
| `--seg-video` | also write `segmentation.mp4` |
| `--weights`, `--device`, `--autocast` | other weights, `cpu`, fp16 on CUDA |

### Python

```python
from vogeo_gaze import Engine, Options, load_model

Engine(load_model(), Options()).run([("video.mp4", "video_vogeo")])
```

`load_model()` returns the `torch.nn.Module`.  Its input is a clip of shape (B, T, 3, 240, 320) with
values in [0, 1], and it returns per-frame tensors of shape (B, T, …).

## Output columns (`gaze.csv`)

3-D quantities are in millimetres in the camera frame: x to the image right, y image down, z along
the optical axis.  Pixel quantities refer to the 320 × 240 model canvas.

| Column | Meaning |
|---|---|
| `frame`, `timestamp` | source frame index; seconds |
| `theta_h`, `theta_v` | gaze angles θ<sub>H</sub> = atan2(g<sub>x</sub>, −g<sub>z</sub>), θ<sub>V</sub> = asin(g<sub>y</sub>), in degrees |
| `gaze_x/y/z` | unit gaze vector (optic axis **n**) |
| `c_eye_x/y/z` | eyeball centre **c**<sub>eye</sub> |
| `c_pup_x/y/z`, `r_pup` | pupil centre **c**<sub>pup</sub> and radius r<sub>pup</sub> |
| `pupil_conf`, `iris_conf` | ellipse-fit confidence in [0, 1] |
| `open_eye_score`, `blink_index` | share of the iris inside the open eye; running blink id (0 = open, blink when the score < 0.7) |
| `entpup_theta/cx/cy/a/b` | detected entrance-pupil ellipse Ê<sub>pup</sub> (rad, px) |
| `iris_theta/cx/cy/a/b` | detected iris ellipse E<sub>iris</sub> (rad, px) |
| `gaze_ok` | 0 where the pupil ray missed the eyeball sphere and the clip's gaze was kept |

## Camera and frame rate

The model uses a pinhole camera with a focal length of 1066.7 px at the 320 × 240 canvas.  By default
each video is scaled to fill the canvas, which implies a native focal length of 1066.7 px divided by
that scale; the run prints this value.  If the native focal length is known, `--source-fpx` resamples
the video to the model's focal length instead.  Gaze angles are robust to this assumption; absolute
depths in millimetres are not.

F<sub>tem</sub> was trained at about 12 Hz.  Faster videos are therefore read in clips of 16 × *s*
frames, with *s* = round(fps / 12), and every *s*-th frame feeds F<sub>tem</sub>.  Segmentation,
ellipse fits and reconstruction still run on every frame.

## Speed

Single RTX 4090, one video at a time, including video decoding:

| Video | network pass | overlay drawing (CPU) |
|---|---|---|
| 640 × 480, 95 Hz | 590 frames/s | 720 frames/s |
| 188 × 120, 220 Hz | 800 frames/s | 750 frames/s |

## Citation

```bibtex
@InProceedings{ZhaJin_VOGeoGaze_MICCAI2026,
  author    = {Zhao, Jingkang and Ahmadi, Seyed-Ahmad and Decker, Julian and zu Eulenburg, Peter
               and Zwergal, Andreas and Flanagin, Virginia L. and Wuehr, Max},
  title     = {{VOGeo-Gaze: Calibration-Free, Geometry-Aware Deep Learning for Real-Time Gaze
               Tracking in Clinical Video-Oculography}},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  volume    = {LNCS 16896}
}
```

## License

Apache License 2.0; see `LICENSE`.
