"""VOGeo-Gaze (paper, Fig. 2).

  (a) ResUNet segmentation of pupil, iris and open eye; ellipse fits E^_pup, E_iris
  (b) geometric tokens Z = [Z_spt | Z_tmp]; F_dir signs the pupil minor axis -> g
  (c) F_ref corrects the entrance pupil for corneal refraction -> E_pup
  (e) closed-form reconstruction of c_eye, n, c_pup, r_pup under the anatomical priors
  (f) F_tem estimates one eyeball centre per clip; each frame's gaze is its pupil
      ray lifted onto the eyeball sphere about that centre
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SegResNet

from vogeo_gaze.geometry.ellipse import el2_theta, el2_to_el, minor_axis
from vogeo_gaze.models.ellipse_fit import EllipseFit
from vogeo_gaze.models.reconstruction import reconstruct_eye
from vogeo_gaze.models.refraction import RefractionCorrector
from vogeo_gaze.models.temporal import TemporalRefiner


def descriptor(dim_in=256, hidden=256, dim_out=128):
    """Pooled bottleneck [mean | max] -> one 128-D half of the token Z."""
    return nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, hidden), nn.GELU(),
                         nn.Linear(hidden, dim_out), nn.LayerNorm(dim_out))


def gaze_angles(gaze):
    """Unit gaze -> (theta_H, theta_V) in degrees."""
    gaze = F.normalize(gaze, dim=-1)
    pitch = torch.arcsin(gaze[..., 1].clamp(-1, 1))
    yaw = torch.arctan2(gaze[..., 0], -gaze[..., 2])
    return torch.stack([torch.rad2deg(yaw), torch.rad2deg(pitch)], dim=-1)


class VOGeoGaze(nn.Module):
    """Input: clips (B, T, 3, H, W) in [0, 1].  Output: per-frame (B, T, ...) tensors."""

    def __init__(self, img_size=(240, 320), fpx=1066.6666666666667):
        super().__init__()
        self.img_size, self.fpx = tuple(img_size), float(fpx)
        # 4-stage ResUNet, channels 16 -> 128.
        self.resunet = SegResNet(spatial_dims=2, init_filters=16, in_channels=3, out_channels=3,
                                 dropout_prob=0.2, norm="batch")
        self.ellipse_fit = EllipseFit(self.img_size)
        self.z_spt, self.z_tmp = descriptor(), descriptor()
        self.f_dir = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2))
        self.f_ref = RefractionCorrector(self.img_size)
        self.f_tem = TemporalRefiner(self.img_size, self.fpx)

    def forward(self, x, stride=1):
        """``stride``: every stride-th frame feeds F_tem (see models/temporal.py)."""
        B, T, _, H, W = x.shape
        assert (H, W) == self.img_size, f"input {H}x{W}, expected {self.img_size}"
        per_frame = lambda v: v.reshape(B, T, *v.shape[1:])

        bottleneck, skips = self.resunet.encode(x.reshape(B * T, *x.shape[2:]))
        skips.reverse()
        logits = self.resunet.decode(bottleneck, skips)
        fit = self.ellipse_fit(torch.sigmoid(logits))
        pooled = torch.cat([bottleneck.mean(dim=(-2, -1)), bottleneck.amax(dim=(-2, -1))], dim=-1)
        z_spt, z_tmp = per_frame(self.z_spt(pooled)), per_frame(self.z_tmp(pooled))
        direction = F.normalize(self.f_dir(z_spt), dim=-1, eps=1e-6).reshape(B * T, 2)

        ent_el2, iris_el2 = fit["entpup_el2"], fit["iris_el2"]
        pupil_conf, iris_conf = fit["pupil_conf"].reshape(-1, 1), fit["iris_conf"].reshape(-1, 1)
        # Prior gaze direction g = sign(g(theta) . F_dir(Z)) g(theta) (Eq. 1).
        axis = minor_axis(el2_theta(ent_el2))
        g = axis * torch.where((axis * direction).sum(dim=-1, keepdim=True) >= 0.0, 1.0, -1.0).to(axis.dtype)
        pup_el2 = self.f_ref(ent_el2, -g, iris_el2[:, 2:3])
        c_eye, gaze, c_pupil, r_pupil = reconstruct_eye(pup_el2, iris_el2, g, self.fpx, self.img_size)

        geom = {"c_eye": c_eye, "gaze": gaze, "c_pupil": c_pupil, "r_pupil": r_pupil,
                "pupil_el2": pup_el2, "entpup_el2": ent_el2, "iris_el2": iris_el2,
                "pupil_conf": pupil_conf, "iris_conf": iris_conf}
        tem = self.f_tem(z_tmp, {k: per_frame(v) for k, v in geom.items()}, stride)
        return {k: per_frame(v) for k, v in {
            "gaze_angles": gaze_angles(tem["gaze"]),
            "gaze": tem["gaze"], "c_eye": tem["c_eye"], "c_pupil": tem["c_pupil"], "r_pupil": tem["r_pupil"],
            "pupil_conf": pupil_conf, "iris_conf": iris_conf, "open_eye_score": fit["open_eye_score"],
            "entpup_el": el2_to_el(ent_el2), "iris_el": el2_to_el(iris_el2), "seg_logits": logits,
        }.items()}


WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "vogeo_gaze_miccai2026.pt"


def load_model(weights=None, device=None):
    """VOGeo-Gaze with the MICCAI 2026 weights, in eval mode, gradients off."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # TF32 matmuls, the precision the model was validated with.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    model = VOGeoGaze()
    model.load_state_dict(torch.load(weights or WEIGHTS, map_location="cpu", weights_only=True))
    return model.to(device).eval().requires_grad_(False)
