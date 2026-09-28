"""Temporal anatomical stabilisation F_tem (paper, Eq. 4).

One eyeball centre is estimated per clip, c_eye = c_eye,0 + alpha tanh(F_tem(Z_tmp, P, P^2D)),
where c_eye,0 is a confidence-weighted prior.  Every frame's corrected pupil ray is
then lifted onto the sphere of radius L_p about that centre, which gives its gaze.

F_tem was trained on 16-frame clips at about 12 Hz.  Faster videos are read in
clips of 16 x stride frames: every stride-th frame is refined, and the clip centre
is applied to all frames.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vogeo_gaze.geometry.anatomy import R_EYE, R_IRIS
from vogeo_gaze.geometry.camera import pixel_to_ray, rays_plane_intersect
from vogeo_gaze.geometry.ellipse import el2_to_el, ellipse_points

GEOM_DIMS = {"c_eye": 3, "gaze": 3, "c_pupil": 3, "r_pupil": 1, "pupil_el2": 6,
             "entpup_el2": 6, "iris_el2": 6, "pupil_conf": 1, "iris_conf": 1}


class TemporalRefiner(nn.Module):
    """Single-layer Transformer encoder (128-D, 4 heads, FF 256) and a bounded MLP head."""

    def __init__(self, img_size=(240, 320), fpx=1066.6666666666667, alpha_mm=3.0):
        super().__init__()
        self.img_size, self.fpx, self.alpha_mm = tuple(img_size), float(fpx), float(alpha_mm)
        # Z_tmp(128) + c_eye(3) + gaze(3) + r_pup(1) + entrance/iris ellipses(12) + confidences(2)
        self.in_proj = nn.Sequential(nn.LayerNorm(149), nn.Linear(149, 128))
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 3))

    @staticmethod
    def _inputs(geom, dtype):
        """(B,T,d) copies of the per-frame geometry with non-finite values zeroed."""
        B, T = geom["c_eye"].shape[:2]
        return {k: torch.nan_to_num(geom[k].reshape(B, T, d).to(dtype), nan=0.0, posinf=0.0, neginf=0.0)
                for k, d in GEOM_DIMS.items()}

    @staticmethod
    def _trust(geom, g):
        """Per-frame weight: fit confidences, finite geometry, non-degenerate ellipses."""
        B, T = geom["c_eye"].shape[:2]
        return ((g["pupil_conf"].clamp(0, 1) * g["iris_conf"].clamp(0, 1)).sqrt()
                * torch.isfinite(geom["c_eye"].reshape(B, T, 3)).all(-1, keepdim=True)
                * torch.isfinite(geom["pupil_el2"].reshape(B, T, 6)).all(-1, keepdim=True)
                * (g["pupil_el2"][..., 2:3] > 1.0) * (g["iris_el2"][..., 2:3] > 1.0))

    def _lift(self, g, c_eye, trust):
        """Each frame's corrected pupil-centre ray onto the sphere (c_eye, L_p)."""
        B, T = trust.shape[:2]
        c_eye = c_eye[:, None, :].expand(B, T, 3).reshape(B * T, 3)
        l_p = (R_EYE ** 2 - R_IRIS ** 2) ** 0.5
        pupil = g["pupil_el2"].reshape(B * T, 6)
        ray = pixel_to_ray(pupil[:, :2], self.fpx, self.img_size)
        b = (ray * c_eye).sum(-1, keepdim=True)
        disc = b.square() - (c_eye.square().sum(-1, keepdim=True) - l_p ** 2)
        hit = (disc > 1e-8) & (b > 0.0) & (trust.reshape(B * T, 1) > 1e-6)
        c_pupil = torch.where(hit, (b - disc.clamp_min(0.0).sqrt()) * ray, g["c_pupil"].reshape(B * T, 3))
        gaze = torch.where(hit, F.normalize(c_pupil - c_eye, dim=-1, eps=1e-8),
                           F.normalize(g["gaze"].reshape(B * T, 3), dim=-1, eps=1e-8))
        # Pupil radius from the corrected rim on the recovered pupil plane.
        rays = pixel_to_ray(ellipse_points(el2_to_el(pupil), 32), self.fpx, self.img_size)
        points, ray_ok = rays_plane_intersect(rays, c_pupil, gaze, eps=0.03)
        r = (points - c_pupil[:, None, :]).norm(dim=-1).mean(dim=1, keepdim=True)
        ok = hit & ray_ok.all(dim=1, keepdim=True) & torch.isfinite(r) & (r >= 0.5) & (r <= 5.0)
        r_pupil = torch.where(ok, r, g["r_pupil"].reshape(B * T, 1).clamp(0.5, 5.0))
        return c_eye, c_pupil, gaze, r_pupil

    def _refine(self, z_tmp, geom):
        B, T, _ = z_tmp.shape
        g = self._inputs(geom, z_tmp.dtype)
        center = g["c_eye"]
        finite = torch.isfinite(geom["c_eye"].reshape(B, T, 3)).all(-1, keepdim=True)
        trust = self._trust(geom, g)
        # Down-weight 3-D outliers among confident frames (robust prior).
        with torch.no_grad():
            median = torch.nanmedian(torch.where(trust > 0, center, torch.full_like(center, float("nan"))),
                                     dim=1).values
            residual = (center - torch.nan_to_num(median, nan=0.0)[:, None, :]).norm(dim=-1, keepdim=True)
            trust = trust / (1.0 + (residual / 5.0).square())
        weight = trust.sum(dim=1).clamp_min(1e-6)
        support = trust.sum(dim=1) > 1e-6
        c_eye0 = torch.where(support, (trust * center).sum(dim=1) / weight,
                             (center * finite).sum(dim=1) / finite.to(z_tmp.dtype).sum(dim=1).clamp_min(1.0))
        H, W = self.img_size
        scale = z_tmp.new_tensor([W, H, W, H, 1.0, 1.0])
        feats = torch.cat((z_tmp, center / 100.0, g["gaze"], g["r_pupil"] / 5.0, g["entpup_el2"] / scale,
                           g["iris_el2"] / scale, g["pupil_conf"].clamp(0.0, 1.0),
                           g["iris_conf"].clamp(0.0, 1.0)), dim=-1)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        pad = (trust.squeeze(-1) < 1e-4) & support[:, None, 0]
        encoded = self.encoder(self.in_proj(feats), src_key_padding_mask=pad)
        pooled = (encoded * trust).sum(dim=1) / weight
        c_eye = c_eye0 + self.alpha_mm * torch.tanh(self.head(pooled)) * support.to(z_tmp.dtype)
        return c_eye, self._lift(g, c_eye, trust)

    def forward(self, z_tmp, geom, stride=1):
        """z_tmp (B,T,128), geom: (B,T,d) tensors named as in GEOM_DIMS.

        Returns the clip centre ``c_eye_clip`` (B,3) and per-frame c_eye, c_pupil,
        gaze, r_pupil (B*T, d).
        """
        B, T, _ = z_tmp.shape
        if stride <= 1:
            c_eye, lifted = self._refine(z_tmp, geom)
        else:
            idx = torch.arange(0, T, stride, device=z_tmp.device)
            c_eye, lifted_k = self._refine(z_tmp.index_select(1, idx),
                                           {k: v.index_select(1, idx) for k, v in geom.items()})
            # Every frame lifted with the clip centre; refined frames keep the refiner's values.
            g = self._inputs(geom, c_eye.dtype)
            trust = self._trust(geom, g)
            lifted = []
            for full, part in zip(self._lift(g, c_eye, trust), lifted_k):
                full = full.reshape(B, T, -1).clone()
                full.index_copy_(1, idx, part.reshape(B, idx.numel(), -1))
                lifted.append(full.reshape(B * T, -1))
        c_eye_bt, c_pupil, gaze, r_pupil = lifted
        return {"c_eye_clip": c_eye, "c_eye": c_eye_bt, "c_pupil": c_pupil, "gaze": gaze, "r_pupil": r_pupil}
