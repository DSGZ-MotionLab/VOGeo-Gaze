"""Refraction-aware ellipse correction F_ref (paper, Eq. 2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RefractionCorrector(nn.Module):
    """F_ref: E_pup = E^_pup + F_ref(P^2D), a 3-layer MLP (hidden 128).

    Maps the observed entrance-pupil ellipse E^_pup, the signed prior gaze
    direction and the iris size to bounded corrections of the pupil ellipse's
    centre, axes and orientation.
    """

    def __init__(self, img_size=(240, 320), hidden=128):
        super().__init__()
        self.H, self.W = img_size
        self.register_buffer("center_scale", torch.tensor([self.W * 0.25, self.H * 0.25]), persistent=False)
        self.mlp = nn.Sequential(nn.LayerNorm(11), nn.Linear(11, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, 6)

    def forward(self, ent_el2, g2d, iris_a):
        """ent_el2 (N,6) E^_pup, g2d (N,2) signed direction, iris_a (N,1) iris semi-major axis."""
        H, W = self.H, self.W
        u0, v0, a0, b0, c20, s20 = ent_el2.unbind(-1)
        feats = torch.stack([u0 / (W - 1), v0 / (H - 1), a0 / (W - 1), b0 / (H - 1), c20, s20,
                             g2d[:, 0], g2d[:, 1], a0 / b0.clamp_min(1e-6),
                             (a0 * b0).clamp_min(1e-6).log(), iris_a[:, 0] / (W - 1)], dim=-1)
        d = self.head(self.mlp(feats))
        duv = torch.tanh(d[..., 0:2]) * self.center_scale
        u = (u0 + duv[..., 0]).clamp(0, W - 1)
        v = (v0 + duv[..., 1]).clamp(0, H - 1)
        a = (a0 * (1.0 + 0.5 * torch.tanh(d[..., 2]))).clamp_min(1e-3)
        b = (b0 * (1.0 + 0.5 * torch.tanh(d[..., 3]))).clamp_min(1e-3)
        d2 = 2.0 * (torch.tanh(d[..., 4]) * 0.25)          # orientation change, |dtheta| <= 0.25 rad
        cosd, sind = torch.cos(d2), torch.sin(d2)
        c2s2 = F.normalize(torch.stack([c20 * cosd - s20 * sind, c20 * sind + s20 * cosd], -1), dim=-1)
        return torch.stack([u, v, a, b, c2s2[..., 0], c2s2[..., 1]], dim=-1)
