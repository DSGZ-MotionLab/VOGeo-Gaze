"""Projected eye model for the overlay: eyeball and cornea wireframes, limbus, centres."""
import math

import torch

from vogeo_gaze.geometry.anatomy import R_CORNEA, R_EYE, R_IRIS
from vogeo_gaze.geometry.camera import circle_to_ellipse, project


def axis_angle_to_matrix(aa):
    """(N,3) axis-angle -> (N,3,3); Rodrigues with a first-order expansion near zero,
    following kornia.geometry.conversions.axis_angle_to_rotation_matrix (Apache-2.0)."""
    col = aa.unsqueeze(1)
    theta2 = torch.matmul(col, col.transpose(1, 2)).squeeze(1)
    theta = torch.sqrt(theta2)
    wx, wy, wz = torch.chunk(aa / (theta + 1e-6), 3, dim=1)
    c, s = torch.cos(theta), torch.sin(theta)
    normal = torch.cat([c + wx * wx * (1.0 - c), wx * wy * (1.0 - c) - wz * s, wy * s + wx * wz * (1.0 - c),
                        wz * s + wx * wy * (1.0 - c), c + wy * wy * (1.0 - c), -wx * s + wy * wz * (1.0 - c),
                        -wy * s + wx * wz * (1.0 - c), wx * s + wy * wz * (1.0 - c), c + wz * wz * (1.0 - c)],
                       dim=1).view(-1, 3, 3)
    rx, ry, rz = torch.chunk(aa, 3, dim=1)
    one = torch.ones_like(rx)
    taylor = torch.cat([one, -rz, ry, rz, one, -rx, -ry, rx, one], dim=1).view(-1, 3, 3)
    mask = (theta2 > 1e-6).view(-1, 1, 1)
    return mask.type_as(theta2) * normal + (~mask).type_as(theta2) * taylor


def rotation_from_z(g, eps=1e-8):
    """(N,3,3) rotations taking +z to the direction of ``g``."""
    B = g.shape[0]
    g = g / g.norm(dim=-1, keepdim=True).clamp_min(eps)
    z = torch.tensor([0.0, 0.0, 1.0], device=g.device, dtype=g.dtype).view(1, 3).expand(B, 3)
    angle = torch.acos((g * z).sum(dim=-1).clamp(-1.0, 1.0))
    axis = torch.cross(g, z, dim=-1)
    R = axis_angle_to_matrix(axis / axis.norm(dim=-1, keepdim=True).clamp_min(eps) * angle.unsqueeze(-1))
    I = torch.eye(3, device=g.device, dtype=g.dtype).unsqueeze(0).expand(B, 3, 3)
    flip = I.clone()
    flip[:, 1, 1] = flip[:, 2, 2] = -1
    R = torch.where((angle < 1e-7).view(B, 1, 1), I, R)
    R = torch.where((torch.abs(angle - torch.pi) < 1e-7).view(B, 1, 1), flip, R)
    return R.transpose(-1, -2)


def sphere_mesh(r, extra_theta, n=25):
    """(N,1) radii -> (N, n+1, n, 3) polar grid about +z, with one extra latitude."""
    theta = torch.cat([torch.linspace(0.0, math.pi, n, device=r.device).view(1, n).expand(r.shape[0], n),
                       extra_theta], dim=-1).sort(dim=-1)[0].unsqueeze(-1)
    phi = torch.linspace(0.0, 2.0 * math.pi, n, device=r.device).view(1, 1, n)
    r = r.view(-1, 1, 1)
    X = r * torch.sin(theta) * torch.cos(phi)
    Y = r * torch.sin(theta) * torch.sin(phi)
    return torch.stack([X, Y, (r * torch.cos(theta)).expand_as(X)], dim=-1)


def render_eye(c_eye, gaze, fpx, img_size, mesh=True, n=25):
    """Project the fitted eye (LeGrand anatomy) for drawing.

    c_eye, gaze: (N,3).  Returns pixel positions of the eyeball and pupil centres,
    the limbus ellipse ``iris_el`` (N,5), the corneal sphere (for refraction) and,
    with ``mesh``, the eyeball and cornea wireframes (N, n+1, n, 2).
    """
    full = lambda v: torch.full((c_eye.shape[0], 1), v)
    r_eye, r_cornea, r_iris = full(R_EYE), full(R_CORNEA), full(R_IRIS)
    L_p = torch.sqrt((r_eye ** 2 - r_iris ** 2).clamp_min_(1e-6))
    L_ec = L_p - torch.sqrt((r_cornea ** 2 - r_iris ** 2).clamp_min_(1e-6))
    c_cornea = c_eye + L_ec * gaze
    c_pupil = c_eye + L_p * gaze
    iris_el, ok = circle_to_ellipse(c_pupil, gaze, r_iris, fpx, img_size)
    if not ok.all():            # near edge-on: small finite axes
        iris_el = iris_el.clone()
        iris_el[~ok, 3:5] = iris_el.new_tensor([1.0])
    out = {"c_eye2d": project(c_eye, fpx, img_size), "c_pupil2d": project(c_pupil, fpx, img_size),
           "iris_el": iris_el, "c_cornea": c_cornea, "r_cornea": r_cornea, "c_pupil": c_pupil}
    if mesh:
        R = rotation_from_z(gaze)
        B = c_eye.shape[0]
        place = lambda m, c: torch.bmm(m.reshape(B, -1, 3), R.transpose(1, 2)).reshape(m.shape) + c[:, None, None, :]
        eye = place(sphere_mesh(r_eye, torch.arccos(L_p / r_eye), n), c_eye)
        corn = place(sphere_mesh(r_cornea, torch.arccos((L_p - L_ec) / r_cornea), n), c_cornea)
        # Keep only the outer shell where the two spheres intersect.
        r2_eye, r2_corn = (r_eye.view(-1) ** 2)[:, None, None], (r_cornea.view(-1) ** 2)[:, None, None]
        d_eye = ((eye - c_cornea[:, None, None, :]) ** 2).sum(-1)
        d_corn = ((corn - c_eye[:, None, None, :]) ** 2).sum(-1)
        eye = torch.where((d_eye < (r2_corn - (r2_corn * 2e-3).clamp_min(1e-4)))[..., None], torch.nan, eye)
        corn = torch.where((d_corn >= (r2_eye - (r2_eye * 2e-3).clamp_min(1e-4)))[..., None], corn, torch.nan)
        out["eyeball_mesh"] = project(eye, fpx, img_size, eps=0)
        out["cornea_mesh"] = project(corn, fpx, img_size, eps=0)
    return out
