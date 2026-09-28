"""Pinhole camera: rays, projection, plane/sphere intersections, corneal refraction.

Camera frame in mm: x image right, y image down, z along the optical axis; the
principal point is the canvas centre and ``fpx`` the focal length in pixels.
"""
import torch
import torch.nn.functional as F

from vogeo_gaze.geometry.anatomy import N_AIR, N_CORNEA


def pixel_to_ray(uv, fpx, img_size):
    """(..., 2) pixels -> (..., 3) unit rays."""
    height, width = img_size
    x = uv[..., 0] - (width - 1) * 0.5
    y = uv[..., 1] - (height - 1) * 0.5
    return F.normalize(torch.stack([x, y, torch.ones_like(x) * fpx], dim=-1), dim=-1)


def project(p, fpx, img_size, eps=1e-6):
    """(..., 3) points -> (..., 2) pixels."""
    cx, cy = (img_size[1] - 1) / 2, (img_size[0] - 1) / 2
    return torch.stack([p[..., 0] / p[..., 2].clamp_min(eps) * fpx + cx,
                        p[..., 1] / p[..., 2].clamp_min(eps) * fpx + cy], dim=-1)


def circle_to_ellipse(center, normal, radius, fpx, img_size, eps=1e-12):
    """Exact image of a 3-D circle: (N,5) el [theta, cx, cy, a, b] and a validity mask."""
    B, dev, out_dtype = center.shape[0], center.device, center.dtype
    f64 = lambda t: t.to(device=dev, dtype=torch.float64)
    c_p, n = f64(center), F.normalize(f64(normal), dim=-1)
    f = torch.tensor(float(fpx), device=dev, dtype=torch.float64)
    r = f64(radius.reshape(-1))
    eps = torch.tensor(eps, dtype=torch.float64, device=dev)
    K = torch.tensor([[f, 0.0, (img_size[1] - 1) * 0.5], [0.0, f, (img_size[0] - 1) * 0.5],
                      [0.0, 0.0, 1.0]], device=dev, dtype=torch.float64).expand(B, -1, -1)
    # Plane basis (u, v) and the homography (s, t, 1) -> image, H = K [u v c].
    ax = torch.tensor([1., 0., 0.], device=dev, dtype=torch.float64).expand_as(n)
    ay = torch.tensor([0., 1., 0.], device=dev, dtype=torch.float64).expand_as(n)
    u = F.normalize(torch.cross(n, torch.where((n[..., 0].abs() >= 0.9)[..., None], ay, ax), dim=-1), dim=-1)
    v = F.normalize(torch.cross(n, u, dim=-1), dim=-1)
    H = torch.einsum("bij,bjk->bik", K, torch.stack([u, v, c_p], dim=-1))
    good_h = torch.linalg.det(H).abs() > eps
    # Circle conic diag(1, 1, -r^2) mapped to the image: H^-T C H^-1.
    C = torch.zeros(B, 3, 3, device=dev, dtype=torch.float64)
    C[:, 0, 0], C[:, 1, 1], C[:, 2, 2] = 1.0, 1.0, -(r * r).clamp_min(eps)
    H_inv = torch.linalg.pinv(H)
    Ci = torch.einsum("bij,bjk,bkl->bil", H_inv.transpose(1, 2), C, H_inv)
    Ci = 0.5 * (Ci + Ci.transpose(1, 2))
    Ci = Ci / torch.stack([Ci[:, 0, 0], 2 * Ci[:, 0, 1], Ci[:, 1, 1]], dim=-1).norm(dim=-1).clamp_min(eps)[:, None, None]
    A, Bq, Cq, D, E, Fv = Ci[:, 0, 0], Ci[:, 0, 1], Ci[:, 1, 1], Ci[:, 0, 2], Ci[:, 1, 2], Ci[:, 2, 2]
    Q2 = torch.stack([torch.stack([2 * A, 2 * Bq], dim=-1), torch.stack([2 * Bq, 2 * Cq], dim=-1)], dim=1)
    cxy = torch.linalg.pinv(Q2).bmm(torch.stack([-2 * D, -2 * E], dim=-1)[..., None]).squeeze(-1)
    cx, cy = cxy[:, 0], cxy[:, 1]
    F0 = A * cx * cx + 2 * Bq * cx * cy + Cq * cy * cy + 2 * D * cx + 2 * E * cy + Fv
    evals, evecs = torch.linalg.eigh(Q2 / 2)
    a = torch.sqrt((F0.abs() / evals[:, 0].abs().clamp_min(eps)).clamp_min(eps))
    b = torch.sqrt((F0.abs() / evals[:, 1].abs().clamp_min(eps)).clamp_min(eps))
    swap = b > a
    a, b = torch.where(swap, b, a), torch.where(swap, a, b)
    major = torch.where(swap.unsqueeze(-1), evecs[:, :, 1], evecs[:, :, 0])
    theta = ((torch.atan2(major[:, 1], major[:, 0]) + torch.pi / 2) % torch.pi) - torch.pi / 2
    ok = (torch.isfinite(a) & torch.isfinite(b) & torch.isfinite(cx) & torch.isfinite(cy)
          & (torch.sign(evals[:, 0]) == torch.sign(evals[:, 1])) & (F0.abs() > eps) & good_h)
    return torch.stack([theta, cx, cy, a, b], dim=-1).to(out_dtype), ok


def rays_plane_intersect(ray, c_plane, n_plane, eps=1e-8):
    """Rays from the origin (B,N,3) hit the plane (c_plane, n_plane): points (B,N,3), valid (B,N)."""
    n = n_plane / (n_plane.norm(dim=-1, keepdim=True) + eps)
    n_dot_c = (n * c_plane).sum(dim=-1, keepdim=True)
    n_dot_q = (n.unsqueeze(1) * ray).sum(dim=-1, keepdim=True)
    valid = n_dot_q.abs().squeeze(-1) > eps
    sign = torch.where(n_dot_q >= 0, torch.ones_like(n_dot_q), -torch.ones_like(n_dot_q))
    t = n_dot_c.unsqueeze(1) / torch.where(n_dot_q.abs() < eps, sign * eps, n_dot_q)
    return torch.where(valid.unsqueeze(-1), t * ray, c_plane.unsqueeze(1).expand_as(ray)), valid


def line_plane_intersect(p_line, d_line, p_plane, n_plane, eps=1e-8):
    """Line p + t d (B,3) meets the plane (p_plane, n_plane): point (B,3)."""
    num = (n_plane * (p_plane - p_line)).sum(dim=-1, keepdim=True)
    den = (n_plane * d_line).sum(dim=-1, keepdim=True)
    valid = den.abs().squeeze(-1) > eps
    sign = torch.where(den >= 0, torch.ones_like(den), -torch.ones_like(den))
    x = p_line + num / torch.where(den.abs() < eps, sign * eps, den) * d_line
    return torch.where(valid.unsqueeze(-1), x, p_plane)


def refract_to_pupil_plane(c_cornea, r_cornea, gaze, c_pupil, rays, min_cos=0.03, eps=1e-6):
    """Camera rays (B,N,3) refracted at the corneal sphere onto the pupil plane: (B,N,3).

    Snell's law at the sphere (c_cornea, r_cornea); the pupil plane passes through
    c_pupil with normal ``gaze``.  Rays that miss the cornea continue unrefracted.
    """
    B, N, _ = rays.shape
    # Nearer intersection of each ray with the corneal sphere.
    l = rays / torch.norm(rays, dim=-1, keepdim=True).clamp_min(eps)
    oc = (0.0 - c_cornea.view(B, 1, 3)).expand(B, N, 3)
    b = torch.sum(l * oc, dim=2)
    r = r_cornea.view(B, 1)
    delta = b * b - torch.sum(oc * oc, dim=2) + (r * r).expand(B, N)
    miss = (delta < 0) | torch.isnan(delta)
    root = torch.where(miss, torch.zeros_like(delta), torch.sqrt(delta.clamp_min(0.0) + eps))
    d = torch.min(-b + root, -b - root).masked_fill(miss, 0.0)
    hit = d.unsqueeze(-1) * rays
    normal = F.normalize(hit - c_cornea.view(B, 1, 3), dim=-1)
    # Snell's law.
    ratio = N_CORNEA / N_AIR
    cos_air = (rays * normal).sum(dim=-1)
    k = cos_air + torch.sqrt((ratio ** 2 - 1 + cos_air ** 2).clamp(min=0.0))
    refr = ratio * (rays - k.unsqueeze(-1) * normal)
    refr = refr / torch.norm(refr, dim=-1, keepdim=True)
    refr = torch.where(miss.unsqueeze(-1), rays, refr)
    # Onto the pupil plane; near-parallel paths keep their sign but a bounded slope.
    g = gaze.view(B, 1, 3)
    num = ((c_pupil.view(B, 1, 3) - hit) * g).sum(dim=-1)
    den = (refr * g).sum(dim=-1)
    den = den.sign().masked_fill(den == 0, 1.0) * den.abs().clamp_min(min_cos)
    return hit + (num / den).unsqueeze(-1) * refr
