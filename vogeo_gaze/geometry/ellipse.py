"""Ellipse forms and conversions.

Two parameterisations are used, both in canvas pixels:

  el   (..., 5)  [theta, cx, cy, a, b]              theta in rad, a >= b semi-axes
  el2  (..., 6)  [cx, cy, a, b, cos 2theta, sin 2theta]   continuous in theta
"""
import math

import torch
import torch.nn.functional as F


def el2_theta(el2, eps=1e-8):
    c2, s2 = el2[..., 4], el2[..., 5]
    theta = 0.5 * torch.atan2(s2, c2)
    return torch.where((c2.abs() < eps) & (s2.abs() < eps), torch.zeros_like(theta), theta)


def el2_to_el(el2):
    u, v, a, b = el2.unbind(-1)[:4]
    return torch.stack([el2_theta(el2), u, v, a, b], dim=-1)


def canon_el2(el2, eps=1e-8):
    """a >= b >= 1, unit (cos 2t, sin 2t), and zero angle for near-circles."""
    u, v, a, b, c2, s2 = el2.unbind(dim=-1)
    a, b = a.clamp_min(1.0), b.clamp_min(1.0)
    ang = F.normalize(torch.stack([c2, s2], dim=-1), dim=-1, eps=eps)
    c2, s2 = ang[:, 0], ang[:, 1]
    swap = b > a
    a, b = torch.where(swap, b, a), torch.where(swap, a, b)
    c2, s2 = torch.where(swap, -c2, c2), torch.where(swap, -s2, s2)
    circ = (b / a) > 0.97
    c2 = torch.where(circ, torch.ones_like(c2), c2)
    s2 = torch.where(circ, torch.zeros_like(s2), s2)
    return torch.stack([u, v, a, b, c2, s2], dim=-1)


def conic_to_el2(conic, eps=1e-8):
    """(N,6) [A,B,C,D,E,F] of Ax^2+Bxy+Cy^2+Dx+Ey+F=0 -> (canonical el2, valid)."""
    in_dtype = conic.dtype
    if conic.dtype in (torch.float16, torch.bfloat16):
        conic = conic.float()
    conic = torch.where(((conic[:, 0] + conic[:, 2]) < 0).unsqueeze(-1), -conic, conic)
    A, B, C, D, E, F0 = conic.unbind(1)
    M = torch.stack([torch.stack([2 * A, B], -1), torch.stack([B, 2 * C], -1)], -2)
    M = M + eps * torch.eye(2, device=conic.device, dtype=conic.dtype).unsqueeze(0)
    uv = torch.linalg.solve(M, torch.stack([-D, -E], -1).unsqueeze(-1)).squeeze(-1)
    cx, cy = uv[:, 0], uv[:, 1]
    Fc = F0 + A * cx * cx + B * cx * cy + C * cy * cy + D * cx + E * cy
    # Closed-form 2x2 eigenvalues of the quadratic part.
    disc = torch.sqrt(((A - C) * (A - C) + B * B).clamp_min(0.0)) / 2.0
    l1, l2 = (A + C) / 2.0 - disc, (A + C) / 2.0 + disc
    valid = (l1 > eps) & (l2 > eps) & (Fc < -eps)
    a1 = torch.sqrt((-Fc / l1).clamp_min(eps))
    b1 = torch.sqrt((-Fc / l2).clamp_min(eps))
    theta = torch.atan2(l1 - A, B / 2.0 + eps)
    theta = torch.where(b1 > a1, theta + math.pi / 2.0, theta)
    el = torch.stack([theta, cx, cy, torch.maximum(a1, b1), torch.minimum(a1, b1)], -1).to(in_dtype)
    theta, u, v, a, b = el.unbind(dim=-1)
    el2 = torch.stack([u, v, a, b, torch.cos(2.0 * theta), torch.sin(2.0 * theta)], dim=-1)
    return canon_el2(el2, eps=eps), valid


def ellipse_points(el, n):
    """(N,5) el -> (N,n,2) rim points, first and last coinciding."""
    theta, cx, cy, a, b = (el[:, i:i + 1] for i in range(5))
    t = torch.linspace(0.0, 2.0 * torch.pi, steps=n, device=el.device, dtype=el.dtype)
    dx, dy = a * t.cos(), b * t.sin()
    ct, st = theta.cos(), theta.sin()
    return torch.stack((cx + dx * ct - dy * st, cy + dx * st + dy * ct), dim=-1)


def minor_axis(theta):
    """Unit minor-axis direction g(theta) of an ellipse with major-axis angle theta."""
    return F.normalize(torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1), dim=-1)
