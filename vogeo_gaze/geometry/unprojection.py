"""Closed-form unprojection of an image ellipse to a 3-D circle (Safaee-Rad et al. 1992).

``unproject_ellipse`` is the operator Pi_{f,r}(E, g) of the paper: an ellipse seen by
a pinhole camera of focal length f is the image of two circles of radius r; the
prior 2-D gaze direction g picks one.  All algebra runs in float64.
"""
import math

import torch
import torch.nn.functional as F

_F64 = torch.float64


def _conic(xc, yc, a, b, theta):
    """Ellipse -> A x^2 + B xy + C y^2 + D x + E y + F = 0."""
    a2, b2 = a ** 2, b ** 2
    ct, st = torch.cos(theta), torch.sin(theta)
    A = (ct ** 2) / a2 + (st ** 2) / b2
    B = 2 * ct * st * (1 / a2 - 1 / b2)
    C = (st ** 2) / a2 + (ct ** 2) / b2
    D = -2 * A * xc - B * yc
    E = -B * xc - 2 * C * yc
    F0 = A * xc ** 2 + B * xc * yc + C * yc ** 2 - 1
    return A, B, C, D, E, F0


def _cubic_roots(coef):
    """Real parts of the roots of monic-normalised cubics, sorted descending."""
    a, b, c, d = coef.unbind(1)
    a = torch.where(a.abs() > 1e-7, a, torch.ones_like(a))
    b, c, d = b / a, c / a, d / a
    finite = torch.isfinite(b) & torch.isfinite(c) & torch.isfinite(d)
    b, c, d = (torch.where(finite, v, torch.zeros_like(v)) for v in (b, c, d))
    comp = torch.zeros((coef.shape[0], 3, 3), device=coef.device, dtype=coef.dtype)
    comp[:, 0, 1] = 1
    comp[:, 1, 2] = 1
    comp[:, 2, 0], comp[:, 2, 1], comp[:, 2, 2] = -d, -c, -b
    roots = torch.linalg.eigvals(comp.clamp(-1e12, 1e12)).real.clone()
    roots[~finite] = float("nan")
    return torch.sort(roots, descending=True, dim=1)[0]


def _eigvec(lam, a, b, g, f, h, eps=1e-12):
    """Unit eigenvector (l, m, n) of the cone matrix for eigenvalue ``lam``."""
    t1 = (b - lam) * g - f * h
    t2 = (a - lam) * f - g * h
    small_t2, small_g = t2.abs() < eps, g.abs() < eps
    t2 = torch.where(small_t2, torch.full_like(t2, eps), t2)
    g_safe = torch.where(small_g, torch.full_like(g, eps), g)
    ratio = t1 / t2
    t3 = (-(a - lam) * ratio / g_safe - (h / g_safe)).clamp(-1e6, 1e6)
    m = 1.0 / torch.sqrt(1.0 + ratio ** 2 + t3 ** 2)
    l, n = ratio * m, t3 * m
    bad = small_t2 & small_g
    if bad.any():
        l[bad], m[bad], n[bad] = 1.0 / math.sqrt(2), 1.0 / math.sqrt(2), 0.0
    return l, m, n


def _normals_canonical(l1, l2, l3):
    """The two circle normals (columns 0/1) in the cone's eigenframe."""
    tie = torch.isclose(l1, l2, rtol=1e-7, atol=1e-10)
    out = torch.zeros(l1.shape[0], 3, 2, dtype=l1.dtype, device=l1.device)
    lt = (l1 < l2) & ~tie
    gt = (l1 > l2) & ~tie
    # l1 < l2: m = +-sqrt((l2-l1)/(l2-l3)), n = sqrt((l1-l3)/(l2-l3)); l1 > l2: l and m swap.
    den_lt = (l2 - l3).clamp_min(1e-12)
    den_gt = (l1 - l3).clamp_min(1e-12)
    m_lt = torch.sqrt((l2 - l1).clamp_min(0.0) / den_lt)
    n_lt = torch.sqrt((l1 - l3).clamp_min(0.0) / den_lt)
    l_gt = torch.sqrt((l1 - l2).clamp_min(0.0) / den_gt)
    n_gt = torch.sqrt((l2 - l3).clamp_min(0.0) / den_gt)
    for col, sgn in ((0, 1.0), (1, -1.0)):
        out[:, 0, col] = torch.where(gt, sgn * l_gt, out[:, 0, col])
        out[:, 1, col] = torch.where(lt, sgn * m_lt, out[:, 1, col])
        out[:, 2, col] = torch.where(lt, n_lt, torch.where(gt, n_gt, out[:, 2, col]))
    out[:, 2][tie] = 1.0
    return out


def _T3(l, m, n, eps=1e-12):
    lm = torch.sqrt((l ** 2 + m ** 2).clamp_min(eps))
    T = torch.zeros((l.shape[0], 4, 4), dtype=_F64, device=l.device)
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 2] = -m / lm, -(l * n) / lm, l
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2] = l / lm, -(m * n) / lm, m
    T[:, 2, 1], T[:, 2, 2], T[:, 3, 3] = lm, n, 1.0
    return T


def _center_canonical(T3, lambs, r):
    li, mi, ni = T3[:, 0:3, 0], T3[:, 0:3, 1], T3[:, 0:3, 2]
    A = torch.sum(li ** 2 * lambs, dim=1)
    B = torch.sum(li * ni * lambs, dim=1)
    C = torch.sum(mi * ni * lambs, dim=1)
    D = torch.sum(ni ** 2 * lambs, dim=1)
    Z = (A * r) / torch.sqrt((B ** 2 + C ** 2 - A * D).clamp_min(1e-12))
    return torch.stack([(-B / A) * Z, (-C / A) * Z, Z, torch.ones_like(Z)], dim=1).unsqueeze(-1)


def _unproject_both(conic, fpx, radius):
    """Both circle solutions: (normal+, normal-, centre+, centre-), each (N,3)."""
    A, Bc, C, D, E, F0 = conic
    N, dev = A.shape[0], A.device
    # Cone through the camera centre (alpha, beta, gamma) = (0, 0, -f) and the ellipse.
    alpha, beta, gamma = torch.zeros_like(fpx), torch.zeros_like(fpx), -fpx
    ap, hp, bp, gp, fp, dp = A, Bc / 2, C, D / 2, E / 2, F0
    g2 = gamma * gamma
    a, b, h = g2 * ap, g2 * bp, g2 * hp
    c = (ap * alpha ** 2 + 2.0 * hp * alpha * beta + bp * beta ** 2
         + 2.0 * gp * alpha + 2.0 * fp * beta + dp)
    f = -gamma * (bp * beta + hp * alpha + fp)
    g = -gamma * (hp * beta + ap * alpha + gp)
    u, v = g2 * gp, g2 * fp
    w = -gamma * (fp * beta + gp * alpha + dp)
    coef = torch.stack([torch.ones(N, dtype=_F64, device=dev), -(a + b + c),
                        (b * c + c * a + a * b - f * f - g * g - h * h),
                        -(a * b * c + 2 * f * g * h - a * f * f - b * g * g - c * h * h)], dim=1)
    roots = _cubic_roots(coef)
    lam1, lam2, lam3 = roots[:, 0], roots[:, 1], roots[:, 2]
    lmn = _normals_canonical(lam1, lam2, lam3)                          # (N,3,2)

    T1 = torch.zeros((N, 4, 4), dtype=_F64, device=dev)
    for col, lam in enumerate((lam1, lam2, lam3)):
        T1[:, 0, col], T1[:, 1, col], T1[:, 2, col] = _eigvec(lam, a, b, g, f, h)
    T1[:, 3, 3] = 1.0
    li, mi, ni = T1[:, 0, 0:3], T1[:, 1, 0:3], T1[:, 2, 0:3]            # views into T1
    flip = torch.sum(torch.cross(li, mi, dim=1) * ni, dim=1) < 0
    li[flip], mi[flip], ni[flip] = -li[flip], -mi[flip], -ni[flip]

    T2 = torch.eye(4, dtype=_F64, device=dev).unsqueeze(0).repeat(N, 1, 1)
    lambs = torch.stack([lam1, lam2, lam3], dim=1)
    num = u.unsqueeze(1) * li + v.unsqueeze(1) * mi + w.unsqueeze(1) * ni
    T2[:, 0:3, 3] = -num / torch.where(lambs.abs() < 1e-12, lambs.sign() * 1e-12, lambs)
    T0 = torch.eye(4, dtype=_F64, device=dev).unsqueeze(0).repeat(N, 1, 1)
    T0[:, 2, 3] = -gamma

    one = torch.ones(N, 1, dtype=_F64, device=dev)
    out = []
    for k in (0, 1):
        normal = torch.matmul(T1, torch.cat([lmn[:, :, k], one], 1).unsqueeze(-1))
        T3 = _T3(lmn[:, 0, k], lmn[:, 1, k], lmn[:, 2, k])
        centre = _center_canonical(T3, lambs, radius)
        # The circle lies in front of the camera: flip the canonical centre if needed.
        z = (T0 @ (T1 @ (T2 @ (T3 @ centre))))[:, 2, 0]
        centre = centre.clone()
        centre[:, 0:3, :] *= (1.0 - 2.0 * (z < 0).view(-1, 1, 1).to(_F64))
        centre = T0 @ (T1 @ (T2 @ (T3 @ centre)))
        # The normal points towards the camera.
        normal = normal.clone()
        normal[:, 0:3, :] *= torch.where(normal[:, 2, 0] > 0, -1.0, 1.0).to(_F64).view(-1, 1, 1)
        out.append((normal[:, 0:3, 0], centre[:, 0:3, 0]))
    (n_pos, c_pos), (n_neg, c_neg) = out
    return n_pos, n_neg, c_pos, c_neg


def unproject_ellipse(el_centered, radius, fpx, g2d):
    """Pi_{f,r}(E, g): (N,5) camera-centred el -> (normal (N,3), centre (N,3)).

    ``radius`` (N,1) and ``fpx`` (N,1) share units with the output centre; ``g2d``
    (N,2) is the prior 2-D gaze direction that selects one of the two solutions.
    """
    dtype = el_centered.dtype
    conic = [x.to(_F64) for x in _conic(el_centered[:, 1], el_centered[:, 2], el_centered[:, 3],
                                        el_centered[:, 4], el_centered[:, 0])]
    n_pos, n_neg, c_pos, c_neg = _unproject_both(conic, fpx.to(_F64).view(-1), radius.to(_F64).view(-1))
    n_pos = F.normalize(n_pos.to(dtype), dim=-1)
    n_neg = F.normalize(n_neg.to(dtype), dim=-1)
    g = F.normalize(g2d, dim=-1)
    loss_pos = 1.0 - F.cosine_similarity(g, F.normalize(n_pos[:, :2], dim=-1), dim=1)
    loss_neg = 1.0 - F.cosine_similarity(g, F.normalize(n_neg[:, :2], dim=-1), dim=1)
    swap = (loss_pos > loss_neg).unsqueeze(-1)
    normal = torch.nan_to_num(torch.where(swap, n_neg, n_pos), nan=0.0)
    centre = torch.nan_to_num(torch.where(swap, c_neg.to(dtype), c_pos.to(dtype)), nan=0.0)
    return normal, centre
