"""Pupil and iris ellipses from the segmentation (paper, Fig. 2 I(a)).

Each mask is thresholded and its boundary pixels are fitted with OpenCV.  Where
that fails although the object is present (e.g. an iris mostly hidden by the
eyelid) a soft conic fit on the probability map is used instead.  Confidence is
the share of the visible mask covered by the ellipse times the share of the
ellipse rim the mask actually observes.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vogeo_gaze.geometry.ellipse import canon_el2, conic_to_el2, el2_theta

EPS = 1e-6
MIN_AREA_PX = 16        # below this an object counts as absent (e.g. a blink)


# -- OpenCV boundary fit ------------------------------------------------------------

def _perimeter(bw, mask=None):
    """Pixels of ``bw`` with a 4-neighbour outside it (image border excluded)."""
    p = F.pad(bw, (1, 1, 1, 1), mode="constant", value=0)
    interior = ((p[:, :-2, 1:-1] == bw) & (p[:, 2:, 1:-1] == bw)
                & (p[:, 1:-1, :-2] == bw) & (p[:, 1:-1, 2:] == bw))
    perim = (~interior) & bw
    perim[:, 0, :] = perim[:, -1, :] = perim[:, :, 0] = perim[:, :, -1] = False
    return perim if mask is None else perim & mask


def _cv2_fit(perim):
    """cv2.fitEllipse per frame: (N,6) el2 (NaN where it fails) and a valid mask."""
    rows = []
    for img in perim.detach().cpu().numpy():
        pts = np.column_stack(np.where(img))
        if pts.shape[0] > 6:
            try:
                # Points are (row, col), so cv2's (x, y) centre comes back as (row, col).
                (r, c), (w, h), ang = cv2.fitEllipse(np.ascontiguousarray(pts.astype(np.float32)))
                rows.append(([c, r], w / 2.0, h / 2.0, np.pi / 2 - np.deg2rad(ang), True))
                continue
            except cv2.error:
                pass
        rows.append(([np.nan, np.nan], np.nan, np.nan, np.nan, False))
    centers, ws, hs, rads, ok = zip(*rows)
    dev = perim.device
    center = torch.tensor(centers, dtype=torch.float32, device=dev)
    two_theta = 2.0 * torch.tensor(rads, dtype=torch.float32, device=dev)
    el2 = canon_el2(torch.stack([center[:, 0], center[:, 1],
                                 torch.tensor(ws, dtype=torch.float32, device=dev),
                                 torch.tensor(hs, dtype=torch.float32, device=dev),
                                 torch.cos(two_theta), torch.sin(two_theta)], dim=-1))
    return el2, torch.tensor(ok, dtype=torch.bool, device=dev)


def coverage(prob, el2, visible=None, thr=0.5):
    """|ellipse & mask| / |mask| over the visible eye, and the ellipse mask."""
    dev = prob.device
    yy, xx = torch.meshgrid(torch.arange(prob.shape[-2], device=dev),
                            torch.arange(prob.shape[-1], device=dev), indexing="ij")
    theta = el2_theta(el2)
    x = xx.unsqueeze(0) - el2[:, 0].reshape(-1, 1, 1)
    y = yy.unsqueeze(0) - el2[:, 1].reshape(-1, 1, 1)
    ct, st = torch.cos(theta).reshape(-1, 1, 1), torch.sin(theta).reshape(-1, 1, 1)
    inside = (((x * ct + y * st) / el2[:, 2].reshape(-1, 1, 1)) ** 2
              + ((-x * st + y * ct) / el2[:, 3].reshape(-1, 1, 1)) ** 2) < 1
    seg = prob > thr
    if visible is not None:
        seg = seg & visible.bool()
    conf = (inside & seg).sum(dim=(-2, -1)).float() / (seg.sum(dim=(-2, -1)).float() + 1e-8)
    return conf, inside


def rim_support(prob, el2, n=64, inner=0.85, outer=1.15, thr=0.5):
    """Share of the ellipse rim with mask just inside and none just outside."""
    N, H, W = prob.shape
    th = el2_theta(el2)
    cx, cy, a, b = el2[:, 0], el2[:, 1], el2[:, 2].clamp_min(EPS), el2[:, 3].clamp_min(EPS)
    t = torch.linspace(0.0, 2.0 * math.pi, n + 1, device=prob.device, dtype=prob.dtype)[:-1].view(1, n)
    ct, st = torch.cos(t), torch.sin(t)
    cth, sth = torch.cos(th).view(N, 1), torch.sin(th).view(N, 1)

    def sample(s):
        x = cx.view(N, 1) + s * (a.view(N, 1) * ct * cth - b.view(N, 1) * st * sth)
        y = cy.view(N, 1) + s * (a.view(N, 1) * ct * sth + b.view(N, 1) * st * cth)
        grid = torch.stack([2.0 * x / max(W - 1, 1) - 1.0, 2.0 * y / max(H - 1, 1) - 1.0], dim=-1)
        return F.grid_sample(prob.unsqueeze(1), grid.view(N, 1, n, 2), mode="bilinear",
                             padding_mode="zeros", align_corners=True).view(N, n)

    return ((sample(inner) > thr) & (sample(outer) < thr)).to(prob.dtype).mean(dim=-1)


# -- soft conic fit (fallback) ----------------------------------------------------------

def _topk(w, K, min_rel, eps=1e-8):
    """Top-K weighted pixels (xs, ys, weights), each (N,K)."""
    N, H, W = w.shape
    wf = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).reshape(N, H * W).clamp_min(0.0)
    wf = torch.where(wf >= min_rel * wf.amax(dim=-1, keepdim=True), wf, torch.zeros_like(wf))
    vals, idx = torch.topk(wf, k=min(K, H * W), dim=-1, largest=True, sorted=False)
    wk = vals.clamp_min(0.0)
    wk = torch.where(wk.sum(dim=-1, keepdim=True) <= eps, torch.ones_like(wk), wk)
    return (idx % W).to(w.dtype), (idx // W).to(w.dtype), wk


def _bookstein(w, K=4096, min_rel=0.02, eps=EPS):
    """Ellipse-constrained (4AC - B^2 > 0) conic fit to the top-K weighted pixels."""
    orig = w.dtype
    with torch.autocast(device_type=w.device.type, enabled=False):
        w = w.float()
        N = w.shape[0]
        xs, ys, wk = _topk(w, K, min_rel, eps)
        sw = wk.sum(dim=-1, keepdim=True).clamp_min(eps)
        mx, my = (wk * xs).sum(dim=-1, keepdim=True) / sw, (wk * ys).sum(dim=-1, keepdim=True) / sw
        x0, y0 = xs - mx, ys - my
        s = torch.sqrt(2.0 / ((wk * (x0 * x0 + y0 * y0)).sum(dim=-1, keepdim=True) / sw).clamp_min(eps))
        xn, yn = x0 * s, y0 * s
        terms = torch.stack([xn * xn, xn * yn, yn * yn, xn, yn, torch.ones_like(xn)], dim=-1)
        S = (terms * wk.unsqueeze(-1)).transpose(1, 2) @ terms
        S1, S2, S3 = S[:, :3, :3], S[:, :3, 3:], S[:, 3:, 3:]
        eye3 = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(N, 3, 3)
        S3_inv = torch.linalg.solve(S3 + eps * eye3, eye3)
        C1 = torch.tensor([[0.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]],
                          device=w.device, dtype=w.dtype).unsqueeze(0).expand(N, 3, 3)
        eigvals, eigvecs = torch.linalg.eig(torch.linalg.solve(C1, S1 - S2 @ S3_inv @ S2.transpose(1, 2)))
        eigvals, cands = eigvals.real, eigvecs.real.transpose(1, 2)
        ok = (((4.0 * cands[:, :, 0] * cands[:, :, 2] - cands[:, :, 1] * cands[:, :, 1]) > eps)
              & torch.isfinite(cands).all(-1) & torch.isfinite(eigvals))
        has = ok.any(dim=1) & (wk.sum(dim=-1) > eps)
        a1 = cands[torch.arange(N, device=w.device),
                   torch.where(has, ok.float().argmax(dim=1), eigvals.abs().argmin(dim=1))]
        a2 = -(S3_inv @ (S2.transpose(1, 2) @ a1.unsqueeze(-1))).squeeze(-1)
        A, B, C, D, E, F0 = torch.cat([a1, a2], dim=1).unbind(dim=-1)
        mx, my, s = mx.view(N), my.view(N), s.view(N)
        s2 = s * s
        conic = torch.stack([A * s2, B * s2, C * s2,
                             -2.0 * A * s2 * mx - B * s2 * my + D * s,
                             -B * s2 * mx - 2.0 * C * s2 * my + E * s,
                             A * s2 * mx * mx + B * s2 * mx * my + C * s2 * my * my - D * s * mx - E * s * my + F0],
                            dim=-1)
        conic = conic * (1.0 - 2.0 * (conic[:, 5] > 0).to(conic.dtype).unsqueeze(-1))
        conic = conic / conic.norm(dim=-1, keepdim=True).clamp_min(eps)
    return conic.to(dtype=orig), has


def _refine(el2, w, steps=3, K=1024, min_rel=0.02):
    """Gradient steps on the weighted, normalised ellipse residual."""
    xs, ys, ws = _topk(w, K, min_rel)
    H, W = w.shape[1:]
    for _ in range(steps):
        with torch.enable_grad():
            p = el2.detach().clone().requires_grad_(True)
            ct, st = torch.cos(el2_theta(p)).unsqueeze(-1), torch.sin(el2_theta(p)).unsqueeze(-1)
            a, b = p[:, 2:3].clamp_min(EPS), p[:, 3:4].clamp_min(EPS)
            x, y = xs - p[:, 0:1], ys - p[:, 1:2]
            xp, yp = ct * x + st * y, -st * x + ct * y
            ia2, ib2 = 1.0 / (a * a), 1.0 / (b * b)
            d = (xp * xp * ia2 + yp * yp * ib2 - 1.0) / torch.sqrt(
                (2.0 * xp * ia2) ** 2 + (2.0 * yp * ib2) ** 2 + EPS)
            loss = ((ws * d * d).sum(dim=-1) / ws.sum(dim=-1).clamp_min(EPS)).mean()
            g = torch.autograd.grad(loss, p)[0]
        ang = F.normalize(torch.stack([p[:, 4] - 0.02 * g[:, 4], p[:, 5] - 0.02 * g[:, 5]], dim=-1),
                          dim=-1, eps=EPS)
        el2 = canon_el2(torch.stack([
            (p[:, 0] - 0.25 * g[:, 0]).clamp(0.0, float(W - 1)),
            (p[:, 1] - 0.25 * g[:, 1]).clamp(0.0, float(H - 1)),
            torch.exp(torch.log(p[:, 2].clamp_min(1.0)) - 0.05 * g[:, 2]).clamp_min(1.0),
            torch.exp(torch.log(p[:, 3].clamp_min(1.0)) - 0.05 * g[:, 3]).clamp_min(1.0),
            ang[:, 0], ang[:, 1]], dim=-1), eps=EPS)
    return el2


def _moments(p, ang_eps=1e-4):
    """Ellipse of the filled soft region from its second moments."""
    N, H, W = p.shape
    ys = torch.arange(H, device=p.device, dtype=p.dtype).reshape(1, H, 1)
    xs = torch.arange(W, device=p.device, dtype=p.dtype).reshape(1, 1, W)
    s = p.sum(dim=(1, 2)).clamp_min(EPS)
    u, v = (p * xs).sum(dim=(1, 2)) / s, (p * ys).sum(dim=(1, 2)) / s
    x0, y0 = xs - u.reshape(N, 1, 1), ys - v.reshape(N, 1, 1)
    xx = (p * x0 * x0).sum(dim=(1, 2)) / s
    yy = (p * y0 * y0).sum(dim=(1, 2)) / s
    xy = (p * x0 * y0).sum(dim=(1, 2)) / s
    diff, tr = xx - yy, xx + yy
    disc = torch.sqrt((diff * diff + 4.0 * xy * xy).clamp_min(0.0))
    a = (2.0 * torch.sqrt((0.5 * (tr + disc)).clamp_min(EPS))).clamp_min(1.0)
    b = (2.0 * torch.sqrt((0.5 * (tr - disc)).clamp_min(EPS))).clamp_min(1.0)
    two_theta = torch.atan2(2.0 * xy, diff + EPS)
    ill = disc < ang_eps
    c2 = torch.where(ill, torch.ones_like(two_theta), torch.cos(two_theta))
    s2 = torch.where(ill, torch.zeros_like(two_theta), torch.sin(two_theta))
    return canon_el2(torch.stack([u, v, a, b, c2, s2], dim=-1), eps=EPS)


class EllipseFit(nn.Module):
    def __init__(self, img_size=(240, 320)):
        super().__init__()
        self.H, self.W = img_size
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).reshape(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", kx, persistent=False)
        self.register_buffer("sobel_y", kx.transpose(2, 3).contiguous(), persistent=False)

    def _edge_weights(self, p, center, border):
        """Boundary weight 4p(1-p) x morphological x Sobel edge, windowed about the centre."""
        N, H, W = p.shape
        p = p.clamp(0, 1)
        p4 = p.unsqueeze(1)
        morph = (F.max_pool2d(p4, 5, 1, 2) - (-F.max_pool2d(-p4, 5, 1, 2))).squeeze(1).clamp_min(0.0)
        kx, ky = self.sobel_x.type_as(p), self.sobel_y.type_as(p)
        gx, gy = F.conv2d(p4, kx, padding=1), F.conv2d(p4, ky, padding=1)
        sobel = torch.sqrt(gx * gx + gy * gy + EPS).squeeze(1)
        w = (4.0 * p * (1.0 - p) * (morph * sobel).clamp_min(0.0)).clamp_min(0.0).pow(1.5)
        sigma = (2.0 * torch.sqrt(p.sum(dim=(1, 2)).clamp_min(EPS) / math.pi)).clamp(6.0, 120.0)
        ys = torch.arange(H, device=p.device, dtype=center.dtype).reshape(1, H, 1)
        xs = torch.arange(W, device=p.device, dtype=center.dtype).reshape(1, 1, W)
        win = torch.exp(-((xs - center[:, 0].reshape(N, 1, 1)).square()
                          + (ys - center[:, 1].reshape(N, 1, 1)).square())
                        / (2.0 * sigma.reshape(N, 1, 1).clamp_min(1.0).square()))
        win = torch.where(border.reshape(N, 1, 1), torch.ones_like(win), win)
        return (w * win).clamp_min(0.0)

    def _fit_soft(self, p, p_eye, visible):
        mass = p.sum(dim=(1, 2))
        support = (torch.sigmoid(10.0 * (mass - 50.0) / (50.0 + EPS))
                   * ((p * p_eye).sum(dim=(1, 2)) / p.sum(dim=(1, 2)).clamp_min(EPS)).clamp(0, 1)).clamp(0, 1)
        pc = p.clamp(0, 1)
        border = torch.stack([pc[:, 0, :].amax(dim=-1), pc[:, -1, :].amax(dim=-1),
                              pc[:, :, 0].amax(dim=-1), pc[:, :, -1].amax(dim=-1)], dim=-1).amax(dim=-1) > 0.1
        el_mom = _moments(p)
        w = self._edge_weights(p, el_mom[:, :2], border)
        conic, has = _bookstein(w)
        el_conic, valid = conic_to_el2(conic, eps=EPS)
        # A binary mask has no boundary weight; use the moment ellipse there.
        valid = has & valid & (w.flatten(1).amax(dim=1) > 0)
        fallback = ~valid | ((support < 0.2) & ~border)
        el = canon_el2(torch.where(fallback.unsqueeze(-1), el_mom, _refine(el_conic, w)), eps=EPS)
        # Moment centre (robust on partial masks) with the conic's axes and angle.
        el = canon_el2(torch.cat([el_mom[:, :2], el[:, 2:]], dim=-1), eps=EPS)
        return el, coverage(p, el, visible)[0]

    def _fit_cv2(self, p, mask, visible):
        roi = p > 0.5
        el2, ok = _cv2_fit(_perimeter(roi, mask))
        conf, inside = coverage(p * roi, el2, visible)
        ok &= roi.sum(dim=(-2, -1)).float() != 0
        return torch.nan_to_num(el2, nan=0.0, posinf=0.0, neginf=0.0), inside, conf, ok

    def forward(self, prob):
        """prob (N,3,H,W): pupil / iris / open-eye probabilities."""
        clean = lambda x: torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
        p_pup, p_iris, p_eye = clean(prob[:, 0]), clean(prob[:, 1]), clean(prob[:, 2])
        visible = p_eye > 0.5
        open_eye = prob[:, -1] > 0.5
        pup_present = (p_pup > 0.5).flatten(1).sum(1) >= MIN_AREA_PX
        iris_present = (p_iris > 0.5).flatten(1).sum(1) >= MIN_AREA_PX
        pup_el2, _, pup_conf, pup_ok = self._fit_cv2(p_pup, None, visible)
        iris_el2, iris_in, iris_conf, iris_ok = self._fit_cv2(p_iris, open_eye, visible)
        if not bool(pup_ok.all()) or not bool(iris_ok.all()):
            pup_soft, pup_conf_soft = self._fit_soft(p_pup, p_eye, visible)
            iris_soft, iris_conf_soft = self._fit_soft(p_iris, p_eye, visible)
            use_pup, use_iris = ~pup_ok & pup_present, ~iris_ok & iris_present
            pup_el2 = torch.where(use_pup[:, None], pup_soft, pup_el2)
            iris_el2 = torch.where(use_iris[:, None], iris_soft, iris_el2)
            pup_conf = torch.where(use_pup, pup_conf_soft, pup_conf)
            iris_conf = torch.where(use_iris, iris_conf_soft, iris_conf)
        pup_conf = torch.where(pup_present, pup_conf, torch.zeros_like(pup_conf))
        iris_conf = torch.where(iris_present, iris_conf, torch.zeros_like(iris_conf))
        pup_conf = (pup_conf * rim_support(p_pup.float(), pup_el2.float()).to(p_pup.dtype)).clamp(0, 1)
        iris_conf = (iris_conf * rim_support(p_iris.float(), iris_el2.float()).to(p_iris.dtype)).clamp(0, 1)
        # Share of the iris ellipse inside the open eye; 0 without both ellipses.
        open_score = (iris_in & open_eye).sum((-2, -1)) / (iris_in.sum((-2, -1)) + 1e-8)
        valid = pup_ok & iris_ok & pup_present & iris_present
        open_score = torch.where(valid, open_score, torch.zeros_like(open_score))
        return {"entpup_el2": pup_el2, "iris_el2": iris_el2, "pupil_conf": pup_conf,
                "iris_conf": iris_conf, "open_eye_score": open_score}
