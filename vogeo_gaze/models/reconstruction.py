"""Geometry-constrained eye reconstruction (paper, Eq. 3); no learned parameters."""
import torch
import torch.nn.functional as F

from vogeo_gaze.geometry.anatomy import R_EYE, R_IRIS, limbus_distance
from vogeo_gaze.geometry.camera import line_plane_intersect, pixel_to_ray, rays_plane_intersect
from vogeo_gaze.geometry.ellipse import el2_to_el, ellipse_points
from vogeo_gaze.geometry.unprojection import unproject_ellipse


def _camera_centred(el, img_size):
    """el with a >= b, theta in [-pi/2, pi/2), and the centre relative to the principal point."""
    theta, cx, cy, a, b = el.unbind(-1)
    swap = b > a
    a, b = torch.where(swap, b, a), torch.where(swap, a, b)
    theta = torch.where(swap, theta + torch.pi / 2.0, theta)
    theta = ((theta + torch.pi / 2) % torch.pi) - torch.pi / 2
    H, W = img_size
    return torch.stack([theta, cx - (W - 1) / 2.0, cy - (H - 1) / 2.0, a.clamp_min(1e-9), b.clamp_min(1e-9)], dim=-1)


def reconstruct_eye(pup_el2, iris_el2, g, fpx, img_size):
    """Eye parameters P from the corrected pupil and the iris ellipse, per frame.

    pup_el2 (N,6) refraction-corrected pupil E_pup, iris_el2 (N,6) E_iris,
    g (N,2) signed prior gaze direction.  Returns c_eye, n (gaze), c_pup (N,3), r_pup (N,1).
    """
    N, dev = pup_el2.shape[0], pup_el2.device
    full = lambda v: torch.full((N, 1), v, device=dev, dtype=torch.float32)
    f = full(float(fpx))
    pup_el = el2_to_el(pup_el2)
    pup_c = _camera_centred(pup_el, img_size)
    iris_c = _camera_centred(el2_to_el(iris_el2), img_size)
    # (c~_pup, n) = Pi_{f,1}(E_pup, g).
    n, c_pup0 = unproject_ellipse(pup_c, full(1.0), f, g)
    # The iris ellipse takes the pupil's shape at its own size (concentric circles).
    axes = pup_c[:, 3:5]
    iris_c = torch.cat([iris_c[:, :3], (iris_c[:, 3:5] / axes).mean(dim=1, keepdim=True) * axes], dim=1)
    # c~_iris = Pi_{f,r_iris}(E_iris, g); c_pup is the pupil ray's hit on the iris plane.
    _, c_iris0 = unproject_ellipse(iris_c, full(R_IRIS), f, g)
    cam = torch.zeros_like(n)
    c_pup = line_plane_intersect(cam, F.normalize(c_pup0 - cam, dim=-1, eps=1e-6), c_iris0, n)
    c_eye = c_pup - limbus_distance(full(R_EYE), full(R_IRIS)) * n
    # Pupil radius: the corrected rim back-projected onto the pupil plane.
    rays = pixel_to_ray(ellipse_points(pup_el, 100), fpx, img_size)
    rim, _ = rays_plane_intersect(rays, c_pup, n)
    r_pup = torch.linalg.norm(rim - c_pup[:, None, :], dim=-1).mean(dim=1, keepdim=True)
    return c_eye, n, c_pup, r_pup
