"""Anatomical priors A (paper, Table 1), in mm."""
R_EYE = 12.0        # eyeball radius r_eye
R_CORNEA = 7.8      # corneal radius r_cor
R_IRIS = 6.0        # iris radius r_iris
N_CORNEA = 1.3375   # refractive index n_ref
N_AIR = 1.0


def limbus_distance(r_eye, r_iris):
    """L_p: eyeball centre to the iris/pupil plane (tensors)."""
    return (r_eye ** 2 - r_iris ** 2).clamp_min(1e-12).sqrt()


def cornea_center_distance(r_eye, r_iris, r_cornea):
    """L_ec: eyeball centre to corneal-sphere centre, so the limbus lies on the cornea."""
    return limbus_distance(r_eye, r_iris) - (r_cornea ** 2 - r_iris ** 2).clamp_min(1e-12).sqrt()
