from __future__ import annotations

import numpy as np


def j2_acceleration(
    position_m: np.ndarray,
    mu_m3_s2: float,
    equatorial_radius_m: float,
    j2: float,
) -> np.ndarray:
    radius_m = np.linalg.norm(position_m)
    if radius_m == 0.0:
        raise ValueError("Position vector norm must be non-zero.")

    x_m, y_m, z_m = position_m
    radius_sq_m2 = radius_m * radius_m
    z_sq_m2 = z_m * z_m
    scale = 1.5 * j2 * mu_m3_s2 * equatorial_radius_m**2 / radius_m**5
    common_xy = 5.0 * z_sq_m2 / radius_sq_m2 - 1.0
    common_z = 5.0 * z_sq_m2 / radius_sq_m2 - 3.0

    return scale * np.array(
        [x_m * common_xy, y_m * common_xy, z_m * common_z],
        dtype=float,
    )
