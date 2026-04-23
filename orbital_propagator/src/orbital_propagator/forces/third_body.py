from __future__ import annotations

import numpy as np


def third_body_point_mass_acceleration(
    spacecraft_position_m: np.ndarray,
    third_body_position_m: np.ndarray,
    third_body_mu_m3_s2: float,
) -> np.ndarray:
    relative_position_m = third_body_position_m - spacecraft_position_m
    relative_norm_m = np.linalg.norm(relative_position_m)
    body_norm_m = np.linalg.norm(third_body_position_m)
    if relative_norm_m == 0.0 or body_norm_m == 0.0:
        raise ValueError("Third-body geometry must be non-degenerate.")

    return third_body_mu_m3_s2 * (
        relative_position_m / relative_norm_m**3
        - third_body_position_m / body_norm_m**3
    )
