from __future__ import annotations

import numpy as np


def central_gravity_acceleration(position_m: np.ndarray, mu_m3_s2: float) -> np.ndarray:
    radius_m = np.linalg.norm(position_m)
    if radius_m == 0.0:
        raise ValueError("Position vector norm must be non-zero.")
    return -mu_m3_s2 * position_m / radius_m**3
