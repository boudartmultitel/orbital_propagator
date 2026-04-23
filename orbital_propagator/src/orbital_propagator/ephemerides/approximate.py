from __future__ import annotations

from math import cos, pi, sin

import numpy as np


AU_M = 149_597_870_700.0
SUN_MU_M3_S2 = 1.32712440018e20
MOON_MU_M3_S2 = 4.9048695e12
MOON_SEMIMAJOR_AXIS_M = 384_400_000.0
MOON_INCLINATION_DEG = 5.145
SOLAR_RADIATION_PRESSURE_1AU_N_M2 = 4.56e-6
SIDEREAL_YEAR_S = 365.256363004 * 86400.0
SIDEREAL_MONTH_S = 27.321661 * 86400.0


def _rotation_x(angle_rad: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(angle_rad), -sin(angle_rad)],
            [0.0, sin(angle_rad), cos(angle_rad)],
        ],
        dtype=float,
    )


def approximate_body_position_m(body: str, elapsed_s: float) -> np.ndarray:
    if body == "sun":
        mean_motion = 2.0 * pi / SIDEREAL_YEAR_S
        angle = mean_motion * elapsed_s
        return AU_M * np.array([cos(angle), sin(angle), 0.0], dtype=float)

    if body == "moon":
        mean_motion = 2.0 * pi / SIDEREAL_MONTH_S
        angle = mean_motion * elapsed_s
        orbital_plane_position = MOON_SEMIMAJOR_AXIS_M * np.array(
            [cos(angle), sin(angle), 0.0],
            dtype=float,
        )
        return _rotation_x(np.deg2rad(MOON_INCLINATION_DEG)) @ orbital_plane_position

    raise ValueError(f"Unsupported approximate body: {body}")
