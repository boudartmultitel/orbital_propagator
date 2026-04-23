from __future__ import annotations

import numpy as np


def solar_radiation_pressure_acceleration(
    spacecraft_position_m: np.ndarray,
    sun_position_m: np.ndarray,
    reflectivity_coefficient: float,
    cross_section_area_m2: float,
    mass_kg: float,
    solar_pressure_1au_n_m2: float,
    astronomical_unit_m: float,
) -> np.ndarray:
    sun_to_spacecraft_m = spacecraft_position_m - sun_position_m
    distance_m = np.linalg.norm(sun_to_spacecraft_m)
    if distance_m == 0.0:
        raise ValueError("Spacecraft position must not coincide with the Sun position.")

    pressure_n_m2 = solar_pressure_1au_n_m2 * (astronomical_unit_m / distance_m) ** 2
    scale = pressure_n_m2 * reflectivity_coefficient * cross_section_area_m2 / mass_kg
    return scale * sun_to_spacecraft_m / distance_m
