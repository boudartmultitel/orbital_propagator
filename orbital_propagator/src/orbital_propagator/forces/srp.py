from __future__ import annotations

import numpy as np


def cylindrical_eclipse_visibility(
    spacecraft_position_m: np.ndarray,
    sun_position_m: np.ndarray,
    body_radius_m: float,
) -> float:
    """Return binary SRP visibility for a cylindrical central-body shadow."""
    if body_radius_m <= 0.0:
        raise ValueError("Central-body radius must be strictly positive.")
    sun_distance_m = np.linalg.norm(sun_position_m)
    if sun_distance_m == 0.0:
        raise ValueError("Sun position must not coincide with the central body.")

    sun_direction = sun_position_m / sun_distance_m
    axial_position_m = float(np.dot(spacecraft_position_m, sun_direction))
    radial_from_shadow_axis_m = np.linalg.norm(
        spacecraft_position_m - axial_position_m * sun_direction
    )
    eclipsed = axial_position_m < 0.0 and radial_from_shadow_axis_m < body_radius_m
    return 0.0 if eclipsed else 1.0


def solar_radiation_pressure_acceleration(
    spacecraft_position_m: np.ndarray,
    sun_position_m: np.ndarray,
    reflectivity_coefficient: float,
    cross_section_area_m2: float,
    mass_kg: float,
    solar_pressure_1au_n_m2: float,
    astronomical_unit_m: float,
    visibility: float = 1.0,
) -> np.ndarray:
    if not 0.0 <= visibility <= 1.0:
        raise ValueError("SRP visibility must be between zero and one.")
    sun_to_spacecraft_m = spacecraft_position_m - sun_position_m
    distance_m = np.linalg.norm(sun_to_spacecraft_m)
    if distance_m == 0.0:
        raise ValueError("Spacecraft position must not coincide with the Sun position.")

    pressure_n_m2 = solar_pressure_1au_n_m2 * (astronomical_unit_m / distance_m) ** 2
    scale = (
        visibility
        * pressure_n_m2
        * reflectivity_coefficient
        * cross_section_area_m2
        / mass_kg
    )
    return scale * sun_to_spacecraft_m / distance_m
