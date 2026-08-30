from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, sqrt

import numpy as np


@dataclass(frozen=True)
class CentralBodyConfig:
    name: str
    mu_m3_s2: float
    radius_m: float
    j2: float | None = None
    rotation_rate_rad_s: float = 0.0
    atmosphere_density_sea_level_kg_m3: float = 0.0
    atmosphere_scale_height_m: float = 1.0
    atmosphere_model: str = "none"
    heliocentric_distance_au: float | None = None
    j2_reference_radius_m: float | None = None


@dataclass(frozen=True)
class IntegratorConfig:
    backend: str = "auto"
    method: str = "DOP853"
    rtol: float = 1e-9
    atol: float = 1e-9
    max_step_s: float | None = None


@dataclass(frozen=True)
class PropagationConfig:
    duration_s: float
    sample_count: int
    start_time_s: float = 0.0
    start_epoch_utc: str = "2026-01-01T00:00:00Z"


@dataclass(frozen=True)
class SpacecraftConfig:
    mass_kg: float = 1000.0
    cross_section_area_m2: float = 10.0
    drag_coefficient: float = 2.2
    reflectivity_coefficient: float = 1.2


@dataclass(frozen=True)
class ForceModelConfig:
    central_gravity: bool = True
    j2: bool = False
    drag: bool = False
    atmosphere_model: str = "piecewise_exponential"
    corotating_atmosphere: bool = True
    solar_radiation_pressure: bool = False
    third_body_sun: bool = False
    third_body_moon: bool = False


@dataclass(frozen=True)
class SimulationRequest:
    run_name: str
    producer: str
    central_body: CentralBodyConfig
    initial_state_m_s: np.ndarray
    propagation: PropagationConfig
    integrator: IntegratorConfig
    spacecraft: SpacecraftConfig
    forces: ForceModelConfig


def validate_force_model(
    central_body: CentralBodyConfig,
    forces: ForceModelConfig,
) -> None:
    """Reject force combinations unsupported by the selected central body."""
    if forces.j2 and central_body.j2 is None:
        raise ValueError(f"J2 is unavailable for central body {central_body.name}.")
    if forces.drag and central_body.atmosphere_model == "none":
        raise ValueError(
            f"Atmospheric drag is unavailable for central body {central_body.name}."
        )
    if forces.third_body_moon and central_body.name.lower() != "earth":
        raise ValueError("The Moon third-body model is only available for Earth.")
    if (
        forces.third_body_sun or forces.solar_radiation_pressure
    ) and central_body.heliocentric_distance_au is None:
        raise ValueError(
            f"Heliocentric distance is required for central body {central_body.name}."
        )


def _validate_orbit_above_surface(
    *,
    central_body: CentralBodyConfig,
    periapsis_radius_m: float,
) -> None:
    periapsis_altitude_m = periapsis_radius_m - central_body.radius_m
    if periapsis_altitude_m < 0.0:
        raise ValueError(
            "The requested orbit intersects the central body: "
            f"periapsis altitude is {periapsis_altitude_m / 1_000.0:.3f} km."
        )


def _perifocal_to_inertial_rotation(
    *,
    inclination_deg: float,
    raan_deg: float,
    argument_of_periapsis_deg: float,
) -> np.ndarray:
    inclination = radians(inclination_deg)
    raan = radians(raan_deg)
    argument_of_periapsis = radians(argument_of_periapsis_deg)

    rotation_argument_of_periapsis = np.array(
        [
            [cos(argument_of_periapsis), -sin(argument_of_periapsis), 0.0],
            [sin(argument_of_periapsis), cos(argument_of_periapsis), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rotation_raan = np.array(
        [
            [cos(raan), -sin(raan), 0.0],
            [sin(raan), cos(raan), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rotation_inclination = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(inclination), -sin(inclination)],
            [0.0, sin(inclination), cos(inclination)],
        ],
        dtype=float,
    )
    return rotation_raan @ rotation_inclination @ rotation_argument_of_periapsis


def keplerian_orbit_state(
    central_body: CentralBodyConfig,
    semimajor_axis_m: float,
    eccentricity: float,
    inclination_deg: float = 0.0,
    raan_deg: float = 0.0,
    argument_of_periapsis_deg: float = 0.0,
    true_anomaly_deg: float = 0.0,
) -> np.ndarray:
    if semimajor_axis_m <= 0.0:
        raise ValueError("Semimajor axis must be strictly positive.")
    if eccentricity < 0.0 or eccentricity >= 1.0:
        raise ValueError("Eccentricity must satisfy 0 <= e < 1 for elliptical orbits.")
    periapsis_radius_m = semimajor_axis_m * (1.0 - eccentricity)
    _validate_orbit_above_surface(
        central_body=central_body,
        periapsis_radius_m=periapsis_radius_m,
    )

    nu = radians(true_anomaly_deg)
    semi_latus_rectum_m = semimajor_axis_m * (1.0 - eccentricity**2)
    if semi_latus_rectum_m <= 0.0:
        raise ValueError("Semimajor axis and eccentricity produce an invalid orbit.")

    radius_m = semi_latus_rectum_m / (1.0 + eccentricity * cos(nu))
    speed_scale_m_s = sqrt(central_body.mu_m3_s2 / semi_latus_rectum_m)

    position_perifocal = np.array(
        [radius_m * cos(nu), radius_m * sin(nu), 0.0], dtype=float
    )
    velocity_perifocal = np.array(
        [-speed_scale_m_s * sin(nu), speed_scale_m_s * (eccentricity + cos(nu)), 0.0],
        dtype=float,
    )
    rotation_matrix = _perifocal_to_inertial_rotation(
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        argument_of_periapsis_deg=argument_of_periapsis_deg,
    )
    position = rotation_matrix @ position_perifocal
    velocity = rotation_matrix @ velocity_perifocal

    return np.concatenate([position, velocity])


def circular_orbit_state(
    central_body: CentralBodyConfig,
    altitude_m: float,
    inclination_deg: float = 0.0,
    raan_deg: float = 0.0,
    true_anomaly_deg: float = 0.0,
) -> np.ndarray:
    if altitude_m < 0.0:
        raise ValueError("Altitude must be non-negative for circular orbit initialization.")
    semimajor_axis_m = central_body.radius_m + altitude_m
    return keplerian_orbit_state(
        central_body=central_body,
        semimajor_axis_m=semimajor_axis_m,
        eccentricity=0.0,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        argument_of_periapsis_deg=0.0,
        true_anomaly_deg=true_anomaly_deg,
    )
