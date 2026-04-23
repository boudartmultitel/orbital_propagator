from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, sqrt

import numpy as np


@dataclass(frozen=True)
class CentralBodyConfig:
    name: str
    mu_m3_s2: float
    radius_m: float
    j2: float = 0.0
    rotation_rate_rad_s: float = 0.0
    atmosphere_density_sea_level_kg_m3: float = 0.0
    atmosphere_scale_height_m: float = 1.0


@dataclass(frozen=True)
class IntegratorConfig:
    backend: str = "auto"
    method: str = "RK45"
    rtol: float = 1e-9
    atol: float = 1e-9


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


def circular_orbit_state(
    central_body: CentralBodyConfig,
    altitude_m: float,
    inclination_deg: float = 0.0,
    raan_deg: float = 0.0,
    true_anomaly_deg: float = 0.0,
) -> np.ndarray:
    radius_m = central_body.radius_m + altitude_m
    speed_m_s = sqrt(central_body.mu_m3_s2 / radius_m)

    nu = radians(true_anomaly_deg)
    inclination = radians(inclination_deg)
    raan = radians(raan_deg)

    position_perifocal = np.array(
        [radius_m * cos(nu), radius_m * sin(nu), 0.0], dtype=float
    )
    velocity_perifocal = np.array(
        [-speed_m_s * sin(nu), speed_m_s * cos(nu), 0.0], dtype=float
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

    rotation_matrix = rotation_raan @ rotation_inclination
    position = rotation_matrix @ position_perifocal
    velocity = rotation_matrix @ velocity_perifocal

    return np.concatenate([position, velocity])
