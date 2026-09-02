from __future__ import annotations

import numpy as np

from orbital_propagator.config import (
    CentralBodyConfig,
    ForceModelConfig,
    SpacecraftConfig,
    validate_force_model,
)
from orbital_propagator.ephemerides.approximate import (
    AU_M,
    MOON_MU_M3_S2,
    SOLAR_RADIATION_PRESSURE_1AU_N_M2,
    SUN_MU_M3_S2,
)
from orbital_propagator.ephemerides.provider import (
    body_position_m,
    sun_position_for_central_body_m,
)
from orbital_propagator.forces.drag import atmospheric_drag_acceleration
from orbital_propagator.forces.gravity import central_gravity_acceleration
from orbital_propagator.forces.j2 import j2_acceleration
from orbital_propagator.forces.srp import (
    cylindrical_eclipse_visibility,
    solar_radiation_pressure_acceleration,
)
from orbital_propagator.forces.third_body import third_body_point_mass_acceleration


def evaluate_accelerations(
    elapsed_time_s: float,
    state_m_s: np.ndarray,
    central_body: CentralBodyConfig,
    spacecraft: SpacecraftConfig,
    forces: ForceModelConfig,
    start_epoch_utc: str,
) -> dict[str, np.ndarray]:
    validate_force_model(central_body, forces)
    position_m = state_m_s[:3]
    velocity_m_s = state_m_s[3:]
    accelerations: dict[str, np.ndarray] = {}

    if forces.central_gravity:
        accelerations["central_gravity"] = central_gravity_acceleration(
            position_m, central_body.mu_m3_s2
        )
    if forces.j2:
        accelerations["j2"] = j2_acceleration(
            position_m=position_m,
            mu_m3_s2=central_body.mu_m3_s2,
            equatorial_radius_m=(
                central_body.j2_reference_radius_m or central_body.radius_m
            ),
            j2=central_body.j2,
        )
    if forces.drag:
        accelerations["drag"] = atmospheric_drag_acceleration(
            position_m=position_m,
            velocity_m_s=velocity_m_s,
            start_epoch_utc=start_epoch_utc,
            elapsed_time_s=elapsed_time_s,
            body_radius_m=central_body.radius_m,
            body_rotation_rate_rad_s=central_body.rotation_rate_rad_s,
            density_sea_level_kg_m3=central_body.atmosphere_density_sea_level_kg_m3,
            scale_height_m=central_body.atmosphere_scale_height_m,
            drag_coefficient=spacecraft.drag_coefficient,
            cross_section_area_m2=spacecraft.cross_section_area_m2,
            mass_kg=spacecraft.mass_kg,
            atmosphere_model=forces.atmosphere_model,
            corotating_atmosphere=forces.corotating_atmosphere,
        )
    if forces.third_body_sun or forces.solar_radiation_pressure:
        sun_position = sun_position_for_central_body_m(
            start_epoch_utc,
            elapsed_time_s,
            central_body.heliocentric_distance_au,
        )
    else:
        sun_position = None
    if forces.third_body_sun and sun_position is not None:
        accelerations["third_body_sun"] = third_body_point_mass_acceleration(
            spacecraft_position_m=position_m,
            third_body_position_m=sun_position,
            third_body_mu_m3_s2=SUN_MU_M3_S2,
        )
    if forces.third_body_moon:
        accelerations["third_body_moon"] = third_body_point_mass_acceleration(
            spacecraft_position_m=position_m,
            third_body_position_m=body_position_m("moon", start_epoch_utc, elapsed_time_s),
            third_body_mu_m3_s2=MOON_MU_M3_S2,
        )
    if forces.solar_radiation_pressure and sun_position is not None:
        visibility = cylindrical_eclipse_visibility(
            position_m, sun_position, central_body.radius_m
        )
        accelerations["solar_radiation_pressure"] = (
            solar_radiation_pressure_acceleration(
                spacecraft_position_m=position_m,
                sun_position_m=sun_position,
                reflectivity_coefficient=spacecraft.reflectivity_coefficient,
                cross_section_area_m2=spacecraft.cross_section_area_m2,
                mass_kg=spacecraft.mass_kg,
                solar_pressure_1au_n_m2=SOLAR_RADIATION_PRESSURE_1AU_N_M2,
                astronomical_unit_m=AU_M,
                visibility=visibility,
            )
        )

    return accelerations


def state_derivative(
    elapsed_time_s: float,
    state_m_s: np.ndarray,
    central_body: CentralBodyConfig,
    spacecraft: SpacecraftConfig,
    forces: ForceModelConfig,
    start_epoch_utc: str,
) -> np.ndarray:
    velocity_m_s = state_m_s[3:]
    accelerations = evaluate_accelerations(
        elapsed_time_s,
        state_m_s,
        central_body,
        spacecraft,
        forces,
        start_epoch_utc,
    )
    total_acceleration = sum(accelerations.values())
    return np.concatenate([velocity_m_s, total_acceleration])
