from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from orbital_propagator.config import SimulationRequest
from orbital_propagator.ephemerides.approximate import (
    MOON_MU_M3_S2,
    SOLAR_RADIATION_PRESSURE_1AU_N_M2,
    SUN_MU_M3_S2,
)
from orbital_propagator.propagation.runner import SimulationResult


INPUT_SCHEMA_VERSION = "0.1.0"
INPUT_STORAGE_FORMAT = "referenced_series_v1"
FEATURE_NAMES = (
    "x_m",
    "y_m",
    "z_m",
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "radius_m",
    "altitude_m",
    "inverse_radius_1_m",
    "x_over_r",
    "y_over_r",
    "z_over_r",
    "atmosphere_vx_m_s",
    "atmosphere_vy_m_s",
    "atmosphere_vz_m_s",
    "density_kg_m3",
    "mu_m3_s2",
    "body_radius_m",
    "J2",
    "J2_reference_radius_m",
    "rotation_rate_rad_s",
    "sun_mu_m3_s2",
    "moon_mu_m3_s2",
    "drag_coefficient",
    "reflectivity_coefficient",
    "area_to_mass_m2_kg",
    "sun_x_m",
    "sun_y_m",
    "sun_z_m",
    "moon_x_m",
    "moon_y_m",
    "moon_z_m",
    "P_srp_1AU_N_m2",
)

CONSTANT_FEATURE_NAMES = (
    "mu_m3_s2",
    "body_radius_m",
    "J2",
    "J2_reference_radius_m",
    "rotation_rate_rad_s",
    "sun_mu_m3_s2",
    "moon_mu_m3_s2",
    "drag_coefficient",
    "reflectivity_coefficient",
    "area_to_mass_m2_kg",
    "P_srp_1AU_N_m2",
)

_POSITION_FEATURES = (
    "x_m",
    "y_m",
    "z_m",
    "radius_m",
    "altitude_m",
    "body_radius_m",
    "J2_reference_radius_m",
    "sun_x_m",
    "sun_y_m",
    "sun_z_m",
    "moon_x_m",
    "moon_y_m",
    "moon_z_m",
)
_VELOCITY_FEATURES = (
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "atmosphere_vx_m_s",
    "atmosphere_vy_m_s",
    "atmosphere_vz_m_s",
)
_DIMENSIONLESS_FEATURES = (
    "x_over_r",
    "y_over_r",
    "z_over_r",
    "J2",
    "drag_coefficient",
    "reflectivity_coefficient",
)
FEATURE_UNITS = {
    **dict.fromkeys(_POSITION_FEATURES, "m"),
    **dict.fromkeys(_VELOCITY_FEATURES, "m/s"),
    "inverse_radius_1_m": "1/m",
    **dict.fromkeys(_DIMENSIONLESS_FEATURES, "dimensionless"),
    "density_kg_m3": "kg/m3",
    "mu_m3_s2": "m3/s2",
    "sun_mu_m3_s2": "m3/s2",
    "moon_mu_m3_s2": "m3/s2",
    "rotation_rate_rad_s": "rad/s",
    "area_to_mass_m2_kg": "m2/kg",
    "P_srp_1AU_N_m2": "N/m2",
}


def build_constant_inputs(request: SimulationRequest) -> dict[str, float]:
    """Return input features that remain constant over one trajectory."""
    body = request.central_body
    return {
        "mu_m3_s2": body.mu_m3_s2,
        "body_radius_m": body.radius_m,
        "J2": body.j2 or 0.0,
        "J2_reference_radius_m": body.j2_reference_radius_m or body.radius_m,
        "rotation_rate_rad_s": body.rotation_rate_rad_s,
        "sun_mu_m3_s2": SUN_MU_M3_S2,
        "moon_mu_m3_s2": MOON_MU_M3_S2 if body.name.lower() == "earth" else 0.0,
        "drag_coefficient": request.spacecraft.drag_coefficient,
        "reflectivity_coefficient": request.spacecraft.reflectivity_coefficient,
        "area_to_mass_m2_kg": (
            request.spacecraft.cross_section_area_m2 / request.spacecraft.mass_kg
        ),
        "P_srp_1AU_N_m2": SOLAR_RADIATION_PRESSURE_1AU_N_M2,
    }


def _assemble_input_vectors(
    states_m_s: np.ndarray,
    radius_m: np.ndarray,
    altitude_m: np.ndarray,
    environment: Mapping[str, Any],
    constant_inputs: Mapping[str, float],
) -> np.ndarray:
    count = len(states_m_s)
    radius_safe_m = np.maximum(radius_m, np.finfo(float).tiny)
    positions_m = states_m_s[:, :3]
    constants = {
        name: np.full(count, float(constant_inputs[name]))
        for name in CONSTANT_FEATURE_NAMES
    }
    inputs = np.column_stack(
        (
            states_m_s,
            radius_m,
            altitude_m,
            1.0 / radius_safe_m,
            positions_m / radius_safe_m[:, None],
            environment["atmosphere_velocity_m_s"],
            environment["atmosphere_density_kg_m3"],
            constants["mu_m3_s2"],
            constants["body_radius_m"],
            constants["J2"],
            constants["J2_reference_radius_m"],
            constants["rotation_rate_rad_s"],
            constants["sun_mu_m3_s2"],
            constants["moon_mu_m3_s2"],
            constants["drag_coefficient"],
            constants["reflectivity_coefficient"],
            constants["area_to_mass_m2_kg"],
            environment["sun_position_m"],
            environment["moon_position_m"],
            constants["P_srp_1AU_N_m2"],
        )
    ).astype(float)
    expected_shape = (count, len(FEATURE_NAMES))
    if inputs.shape != expected_shape:
        raise ValueError(
            f"Canonical inputs must have shape {expected_shape}, got {inputs.shape}."
        )
    if not np.isfinite(inputs).all():
        raise ValueError("Canonical inputs contain non-finite values.")
    return inputs


def build_input_vectors(
    request: SimulationRequest,
    result: SimulationResult,
) -> np.ndarray:
    """Build the canonical physical [N,33] model input matrix."""
    return _assemble_input_vectors(
        result.states_m_s,
        result.derived_series["radius_m"],
        result.derived_series["altitude_m"],
        result.environment_series,
        build_constant_inputs(request),
    )


def materialize_input_vectors(artifact: Mapping[str, Any]) -> np.ndarray:
    """Reconstruct the canonical matrix from a compact or legacy artifact."""
    if "inputs" in artifact:  # Schema 0.6.0 and earlier.
        inputs = np.asarray(artifact["inputs"], dtype=float)
    else:
        if artifact.get("input_storage_format") != INPUT_STORAGE_FORMAT:
            raise ValueError("Unsupported or missing compact input storage format.")
        if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Artifact feature_names do not match the canonical schema.")
        derived = artifact["derived_series"]
        inputs = _assemble_input_vectors(
            np.asarray(artifact["states_m_s"], dtype=float),
            np.asarray(derived["radius_m"], dtype=float),
            np.asarray(derived["altitude_m"], dtype=float),
            artifact["environment_series"],
            artifact["constant_inputs"],
        )
    if inputs.ndim != 2 or inputs.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Materialized inputs must have {len(FEATURE_NAMES)} columns.")
    return inputs
