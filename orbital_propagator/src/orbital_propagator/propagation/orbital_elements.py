from __future__ import annotations

import numpy as np

from orbital_propagator.config import CentralBodyConfig


def compute_derived_series(
    states_m_s: np.ndarray,
    central_body: CentralBodyConfig,
) -> dict[str, np.ndarray]:
    eps = 1e-12
    positions_m = states_m_s[:, :3]
    velocities_m_s = states_m_s[:, 3:]

    radius_m = np.linalg.norm(positions_m, axis=1)
    radius_safe_m = np.maximum(radius_m, eps)
    altitude_m = radius_m - central_body.radius_m
    speed_m_s = np.linalg.norm(velocities_m_s, axis=1)
    specific_energy_j_kg = 0.5 * speed_m_s**2 - central_body.mu_m3_s2 / radius_safe_m
    specific_energy_safe_j_kg = np.where(
        np.abs(specific_energy_j_kg) < eps,
        np.where(specific_energy_j_kg < 0.0, -eps, eps),
        specific_energy_j_kg,
    )
    semimajor_axis_m = -central_body.mu_m3_s2 / (2.0 * specific_energy_safe_j_kg)

    angular_momentum = np.cross(positions_m, velocities_m_s)
    angular_momentum_norm = np.linalg.norm(angular_momentum, axis=1)
    angular_momentum_norm_safe = np.maximum(angular_momentum_norm, eps)

    k_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    node_vectors = np.cross(np.tile(k_hat, (len(states_m_s), 1)), angular_momentum)
    node_norm = np.linalg.norm(node_vectors, axis=1)

    eccentricity_vectors = (
        np.cross(velocities_m_s, angular_momentum) / central_body.mu_m3_s2
        - positions_m / radius_safe_m[:, None]
    )
    eccentricity = np.linalg.norm(eccentricity_vectors, axis=1)

    inclination_rad = np.arccos(
        np.clip(angular_momentum[:, 2] / angular_momentum_norm_safe, -1.0, 1.0)
    )
    inclination_deg = np.degrees(inclination_rad)

    raan_rad = np.arctan2(node_vectors[:, 1], node_vectors[:, 0])
    raan_deg = np.degrees(np.unwrap(raan_rad))

    return {
        "radius_m": radius_m,
        "altitude_m": altitude_m,
        "speed_m_s": speed_m_s,
        "specific_energy_j_kg": specific_energy_j_kg,
        "semimajor_axis_m": semimajor_axis_m,
        "eccentricity": eccentricity,
        "inclination_deg": inclination_deg,
        "raan_deg": raan_deg,
        "angular_momentum_m2_s": angular_momentum_norm,
        "node_vector_norm_m2_s": node_norm,
    }
