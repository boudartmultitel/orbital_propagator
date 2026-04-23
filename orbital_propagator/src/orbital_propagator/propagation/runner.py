from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from orbital_propagator.ephemerides.provider import ephemeris_source_name, body_position_m
from orbital_propagator.config import SimulationRequest
from orbital_propagator.propagation.dynamics import evaluate_accelerations, state_derivative
from orbital_propagator.propagation.integrators import integrate_states
from orbital_propagator.propagation.orbital_elements import compute_derived_series


@dataclass
class SimulationResult:
    times_s: np.ndarray
    states_m_s: np.ndarray
    accelerations_by_force_m_s2: dict[str, np.ndarray]
    derived_series: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def accelerations_total_m_s2(self) -> np.ndarray:
        return sum(self.accelerations_by_force_m_s2.values())


def run_simulation(request: SimulationRequest) -> SimulationResult:
    times_s = np.linspace(
        0.0,
        request.propagation.duration_s,
        num=request.propagation.sample_count,
        dtype=float,
    )
    derivative = lambda time_s, state_m_s: state_derivative(
        time_s,
        state_m_s,
        request.central_body,
        request.spacecraft,
        request.forces,
        request.propagation.start_epoch_utc,
    )
    states_m_s = integrate_states(
        derivative,
        request.initial_state_m_s,
        times_s,
        request.integrator,
    )

    force_names = tuple(enabled_force_names(request))
    accelerations_by_force = {
        force_name: np.zeros((len(times_s), 3), dtype=float) for force_name in force_names
    }

    for index, (time_s, state) in enumerate(zip(times_s, states_m_s, strict=False)):
        accelerations = evaluate_accelerations(
            time_s,
            state,
            request.central_body,
            request.spacecraft,
            request.forces,
            request.propagation.start_epoch_utc,
        )
        for force_name, acceleration in accelerations.items():
            accelerations_by_force[force_name][index] = acceleration

    derived_series = compute_derived_series(states_m_s, request.central_body)
    reference_vectors_m: dict[str, np.ndarray] = {}
    if request.forces.third_body_sun or request.forces.solar_radiation_pressure:
        reference_vectors_m["sun"] = body_position_m(
            "sun",
            request.propagation.start_epoch_utc,
            request.propagation.start_time_s,
        )
    if request.forces.third_body_moon:
        reference_vectors_m["moon"] = body_position_m(
            "moon",
            request.propagation.start_epoch_utc,
            request.propagation.start_time_s,
        )

    return SimulationResult(
        times_s=times_s,
        states_m_s=states_m_s,
        accelerations_by_force_m_s2=accelerations_by_force,
        derived_series=derived_series,
        metadata={
            "ephemeris_source": ephemeris_source_name(),
            "reference_vectors_m": reference_vectors_m,
        },
    )


def enabled_force_names(request: SimulationRequest) -> list[str]:
    names: list[str] = []
    if request.forces.central_gravity:
        names.append("central_gravity")
    if request.forces.j2:
        names.append("j2")
    if request.forces.drag:
        names.append("drag")
    if request.forces.third_body_sun:
        names.append("third_body_sun")
    if request.forces.third_body_moon:
        names.append("third_body_moon")
    if request.forces.solar_radiation_pressure:
        names.append("solar_radiation_pressure")
    return names
