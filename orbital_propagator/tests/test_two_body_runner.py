from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from orbital_propagator.bodies.earth import EARTH
from orbital_propagator.config import (
    ForceModelConfig,
    IntegratorConfig,
    PropagationConfig,
    SimulationRequest,
    SpacecraftConfig,
    circular_orbit_state,
    keplerian_orbit_state,
)
from orbital_propagator.forces.drag import PYMSIS_AVAILABLE
from orbital_propagator.propagation.dynamics import evaluate_accelerations
from orbital_propagator.io.artifacts import build_run_artifact, save_run_artifact
from orbital_propagator.propagation.orbital_elements import compute_derived_series
from orbital_propagator.propagation.integrators import integrate_states
from orbital_propagator.propagation.runner import run_simulation


def classical_elements_from_state(
    state_m_s: np.ndarray,
    mu_m3_s2: float,
) -> dict[str, float]:
    position_m = state_m_s[:3]
    velocity_m_s = state_m_s[3:]
    radius_m = float(np.linalg.norm(position_m))
    speed_sq_m2_s2 = float(np.dot(velocity_m_s, velocity_m_s))
    specific_energy_j_kg = 0.5 * speed_sq_m2_s2 - mu_m3_s2 / radius_m
    semimajor_axis_m = -mu_m3_s2 / (2.0 * specific_energy_j_kg)

    angular_momentum = np.cross(position_m, velocity_m_s)
    angular_momentum_norm = float(np.linalg.norm(angular_momentum))
    node_vector = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), angular_momentum)
    node_norm = float(np.linalg.norm(node_vector))
    eccentricity_vector = (
        np.cross(velocity_m_s, angular_momentum) / mu_m3_s2
        - position_m / radius_m
    )
    eccentricity = float(np.linalg.norm(eccentricity_vector))

    inclination_rad = float(np.arccos(angular_momentum[2] / angular_momentum_norm))
    raan_rad = float(np.arctan2(node_vector[1], node_vector[0]))
    if raan_rad < 0.0:
        raan_rad += 2.0 * np.pi

    argument_of_periapsis_rad = float(
        np.arctan2(
            np.dot(np.cross(node_vector, eccentricity_vector), angular_momentum)
            / (node_norm * eccentricity * angular_momentum_norm),
            np.dot(node_vector, eccentricity_vector) / (node_norm * eccentricity),
        )
    )
    if argument_of_periapsis_rad < 0.0:
        argument_of_periapsis_rad += 2.0 * np.pi

    true_anomaly_rad = float(
        np.arctan2(
            np.dot(np.cross(eccentricity_vector, position_m), angular_momentum)
            / (eccentricity * radius_m * angular_momentum_norm),
            np.dot(eccentricity_vector, position_m) / (eccentricity * radius_m),
        )
    )
    if true_anomaly_rad < 0.0:
        true_anomaly_rad += 2.0 * np.pi

    return {
        "semimajor_axis_m": semimajor_axis_m,
        "eccentricity": eccentricity,
        "inclination_deg": float(np.degrees(inclination_rad)),
        "raan_deg": float(np.degrees(raan_rad)),
        "argument_of_periapsis_deg": float(np.degrees(argument_of_periapsis_rad)),
        "true_anomaly_deg": float(np.degrees(true_anomaly_rad)),
    }


class TwoBodyRunnerTests(unittest.TestCase):
    def test_scipy_dop853_uses_optional_maximum_step(self) -> None:
        times_s = np.array([0.0, 10.0])
        initial_state = np.array([1.0, 0.0])
        solution = SimpleNamespace(success=True, y=np.column_stack((initial_state, initial_state)))

        with patch("orbital_propagator.propagation.integrators.solve_ivp", return_value=solution) as solver:
            states = integrate_states(
                lambda _time_s, state: state,
                initial_state,
                times_s,
                IntegratorConfig(backend="scipy", method="DOP853", max_step_s=2.5),
            )

        self.assertEqual(states.shape, (2, 2))
        self.assertEqual(solver.call_args.kwargs["method"], "DOP853")
        self.assertEqual(solver.call_args.kwargs["max_step"], 2.5)

    def test_scipy_rejects_non_positive_maximum_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_step_s"):
            integrate_states(
                lambda _time_s, state: state,
                np.array([1.0, 0.0]),
                np.array([0.0, 10.0]),
                IntegratorConfig(backend="scipy", max_step_s=0.0),
            )

    def test_force_magnitudes_match_expected_orders_in_leo(self) -> None:
        state = circular_orbit_state(EARTH, altitude_m=500_000.0, inclination_deg=63.4)
        spacecraft = SpacecraftConfig(
            mass_kg=1000.0,
            cross_section_area_m2=10.0,
            drag_coefficient=2.2,
            reflectivity_coefficient=1.2,
        )
        accelerations = evaluate_accelerations(
            elapsed_time_s=0.0,
            state_m_s=state,
            central_body=EARTH,
            spacecraft=spacecraft,
            forces=ForceModelConfig(
                j2=True,
                drag=True,
                solar_radiation_pressure=True,
                third_body_sun=True,
                third_body_moon=True,
            ),
            start_epoch_utc="2026-04-23T12:00:00Z",
        )
        norms = {name: float(np.linalg.norm(value)) for name, value in accelerations.items()}

        self.assertGreater(norms["central_gravity"], 8.0)
        self.assertLess(norms["central_gravity"], 9.0)
        self.assertGreater(norms["j2"], 1e-3)
        self.assertLess(norms["j2"], 2e-2)
        self.assertGreater(norms["drag"], 1e-8)
        self.assertLess(norms["drag"], 1e-5)
        self.assertGreater(norms["third_body_sun"], 1e-7)
        self.assertLess(norms["third_body_sun"], 1e-6)
        self.assertGreater(norms["third_body_moon"], 5e-7)
        self.assertLess(norms["third_body_moon"], 2e-6)
        self.assertGreater(norms["solar_radiation_pressure"], 1e-8)
        self.assertLess(norms["solar_radiation_pressure"], 1e-7)

    def test_rk4_two_body_run_keeps_specific_energy_nearly_constant(self) -> None:
        altitude_m = 500_000.0
        radius_m = EARTH.radius_m + altitude_m
        orbital_period_s = 2.0 * np.pi * np.sqrt(radius_m**3 / EARTH.mu_m3_s2)
        request = SimulationRequest(
            run_name="energy_check",
            producer="simulation",
            central_body=EARTH,
            initial_state_m_s=circular_orbit_state(
                central_body=EARTH,
                altitude_m=altitude_m,
                inclination_deg=28.5,
            ),
            propagation=PropagationConfig(
                duration_s=orbital_period_s,
                sample_count=721,
            ),
            integrator=IntegratorConfig(backend="rk4"),
            spacecraft=SpacecraftConfig(),
            forces=ForceModelConfig(),
        )

        result = run_simulation(request)
        specific_energy = result.derived_series["specific_energy_j_kg"]
        energy_span = float(np.max(specific_energy) - np.min(specific_energy))

        self.assertLess(energy_span, 5_000.0)

    def test_state_vector_artifact_is_the_default_output(self) -> None:
        request = SimulationRequest(
            run_name="artifact_check",
            producer="simulation",
            central_body=EARTH,
            initial_state_m_s=circular_orbit_state(EARTH, altitude_m=400_000.0),
            propagation=PropagationConfig(duration_s=600.0, sample_count=21),
            integrator=IntegratorConfig(backend="rk4"),
            spacecraft=SpacecraftConfig(),
            forces=ForceModelConfig(),
        )
        result = run_simulation(request)
        artifact = build_run_artifact(request, result)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "artifact.json"
            save_run_artifact(artifact, output_path)

            with output_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)

        self.assertEqual(loaded["run_name"], "artifact_check")
        self.assertEqual(loaded["central_body"], "Earth")
        self.assertEqual(loaded["output_type"], "state_vector")
        self.assertIn("states_m_s", loaded)
        self.assertIn("derived_series", loaded)
        self.assertNotIn("accelerations_total_m_s2", loaded)
        self.assertNotIn("accelerations_by_force_m_s2", loaded)
        self.assertEqual(len(loaded["times_s"]), 21)

    def test_force_breakdown_artifact_contains_total_and_named_accelerations(self) -> None:
        request = SimulationRequest(
            run_name="force_breakdown_check",
            producer="simulation",
            central_body=EARTH,
            initial_state_m_s=circular_orbit_state(EARTH, altitude_m=400_000.0),
            propagation=PropagationConfig(duration_s=600.0, sample_count=21),
            integrator=IntegratorConfig(backend="rk4"),
            spacecraft=SpacecraftConfig(),
            forces=ForceModelConfig(j2=True),
        )
        result = run_simulation(request)
        artifact = build_run_artifact(request, result, force_breakdown=True)

        self.assertEqual(artifact["output_type"], "force_breakdown")
        self.assertIn("states_m_s", artifact)
        self.assertIn("accelerations_total_m_s2", artifact)
        self.assertIn("accelerations_by_force_m_s2", artifact)
        self.assertEqual(
            set(artifact["accelerations_by_force_m_s2"]),
            {"central_gravity", "j2"},
        )
        self.assertEqual(len(artifact["accelerations_total_m_s2"]), 21)

    def test_j2_run_changes_raan_for_inclined_orbit(self) -> None:
        altitude_m = 500_000.0
        radius_m = EARTH.radius_m + altitude_m
        orbital_period_s = 2.0 * np.pi * np.sqrt(radius_m**3 / EARTH.mu_m3_s2)
        request = SimulationRequest(
            run_name="j2_raan_check",
            producer="simulation",
            central_body=EARTH,
            initial_state_m_s=circular_orbit_state(
                central_body=EARTH,
                altitude_m=altitude_m,
                inclination_deg=63.4,
            ),
            propagation=PropagationConfig(
                duration_s=3.0 * orbital_period_s,
                sample_count=721,
            ),
            integrator=IntegratorConfig(backend="rk4"),
            spacecraft=SpacecraftConfig(),
            forces=ForceModelConfig(j2=True),
        )

        result = run_simulation(request)
        raan_deg = result.derived_series["raan_deg"]

        self.assertIn("j2", result.accelerations_by_force_m_s2)
        self.assertGreater(abs(float(raan_deg[-1] - raan_deg[0])), 0.01)

    def test_j2_raan_drift_matches_classical_rate_reasonably(self) -> None:
        altitude_m = 700_000.0
        inclination_deg = 63.4
        radius_m = EARTH.radius_m + altitude_m
        orbital_period_s = 2.0 * np.pi * np.sqrt(radius_m**3 / EARTH.mu_m3_s2)
        duration_s = 5.0 * orbital_period_s
        request = SimulationRequest(
            run_name="j2_rate_check",
            producer="simulation",
            central_body=EARTH,
            initial_state_m_s=circular_orbit_state(
                central_body=EARTH,
                altitude_m=altitude_m,
                inclination_deg=inclination_deg,
            ),
            propagation=PropagationConfig(
                duration_s=duration_s,
                sample_count=1201,
            ),
            integrator=IntegratorConfig(backend="rk4"),
            spacecraft=SpacecraftConfig(),
            forces=ForceModelConfig(j2=True),
        )

        result = run_simulation(request)
        observed_raan_rate_rad_s = np.deg2rad(
            result.derived_series["raan_deg"][-1] - result.derived_series["raan_deg"][0]
        ) / duration_s
        mean_motion_rad_s = np.sqrt(EARTH.mu_m3_s2 / radius_m**3)
        expected_raan_rate_rad_s = (
            -1.5
            * EARTH.j2
            * mean_motion_rad_s
            * (EARTH.radius_m / radius_m) ** 2
            * np.cos(np.deg2rad(inclination_deg))
        )

        relative_error = abs(
            (observed_raan_rate_rad_s - expected_raan_rate_rad_s) / expected_raan_rate_rad_s
        )
        self.assertLess(relative_error, 0.2)

    def test_drag_run_lowers_specific_energy(self) -> None:
        request = SimulationRequest(
            run_name="drag_energy_check",
            producer="simulation",
            central_body=EARTH,
            initial_state_m_s=circular_orbit_state(
                central_body=EARTH,
                altitude_m=120_000.0,
                inclination_deg=28.5,
            ),
            propagation=PropagationConfig(
                duration_s=300.0,
                sample_count=1201,
            ),
            integrator=IntegratorConfig(backend="rk4"),
            spacecraft=SpacecraftConfig(
                mass_kg=200.0,
                cross_section_area_m2=15.0,
                drag_coefficient=2.5,
            ),
            forces=ForceModelConfig(drag=True),
        )

        result = run_simulation(request)
        specific_energy = result.derived_series["specific_energy_j_kg"]
        self.assertIn("drag", result.accelerations_by_force_m_s2)
        self.assertLess(specific_energy[-1], specific_energy[0])

    def test_drag_corotation_toggle_changes_relative_drag_strength(self) -> None:
        state = circular_orbit_state(EARTH, altitude_m=250_000.0, inclination_deg=0.0)
        spacecraft = SpacecraftConfig(
            mass_kg=500.0,
            cross_section_area_m2=12.0,
            drag_coefficient=2.2,
        )

        corotating = evaluate_accelerations(
            elapsed_time_s=0.0,
            state_m_s=state,
            central_body=EARTH,
            spacecraft=spacecraft,
            forces=ForceModelConfig(drag=True, corotating_atmosphere=True),
            start_epoch_utc="2026-04-23T12:00:00Z",
        )["drag"]
        non_corotating = evaluate_accelerations(
            elapsed_time_s=0.0,
            state_m_s=state,
            central_body=EARTH,
            spacecraft=spacecraft,
            forces=ForceModelConfig(drag=True, corotating_atmosphere=False),
            start_epoch_utc="2026-04-23T12:00:00Z",
        )["drag"]

        self.assertLess(np.linalg.norm(corotating), np.linalg.norm(non_corotating))

    def test_keplerian_initialization_reproduces_requested_elements(self) -> None:
        state = keplerian_orbit_state(
            central_body=EARTH,
            semimajor_axis_m=10_500_000.0,
            eccentricity=0.35,
            inclination_deg=41.0,
            raan_deg=23.0,
            argument_of_periapsis_deg=87.0,
            true_anomaly_deg=41.0,
        )

        elements = classical_elements_from_state(state, EARTH.mu_m3_s2)

        self.assertAlmostEqual(elements["semimajor_axis_m"], 10_500_000.0, places=6)
        self.assertAlmostEqual(elements["eccentricity"], 0.35, places=12)
        self.assertAlmostEqual(elements["inclination_deg"], 41.0, places=12)
        self.assertAlmostEqual(elements["raan_deg"], 23.0, places=12)
        self.assertAlmostEqual(elements["argument_of_periapsis_deg"], 87.0, places=10)
        self.assertAlmostEqual(elements["true_anomaly_deg"], 41.0, places=10)

    def test_keplerian_initialization_rejects_orbit_inside_earth(self) -> None:
        with self.assertRaisesRegex(ValueError, "intersects the central body"):
            keplerian_orbit_state(
                central_body=EARTH,
                semimajor_axis_m=7_000_000.0,
                eccentricity=0.2,
                inclination_deg=10.0,
                raan_deg=0.0,
                argument_of_periapsis_deg=0.0,
                true_anomaly_deg=0.0,
            )

    def test_derived_series_reports_argument_of_periapsis_for_elliptical_orbit(self) -> None:
        state = keplerian_orbit_state(
            central_body=EARTH,
            semimajor_axis_m=10_500_000.0,
            eccentricity=0.25,
            inclination_deg=37.0,
            raan_deg=18.0,
            argument_of_periapsis_deg=73.0,
            true_anomaly_deg=12.0,
        )
        derived = compute_derived_series(np.array([state]), EARTH)
        argument_of_periapsis_deg = float(derived["argument_of_periapsis_deg"][0])

        self.assertAlmostEqual(argument_of_periapsis_deg, 73.0, places=10)

    def test_pymsis_drag_selection_fails_cleanly_when_dependency_is_missing(self) -> None:
        if PYMSIS_AVAILABLE:
            self.skipTest("pymsis is installed in this environment")

        request = SimulationRequest(
            run_name="pymsis_missing_check",
            producer="simulation",
            central_body=EARTH,
            initial_state_m_s=circular_orbit_state(
                central_body=EARTH,
                altitude_m=400_000.0,
                inclination_deg=28.5,
            ),
            propagation=PropagationConfig(
                duration_s=60.0,
                sample_count=11,
            ),
            integrator=IntegratorConfig(backend="rk4"),
            spacecraft=SpacecraftConfig(),
            forces=ForceModelConfig(drag=True, atmosphere_model="pymsis"),
        )

        with self.assertRaisesRegex(RuntimeError, "pymsis"):
            run_simulation(request)

    def test_phase2_force_run_records_force_components(self) -> None:
        request = SimulationRequest(
            run_name="phase2_force_check",
            producer="simulation",
            central_body=EARTH,
            initial_state_m_s=circular_orbit_state(
                central_body=EARTH,
                altitude_m=700_000.0,
                inclination_deg=45.0,
            ),
            propagation=PropagationConfig(
                duration_s=3600.0,
                sample_count=121,
            ),
            integrator=IntegratorConfig(backend="rk4"),
            spacecraft=SpacecraftConfig(
                mass_kg=600.0,
                cross_section_area_m2=12.0,
                reflectivity_coefficient=1.4,
            ),
            forces=ForceModelConfig(
                third_body_sun=True,
                third_body_moon=True,
                solar_radiation_pressure=True,
            ),
        )

        result = run_simulation(request)

        self.assertIn("third_body_sun", result.accelerations_by_force_m_s2)
        self.assertIn("third_body_moon", result.accelerations_by_force_m_s2)
        self.assertIn("solar_radiation_pressure", result.accelerations_by_force_m_s2)
        self.assertIn("reference_vector_tracks_m", result.metadata)
        self.assertIn("sun", result.metadata["reference_vector_tracks_m"])
        self.assertIn("moon", result.metadata["reference_vector_tracks_m"])
        self.assertGreater(len(result.metadata["reference_vector_tracks_m"]["sun"]["times_s"]), 1)
        self.assertEqual(
            result.metadata["reference_vector_tracks_m"]["sun"]["vectors_m"].shape[1],
            3,
        )


if __name__ == "__main__":
    unittest.main()
