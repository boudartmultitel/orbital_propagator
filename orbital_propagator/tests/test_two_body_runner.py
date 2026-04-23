from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

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
)
from orbital_propagator.forces.drag import PYMSIS_AVAILABLE
from orbital_propagator.propagation.dynamics import evaluate_accelerations
from orbital_propagator.io.artifacts import build_run_artifact, save_run_artifact
from orbital_propagator.propagation.runner import run_simulation


class TwoBodyRunnerTests(unittest.TestCase):
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

    def test_artifact_save_round_trip_contains_expected_keys(self) -> None:
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
        self.assertIn("states_m_s", loaded)
        self.assertIn("derived_series", loaded)
        self.assertEqual(len(loaded["times_s"]), 21)

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


if __name__ == "__main__":
    unittest.main()
