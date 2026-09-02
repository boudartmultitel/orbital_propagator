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

from orbital_propagator.generation.manifest import (
    append_sampled_trajectories,
    build_manifest_dataset,
    load_manifest,
    simulation_request_from_record,
)
from orbital_propagator.generation.features import materialize_input_vectors


class TrajectoryManifestTests(unittest.TestCase):
    def test_append_writes_one_complete_trajectory_per_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "trajectories.jsonl"
            appended = append_sampled_trajectories(
                manifest_path,
                "multi_planet_two_body",
                2,
                np.random.default_rng(42),
                random_seed=42,
            )
            records = load_manifest(manifest_path)
            line_count = len(manifest_path.read_text().splitlines())

        self.assertEqual(len(appended), 2)
        self.assertEqual(records, appended)
        self.assertEqual(line_count, 2)
        self.assertTrue(
            {
                "trajectory_id",
                "propagation",
                "integrator",
                "mass_kg",
                "cross_section_area_m2",
                "C_D",
                "C_R",
                "A_over_m",
                "gamma_D",
                "gamma_R",
            }.issubset(records[0])
        )
        self.assertEqual(records[0]["sampling_seed"], 42)

    def test_manifest_record_builds_existing_simulation_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "trajectories.jsonl"
            record = append_sampled_trajectories(
                manifest_path,
                "multi_planet_two_body",
                1,
                np.random.default_rng(3),
                integrator_backend="rk4",
            )[0]

            request = simulation_request_from_record(record)

        self.assertEqual(request.run_name, record["trajectory_id"])
        self.assertEqual(request.central_body.name.lower(), record["central_body_name"])
        self.assertAlmostEqual(
            request.spacecraft.cross_section_area_m2 / request.spacecraft.mass_kg,
            record["A_over_m"],
        )

    def test_build_executes_manifest_and_writes_dataset_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "trajectories.jsonl"
            output_directory = root / "dataset"
            append_sampled_trajectories(
                manifest_path,
                "multi_planet_two_body",
                1,
                np.random.default_rng(5),
                duration_s=10.0,
                sample_count=3,
                integrator_backend="rk4",
            )

            written = build_manifest_dataset(manifest_path, output_directory)
            metadata = json.loads((output_directory / "metadata.json").read_text())
            trajectory = json.loads(written[0].read_text())
            resumed = build_manifest_dataset(
                manifest_path, output_directory, skip_existing=True
            )

        self.assertEqual(len(written), 1)
        self.assertEqual(resumed, [])
        self.assertEqual(metadata["trajectory_count"], 1)
        self.assertTrue(metadata["force_breakdown"])
        self.assertEqual(trajectory["schema_version"], "0.7.0")
        self.assertEqual(trajectory["run_id"], trajectory["run_name"])
        self.assertEqual(len(trajectory["feature_names"]), 33)
        self.assertNotIn("inputs", trajectory)
        self.assertEqual(len(trajectory["constant_inputs"]), 11)
        self.assertEqual(materialize_input_vectors(trajectory).shape, (3, 33))


if __name__ == "__main__":
    unittest.main()
