from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from orbital_propagator.bodies.catalog import (
    ALLOWED_CENTRAL_BODY_NAMES,
    central_body_from_catalog,
    load_planet_catalog,
)
from orbital_propagator.bodies.earth import EARTH
from orbital_propagator.config import (
    ForceModelConfig,
    validate_force_model,
)
from orbital_propagator.generation.configuration import (
    load_data_generation_config,
    load_dataset_recipe,
)
from orbital_propagator.generation.sampling import sample_generation_parameters


class DataGenerationConfigTests(unittest.TestCase):
    def test_unified_config_contains_parameter_categories_and_recipes(self) -> None:
        config = load_data_generation_config()

        self.assertIn("fixed_parameters", config)
        self.assertIn("sampled_parameters", config)
        self.assertIn("derived_environment", config)
        self.assertEqual(
            set(config["dataset_recipes"]),
            {
                "multi_planet_two_body",
                "multi_planet_j2",
                "earth_full_perturbations",
            },
        )

    def test_phase_6_recipes_are_individually_loadable(self) -> None:
        recipe = load_dataset_recipe("multi_planet_two_body")

        self.assertEqual(recipe["dataset_name"], "multi_planet_two_body")
        self.assertTrue(recipe["forces"]["two_body"])
        self.assertFalse(recipe["forces"]["third_body"])

    def test_orbit_families_and_spacecraft_priors_cover_phases_3_and_4(self) -> None:
        sampled = load_data_generation_config()["sampled_parameters"]
        families = sampled["orbit_families"]

        self.assertEqual(
            set(families["generic"]),
            {"low_orbit", "high_orbit", "highly_elliptical"},
        )
        self.assertEqual(
            set(families["earth_specific"]), {"leo", "sso", "geo", "molniya"}
        )
        self.assertTrue(
            all(
                family["central_body"] == "earth"
                for family in families["earth_specific"].values()
            )
        )
        self.assertEqual(
            set(sampled["spacecraft_priors"]),
            {"drag_coefficient", "reflectivity_coefficient", "area_to_mass"},
        )

    def test_catalog_contains_only_allowed_planetary_central_bodies(self) -> None:
        catalog = load_planet_catalog()

        self.assertEqual(set(catalog), set(ALLOWED_CENTRAL_BODY_NAMES))
        self.assertNotIn("moon", catalog)

    def test_moon_remains_available_as_an_earth_third_body(self) -> None:
        config = load_data_generation_config()
        moon = config["fixed_parameters"]["third_bodies"]["moon"]

        self.assertEqual(moon["enabled_for_central_bodies"], ["earth"])
        self.assertTrue(
            config["dataset_recipes"]["earth_full_perturbations"]["third_bodies"][
                "moon"
            ]
        )

    def test_earth_catalog_entry_builds_existing_central_body_config(self) -> None:
        earth = central_body_from_catalog("earth")

        self.assertEqual(earth, EARTH)
        self.assertEqual(earth.atmosphere_model, "earth")
        self.assertEqual(earth.heliocentric_distance_au, 1.0)

    def test_all_planet_catalog_entries_build_central_body_configs(self) -> None:
        for body_name in ALLOWED_CENTRAL_BODY_NAMES:
            body = central_body_from_catalog(body_name)
            self.assertGreater(body.mu_m3_s2, 0.0)
            self.assertGreater(body.radius_m, 0.0)

        self.assertIsNone(central_body_from_catalog("venus").j2)

    def test_unsupported_force_and_body_combinations_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "J2 is unavailable"):
            validate_force_model(
                central_body_from_catalog("venus"), ForceModelConfig(j2=True)
            )
        with self.assertRaisesRegex(ValueError, "drag is unavailable"):
            validate_force_model(
                central_body_from_catalog("mars"), ForceModelConfig(drag=True)
            )
        with self.assertRaisesRegex(ValueError, "only available for Earth"):
            validate_force_model(
                central_body_from_catalog("mars"),
                ForceModelConfig(third_body_moon=True),
            )

    def test_phase_7_sampling_is_reproducible_and_includes_spacecraft_inputs(self) -> None:
        first = sample_generation_parameters(
            "multi_planet_two_body", np.random.default_rng(42)
        )
        second = sample_generation_parameters(
            "multi_planet_two_body", np.random.default_rng(42)
        )

        self.assertEqual(first, second)
        self.assertTrue(
            {
                "C_D",
                "C_R",
                "A_over_m",
                "gamma_D",
                "gamma_R",
                "area_to_mass_bin",
            }.issubset(first)
        )
        self.assertAlmostEqual(first["gamma_D"], first["C_D"] * first["A_over_m"])
        self.assertAlmostEqual(first["gamma_R"], first["C_R"] * first["A_over_m"])
        self.assertGreater(first["a_km"] * (1.0 - first["e"]), first["radius_km"])

    def test_earth_full_recipe_enables_sun_and_moon_for_earth(self) -> None:
        sample = sample_generation_parameters(
            "earth_full_perturbations", np.random.default_rng(8)
        )

        self.assertEqual(sample["central_body_name"], "earth")
        self.assertEqual(sample["third_bodies_enabled"], ["sun", "moon"])

    def test_j2_recipe_fails_clearly_when_venus_is_selected(self) -> None:
        with patch(
            "orbital_propagator.generation.sampling.sample_central_body",
            return_value="venus",
        ):
            with self.assertRaisesRegex(ValueError, "J2 is unavailable.*Venus"):
                sample_generation_parameters(
                    "multi_planet_j2", np.random.default_rng(42)
                )


if __name__ == "__main__":
    unittest.main()
