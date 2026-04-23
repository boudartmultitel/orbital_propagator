from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from orbital_propagator.forces.gravity import central_gravity_acceleration
from orbital_propagator.forces.j2 import j2_acceleration
from orbital_propagator.forces.srp import solar_radiation_pressure_acceleration
from orbital_propagator.forces.third_body import third_body_point_mass_acceleration


class GravityForceTests(unittest.TestCase):
    def test_central_gravity_points_toward_origin_with_expected_magnitude(self) -> None:
        mu_m3_s2 = 3.986004418e14
        position_m = np.array([7_000_000.0, 0.0, 0.0])

        acceleration = central_gravity_acceleration(position_m, mu_m3_s2)

        expected_magnitude = mu_m3_s2 / np.linalg.norm(position_m) ** 2
        self.assertAlmostEqual(acceleration[0], -expected_magnitude, places=12)
        self.assertAlmostEqual(acceleration[1], 0.0, places=12)
        self.assertAlmostEqual(acceleration[2], 0.0, places=12)

    def test_j2_has_expected_direction_on_equatorial_x_axis(self) -> None:
        acceleration = j2_acceleration(
            position_m=np.array([7_000_000.0, 0.0, 0.0]),
            mu_m3_s2=3.986004418e14,
            equatorial_radius_m=6_378_136.3,
            j2=1.08262668e-3,
        )

        self.assertLess(acceleration[0], 0.0)
        self.assertAlmostEqual(acceleration[1], 0.0, places=12)
        self.assertAlmostEqual(acceleration[2], 0.0, places=12)

    def test_third_body_acceleration_is_zero_at_central_body_origin(self) -> None:
        acceleration = third_body_point_mass_acceleration(
            spacecraft_position_m=np.zeros(3),
            third_body_position_m=np.array([1.0e11, 0.0, 0.0]),
            third_body_mu_m3_s2=1.32712440018e20,
        )
        self.assertTrue(np.allclose(acceleration, np.zeros(3)))

    def test_srp_points_away_from_sun(self) -> None:
        acceleration = solar_radiation_pressure_acceleration(
            spacecraft_position_m=np.zeros(3),
            sun_position_m=np.array([1.0, 0.0, 0.0]) * 149_597_870_700.0,
            reflectivity_coefficient=1.2,
            cross_section_area_m2=20.0,
            mass_kg=1000.0,
            solar_pressure_1au_n_m2=4.56e-6,
            astronomical_unit_m=149_597_870_700.0,
        )
        self.assertLess(acceleration[0], 0.0)
        self.assertAlmostEqual(acceleration[1], 0.0, places=20)
        self.assertAlmostEqual(acceleration[2], 0.0, places=20)


if __name__ == "__main__":
    unittest.main()
