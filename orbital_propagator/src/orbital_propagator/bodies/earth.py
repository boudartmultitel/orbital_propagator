from __future__ import annotations

from orbital_propagator.config import CentralBodyConfig


EARTH = CentralBodyConfig(
    name="Earth",
    mu_m3_s2=3.986004418e14,
    radius_m=6_378_136.3,
    j2=1.08262668e-3,
    rotation_rate_rad_s=7.2921159e-5,
    atmosphere_density_sea_level_kg_m3=1.225,
    atmosphere_scale_height_m=8_500.0,
)
