from __future__ import annotations

import numpy as np

from orbital_propagator.ephemerides.approximate import AU_M, approximate_body_position_m

try:
    from astropy import units as u
    from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
    from astropy.time import Time

    ASTROPY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local environment
    ASTROPY_AVAILABLE = False


def ephemeris_source_name() -> str:
    return "astropy_builtin" if ASTROPY_AVAILABLE else "approximate_fallback"


def body_position_m(
    body: str,
    start_epoch_utc: str,
    elapsed_s: float,
) -> np.ndarray:
    if ASTROPY_AVAILABLE:
        epoch = Time(start_epoch_utc, scale="utc") + float(elapsed_s) * u.s
        with solar_system_ephemeris.set("builtin"):
            body_barycentric = get_body_barycentric(body, epoch)
            earth_barycentric = get_body_barycentric("earth", epoch)
        relative = body_barycentric - earth_barycentric
        return np.array(
            [
                relative.x.to_value(u.m),
                relative.y.to_value(u.m),
                relative.z.to_value(u.m),
            ],
            dtype=float,
        )

    return approximate_body_position_m(body, elapsed_s)


def sun_position_for_central_body_m(
    start_epoch_utc: str,
    elapsed_s: float,
    heliocentric_distance_au: float | None,
) -> np.ndarray:
    """Approximate a planet-to-Sun vector at the catalog mean distance."""
    if heliocentric_distance_au is None:
        raise ValueError("Heliocentric distance is required for the Sun position.")
    earth_to_sun_m = body_position_m("sun", start_epoch_utc, elapsed_s)
    if heliocentric_distance_au == 1.0:
        return earth_to_sun_m
    return (
        earth_to_sun_m
        / np.linalg.norm(earth_to_sun_m)
        * heliocentric_distance_au
        * AU_M
    )
