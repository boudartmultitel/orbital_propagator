from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import asin, atan2, degrees

import numpy as np

try:
    from astropy import units as u
    from astropy.coordinates import CartesianRepresentation, EarthLocation, GCRS, ITRS
    from astropy.time import Time

    ASTROPY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local environment
    ASTROPY_AVAILABLE = False

try:
    from pymsis import Variable, calculate as pymsis_calculate

    PYMSIS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local environment
    PYMSIS_AVAILABLE = False


ATMOSPHERE_LAYERS = (
    (0.0, 1.225, 7_249.0),
    (25_000.0, 3.899e-2, 6_349.0),
    (30_000.0, 1.774e-2, 6_682.0),
    (40_000.0, 3.972e-3, 7_554.0),
    (50_000.0, 1.057e-3, 8_382.0),
    (60_000.0, 3.206e-4, 7_714.0),
    (70_000.0, 8.770e-5, 6_549.0),
    (80_000.0, 1.905e-5, 5_799.0),
    (90_000.0, 3.396e-6, 5_382.0),
    (100_000.0, 5.297e-7, 5_877.0),
    (110_000.0, 9.661e-8, 7_263.0),
    (120_000.0, 2.438e-8, 9_473.0),
    (130_000.0, 8.484e-9, 12_636.0),
    (140_000.0, 3.845e-9, 16_149.0),
    (150_000.0, 2.070e-9, 22_523.0),
    (180_000.0, 5.464e-10, 29_740.0),
    (200_000.0, 2.789e-10, 37_105.0),
    (250_000.0, 7.248e-11, 45_546.0),
    (300_000.0, 2.418e-11, 53_628.0),
    (350_000.0, 9.518e-12, 53_298.0),
    (400_000.0, 3.725e-12, 58_515.0),
    (450_000.0, 1.585e-12, 60_828.0),
    (500_000.0, 6.967e-13, 63_822.0),
    (600_000.0, 1.454e-13, 71_835.0),
    (700_000.0, 3.614e-14, 88_667.0),
    (800_000.0, 1.170e-14, 124_640.0),
    (900_000.0, 5.245e-15, 181_050.0),
    (1_000_000.0, 3.019e-15, 268_000.0),
)


def _utc_datetime_at_elapsed_seconds(start_epoch_utc: str, elapsed_time_s: float) -> datetime:
    normalized = start_epoch_utc.replace("Z", "+00:00")
    epoch = datetime.fromisoformat(normalized)
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    return (epoch.astimezone(timezone.utc) + timedelta(seconds=float(elapsed_time_s))).replace(
        tzinfo=None
    )


def _approximate_lon_lat_alt_km(
    position_m: np.ndarray,
    elapsed_time_s: float,
    body_radius_m: float,
    body_rotation_rate_rad_s: float,
) -> tuple[float, float, float]:
    rotation_angle_rad = body_rotation_rate_rad_s * float(elapsed_time_s)
    cos_angle = float(np.cos(rotation_angle_rad))
    sin_angle = float(np.sin(rotation_angle_rad))
    rotation = np.array(
        [
            [cos_angle, sin_angle, 0.0],
            [-sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    position_ecef_m = rotation @ position_m
    radius_m = float(np.linalg.norm(position_ecef_m))
    if radius_m == 0.0:
        return 0.0, 0.0, -body_radius_m / 1_000.0

    x_m, y_m, z_m = position_ecef_m
    latitude_deg = degrees(asin(np.clip(z_m / radius_m, -1.0, 1.0)))
    longitude_deg = degrees(atan2(y_m, x_m))
    altitude_km = (radius_m - body_radius_m) / 1_000.0
    return longitude_deg, latitude_deg, altitude_km


def _astropy_lon_lat_alt_km(
    position_m: np.ndarray,
    start_epoch_utc: str,
    elapsed_time_s: float,
) -> tuple[float, float, float]:
    epoch = Time(start_epoch_utc, scale="utc") + float(elapsed_time_s) * u.s
    gcrs = GCRS(
        CartesianRepresentation(position_m * u.m),
        obstime=epoch,
    )
    itrs = gcrs.transform_to(ITRS(obstime=epoch))
    location = EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)
    return (
        float(location.lon.to_value(u.deg)),
        float(location.lat.to_value(u.deg)),
        float(location.height.to_value(u.km)),
    )


def pymsis_atmosphere_density(
    position_m: np.ndarray,
    *,
    start_epoch_utc: str,
    elapsed_time_s: float,
    body_radius_m: float,
    body_rotation_rate_rad_s: float,
) -> float:
    if not PYMSIS_AVAILABLE:
        raise RuntimeError(
            "The pymsis atmosphere model was requested, but pymsis is not installed."
        )

    if ASTROPY_AVAILABLE:
        longitude_deg, latitude_deg, altitude_km = _astropy_lon_lat_alt_km(
            position_m=position_m,
            start_epoch_utc=start_epoch_utc,
            elapsed_time_s=elapsed_time_s,
        )
    else:
        longitude_deg, latitude_deg, altitude_km = _approximate_lon_lat_alt_km(
            position_m=position_m,
            elapsed_time_s=elapsed_time_s,
            body_radius_m=body_radius_m,
            body_rotation_rate_rad_s=body_rotation_rate_rad_s,
        )

    epoch = np.array(
        [_utc_datetime_at_elapsed_seconds(start_epoch_utc, elapsed_time_s)],
        dtype="datetime64[ns]",
    )
    try:
        output = pymsis_calculate(
            dates=epoch,
            lons=np.array([longitude_deg], dtype=float),
            lats=np.array([latitude_deg], dtype=float),
            alts=np.array([altitude_km], dtype=float),
            version=2.1,
        )
    except Exception as exc:  # pragma: no cover - depends on local pymsis/runtime data
        raise RuntimeError(
            "pymsis failed to evaluate density. Ensure the package is installed and "
            "space-weather input data are available in the container."
        ) from exc

    density_kg_m3 = float(np.asarray(output).reshape(-1, 11)[0, int(Variable.MASS_DENSITY)])
    return max(density_kg_m3, 0.0)


def piecewise_exponential_atmosphere_density(
    altitude_m: float,
    density_sea_level_kg_m3: float | None = None,
    scale_height_m: float | None = None,
) -> float:
    if altitude_m < 0.0:
        altitude_m = 0.0

    layer_index = 0
    for candidate_index, (base_altitude_m, _base_density, _scale_height_m) in enumerate(
        ATMOSPHERE_LAYERS
    ):
        if altitude_m >= base_altitude_m:
            layer_index = candidate_index
        else:
            break

    base_altitude_m, base_density_kg_m3, layer_scale_height_m = ATMOSPHERE_LAYERS[layer_index]
    return base_density_kg_m3 * np.exp(
        -(altitude_m - base_altitude_m) / layer_scale_height_m
    )


def atmospheric_drag_acceleration(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    start_epoch_utc: str,
    elapsed_time_s: float,
    body_radius_m: float,
    body_rotation_rate_rad_s: float,
    density_sea_level_kg_m3: float,
    scale_height_m: float,
    drag_coefficient: float,
    cross_section_area_m2: float,
    mass_kg: float,
    atmosphere_model: str = "piecewise_exponential",
    corotating_atmosphere: bool = True,
) -> np.ndarray:
    density_kg_m3, _atmosphere_velocity_m_s, relative_velocity_m_s = (
        atmospheric_environment(
            position_m=position_m,
            velocity_m_s=velocity_m_s,
            start_epoch_utc=start_epoch_utc,
            elapsed_time_s=elapsed_time_s,
            body_radius_m=body_radius_m,
            body_rotation_rate_rad_s=body_rotation_rate_rad_s,
            density_sea_level_kg_m3=density_sea_level_kg_m3,
            scale_height_m=scale_height_m,
            atmosphere_model=atmosphere_model,
            corotating_atmosphere=corotating_atmosphere,
        )
    )
    speed_m_s = np.linalg.norm(relative_velocity_m_s)
    if speed_m_s == 0.0:
        return np.zeros(3, dtype=float)

    ballistic_scale = -0.5 * density_kg_m3 * drag_coefficient * cross_section_area_m2 / mass_kg
    return ballistic_scale * speed_m_s * relative_velocity_m_s


def atmospheric_environment(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    start_epoch_utc: str,
    elapsed_time_s: float,
    body_radius_m: float,
    body_rotation_rate_rad_s: float,
    density_sea_level_kg_m3: float,
    scale_height_m: float,
    atmosphere_model: str = "piecewise_exponential",
    corotating_atmosphere: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return density, local atmosphere velocity, and their velocity difference."""
    altitude_m = np.linalg.norm(position_m) - body_radius_m
    if atmosphere_model == "piecewise_exponential":
        density_kg_m3 = piecewise_exponential_atmosphere_density(
            altitude_m,
            density_sea_level_kg_m3=density_sea_level_kg_m3,
            scale_height_m=scale_height_m,
        )
    elif atmosphere_model == "pymsis":
        density_kg_m3 = pymsis_atmosphere_density(
            position_m=position_m,
            start_epoch_utc=start_epoch_utc,
            elapsed_time_s=elapsed_time_s,
            body_radius_m=body_radius_m,
            body_rotation_rate_rad_s=body_rotation_rate_rad_s,
        )
    else:
        raise ValueError(f"Unsupported atmosphere model: {atmosphere_model}")

    if corotating_atmosphere:
        omega_vector_rad_s = np.array([0.0, 0.0, body_rotation_rate_rad_s], dtype=float)
        atmosphere_velocity_m_s = np.cross(omega_vector_rad_s, position_m)
    else:
        atmosphere_velocity_m_s = np.zeros(3, dtype=float)
    relative_velocity_m_s = velocity_m_s - atmosphere_velocity_m_s
    return density_kg_m3, atmosphere_velocity_m_s, relative_velocity_m_s
