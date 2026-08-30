from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from orbital_propagator.config import CentralBodyConfig
from orbital_propagator.generation.configuration import (
    DataGenerationConfigError,
    get_config_section,
    load_data_generation_config,
)


ALLOWED_CENTRAL_BODY_NAMES = frozenset(
    {"mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"}
)


def load_planet_catalog(
    path: str | Path | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Load central-body entries from the unified generation configuration."""
    config = load_data_generation_config(path)
    fixed_parameters = get_config_section(config, "fixed_parameters")
    catalog = fixed_parameters.get("central_bodies")
    if not isinstance(catalog, Mapping):
        raise DataGenerationConfigError(
            "fixed_parameters.central_bodies must be a mapping."
        )

    catalog_names = set(catalog)
    if catalog_names != ALLOWED_CENTRAL_BODY_NAMES:
        missing = sorted(ALLOWED_CENTRAL_BODY_NAMES - catalog_names)
        unexpected = sorted(catalog_names - ALLOWED_CENTRAL_BODY_NAMES)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise DataGenerationConfigError(
            "Central-body catalog must contain exactly the eight allowed planets ("
            + "; ".join(details)
            + ")."
        )
    return catalog


def load_third_body_catalog(
    path: str | Path | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Load perturbing bodies separately from the central-body catalog."""
    config = load_data_generation_config(path)
    fixed_parameters = get_config_section(config, "fixed_parameters")
    catalog = fixed_parameters.get("third_bodies")
    if not isinstance(catalog, Mapping):
        raise DataGenerationConfigError(
            "fixed_parameters.third_bodies must be a mapping."
        )
    return catalog


def central_body_from_catalog(
    body_name: str,
    *,
    path: str | Path | None = None,
) -> CentralBodyConfig:
    """Build the existing SI-based propagator config from a catalog entry."""
    normalized_name = body_name.strip().lower()
    catalog = load_planet_catalog(path)
    if normalized_name not in catalog:
        raise DataGenerationConfigError(
            f"Unsupported central body {body_name!r}; expected one of "
            + ", ".join(sorted(ALLOWED_CENTRAL_BODY_NAMES))
            + "."
        )

    entry = catalog[normalized_name]
    missing_required = [
        field_name
        for field_name in ("mu_km3_s2", "radius_km")
        if entry.get(field_name) is None
    ]
    if missing_required:
        raise DataGenerationConfigError(
            f"Central body {normalized_name!r} has no authoritative value for: "
            + ", ".join(missing_required)
            + "."
        )

    config = load_data_generation_config(path)
    environment = get_config_section(config, "derived_environment")
    atmosphere_models = environment.get("atmosphere_models", {})
    atmosphere_name = str(entry.get("atmosphere_model", "none"))
    atmosphere = atmosphere_models.get(atmosphere_name)
    if atmosphere_name != "none" and not isinstance(atmosphere, Mapping):
        raise DataGenerationConfigError(
            f"Central body {normalized_name!r} references unknown atmosphere model "
            f"{atmosphere_name!r}."
        )

    return CentralBodyConfig(
        name=str(entry.get("name", normalized_name.title())),
        mu_m3_s2=float(entry["mu_km3_s2"]) * 1.0e9,
        radius_m=float(entry["radius_km"]) * 1.0e3,
        j2=float(entry["J2"]) if entry.get("J2") is not None else None,
        rotation_rate_rad_s=(
            float(entry["rotation_rate_rad_s"])
            if entry.get("rotation_rate_rad_s") is not None
            else 0.0
        ),
        atmosphere_model=atmosphere_name,
        heliocentric_distance_au=float(entry["heliocentric_distance_au"]),
        j2_reference_radius_m=(
            float(entry["J2_reference_radius_km"]) * 1.0e3
            if entry.get("J2_reference_radius_km") is not None
            else None
        ),
        atmosphere_density_sea_level_kg_m3=(
            float(atmosphere["density_sea_level_kg_m3"])
            if isinstance(atmosphere, Mapping)
            else 0.0
        ),
        atmosphere_scale_height_m=(
            float(atmosphere["scale_height_m"])
            if isinstance(atmosphere, Mapping)
            else 1.0
        ),
    )
