from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, pi
from pathlib import Path
from typing import Any

import numpy as np

from orbital_propagator.bodies.catalog import central_body_from_catalog
from orbital_propagator.config import (
    CentralBodyConfig,
    ForceModelConfig,
    validate_force_model,
)
from orbital_propagator.generation.configuration import (
    DataGenerationConfigError,
    load_data_generation_config,
    load_dataset_recipe,
    load_orbit_family_catalog,
)


EARTH_SPECIFIC_FAMILIES = frozenset(
    {"drag_leo", "leo", "sso", "geo", "molniya"}
)


def _weighted_choice(
    weights: Mapping[str, Any], rng: np.random.Generator, label: str
) -> str:
    if not weights:
        raise DataGenerationConfigError(f"{label} weights must not be empty.")
    names = list(weights)
    values = np.asarray([weights[name] for name in names], dtype=float)
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or values.sum() <= 0.0
    ):
        raise DataGenerationConfigError(
            f"{label} weights must be finite, non-negative, and sum above zero."
        )
    return str(rng.choice(names, p=values / values.sum()))


def _uniform_range(
    config: Mapping[str, Any], key: str, rng: np.random.Generator
) -> float:
    bounds = config.get(key)
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise DataGenerationConfigError(f"{key} must be a two-value range.")
    low, high = map(float, bounds)
    if not isfinite(low) or not isfinite(high) or low > high:
        raise DataGenerationConfigError(f"{key} contains invalid bounds.")
    return float(rng.uniform(low, high)) if low < high else low


def sample_central_body(
    recipe: Mapping[str, Any], rng: np.random.Generator
) -> str:
    weights = recipe.get("central_bodies")
    if not isinstance(weights, Mapping):
        raise DataGenerationConfigError("recipe.central_bodies must be a mapping.")
    return _weighted_choice(weights, rng, "central-body")


def sample_orbit_family(
    recipe: Mapping[str, Any], selected_body: str, rng: np.random.Generator
) -> str:
    weights = recipe.get("orbit_families")
    body_weights = recipe.get("orbit_families_by_body", {})
    if not isinstance(body_weights, Mapping):
        raise DataGenerationConfigError(
            "recipe.orbit_families_by_body must be a mapping."
        )
    weights = body_weights.get(selected_body, weights)
    if not isinstance(weights, Mapping):
        raise DataGenerationConfigError("recipe.orbit_families must be a mapping.")
    if selected_body != "earth" and EARTH_SPECIFIC_FAMILIES.intersection(weights):
        raise DataGenerationConfigError(
            "Earth-specific orbit families cannot be sampled around " + selected_body + "."
        )
    return _weighted_choice(weights, rng, "orbit-family")


def _family_config(
    family_name: str,
    selected_body: str,
    catalog: Mapping[str, Any],
) -> Mapping[str, Any]:
    group_name = (
        "earth_specific" if family_name in EARTH_SPECIFIC_FAMILIES else "generic"
    )
    group = catalog.get(group_name)
    family = group.get(family_name) if isinstance(group, Mapping) else None
    if not isinstance(family, Mapping):
        raise DataGenerationConfigError(f"Unknown orbit family {family_name!r}.")
    required_body = family.get("central_body")
    if required_body is not None and required_body != selected_body:
        raise DataGenerationConfigError(
            f"Orbit family {family_name!r} requires central body {required_body!r}."
        )
    return family


def sample_orbital_elements(
    family_config: Mapping[str, Any],
    planet_config: CentralBodyConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample valid elliptical elements, converting body-scaled ranges to km."""
    radius_km = planet_config.radius_m / 1_000.0
    for _ in range(1_000):
        if "perigee_altitude_over_radius" in family_config:
            perigee_km = radius_km * (
                1.0
                + _uniform_range(
                    family_config, "perigee_altitude_over_radius", rng
                )
            )
            apogee_km = radius_km * (
                1.0
                + _uniform_range(family_config, "apogee_altitude_over_radius", rng)
            )
            a_km = (perigee_km + apogee_km) / 2.0
            eccentricity = (apogee_km - perigee_km) / (apogee_km + perigee_km)
        elif "perigee_altitude_km" in family_config:
            perigee_km = radius_km + _uniform_range(
                family_config, "perigee_altitude_km", rng
            )
            apogee_km = radius_km + _uniform_range(
                family_config, "apogee_altitude_km", rng
            )
            a_km = (perigee_km + apogee_km) / 2.0
            eccentricity = (apogee_km - perigee_km) / (apogee_km + perigee_km)
        else:
            eccentricity = _uniform_range(family_config, "eccentricity", rng)
            if "altitude_over_radius" in family_config:
                a_km = radius_km * (
                    1.0 + _uniform_range(family_config, "altitude_over_radius", rng)
                )
            elif "semi_major_axis_over_radius" in family_config:
                a_km = radius_km * _uniform_range(
                    family_config, "semi_major_axis_over_radius", rng
                )
            elif "altitude_km" in family_config:
                a_km = radius_km + _uniform_range(
                    family_config, "altitude_km", rng
                )
            else:
                raise DataGenerationConfigError(
                    "Orbit family has no supported size parameter."
                )

        eccentricity_bounds = family_config.get("eccentricity")
        eccentricity_allowed = (
            isinstance(eccentricity_bounds, list)
            and len(eccentricity_bounds) == 2
            and float(eccentricity_bounds[0])
            <= eccentricity
            <= float(eccentricity_bounds[1])
        )
        if eccentricity_allowed and a_km * (1.0 - eccentricity) > radius_km:
            break
    else:
        raise DataGenerationConfigError(
            "Could not sample a valid orbit from the configured family ranges."
        )

    return {
        "a_km": a_km,
        "e": eccentricity,
        "i_rad": _uniform_range(family_config, "inclination_deg", rng) * pi / 180.0,
        "raan_rad": _uniform_range(family_config, "raan_deg", rng) * pi / 180.0,
        "arg_perigee_rad": _uniform_range(family_config, "arg_perigee_deg", rng)
        * pi
        / 180.0,
        "true_anomaly_rad": _uniform_range(family_config, "true_anomaly_deg", rng)
        * pi
        / 180.0,
    }


def sample_spacecraft_parameters(
    rng: np.random.Generator,
    priors: Mapping[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, float | str]:
    """Sample every spacecraft prior and its two derived force coefficients."""
    if priors is None:
        priors = load_data_generation_config(path)["sampled_parameters"][
            "spacecraft_priors"
        ]
    drag_coefficient = _uniform_range(priors["drag_coefficient"], "range", rng)
    reflectivity_coefficient = _uniform_range(
        priors["reflectivity_coefficient"], "range", rng
    )
    bins = priors["area_to_mass"].get("bins")
    if not isinstance(bins, Mapping):
        raise DataGenerationConfigError("area_to_mass.bins must be a mapping.")
    bin_weights = priors["area_to_mass"].get("bin_weights")
    if bin_weights is None:
        bin_name = str(rng.choice(list(bins)))
    elif isinstance(bin_weights, Mapping):
        bin_name = _weighted_choice(bin_weights, rng, "area-to-mass bin")
        if bin_name not in bins:
            raise DataGenerationConfigError(
                f"Unknown area-to-mass bin {bin_name!r} in bin_weights."
            )
    else:
        raise DataGenerationConfigError(
            "area_to_mass.bin_weights must be a mapping."
        )
    area_to_mass = _uniform_range({"range": bins[bin_name]}, "range", rng)
    return {
        "C_D": drag_coefficient,
        "C_R": reflectivity_coefficient,
        "A_over_m": area_to_mass,
        "area_to_mass_bin": bin_name,
        "gamma_D": drag_coefficient * area_to_mass,
        "gamma_R": reflectivity_coefficient * area_to_mass,
    }


def _resolve_availability(
    setting: Any,
    available: bool,
    label: str,
) -> bool:
    if not available:
        return False
    if isinstance(setting, bool):
        return setting
    if setting == "when_available":
        return True
    raise DataGenerationConfigError(
        f"{label} must be true, false, or 'when_available'."
    )


def _force_config(
    recipe: Mapping[str, Any],
    planet: CentralBodyConfig,
) -> ForceModelConfig:
    forces = recipe.get("forces")
    third_bodies = recipe.get("third_bodies")
    if not isinstance(forces, Mapping) or not isinstance(third_bodies, Mapping):
        raise DataGenerationConfigError(
            "Recipe forces and third_bodies must both be mappings."
        )
    enable_third_body = _resolve_availability(
        forces.get("third_body", False), True, "forces.third_body"
    )
    body_name = planet.name.lower()
    return ForceModelConfig(
        central_gravity=_resolve_availability(
            forces.get("two_body", False), True, "forces.two_body"
        ),
        j2=_resolve_availability(
            forces.get("J2", False), planet.j2 is not None, "forces.J2"
        ),
        drag=_resolve_availability(
            forces.get("drag", False),
            planet.atmosphere_model != "none",
            "forces.drag",
        ),
        solar_radiation_pressure=_resolve_availability(
            forces.get("SRP", False),
            planet.heliocentric_distance_au is not None,
            "forces.SRP",
        ),
        third_body_sun=enable_third_body
        and _resolve_availability(
            third_bodies.get("sun", False),
            planet.heliocentric_distance_au is not None,
            "third_bodies.sun",
        ),
        third_body_moon=enable_third_body
        and _resolve_availability(
            third_bodies.get("moon", False),
            body_name == "earth",
            "third_bodies.moon",
        ),
    )


def _resolved_force_flags(forces: ForceModelConfig) -> dict[str, bool]:
    return {
        "two_body": forces.central_gravity,
        "J2": forces.j2,
        "drag": forces.drag,
        "third_body": forces.third_body_sun or forces.third_body_moon,
        "SRP": forces.solar_radiation_pressure,
    }


def validate_sample(sample: Mapping[str, Any]) -> None:
    """Validate the physical and categorical invariants of a sampled object."""
    finite_fields = (
        "mu_km3_s2",
        "radius_km",
        "a_km",
        "e",
        "i_rad",
        "raan_rad",
        "arg_perigee_rad",
        "true_anomaly_rad",
        "C_D",
        "C_R",
        "A_over_m",
        "gamma_D",
        "gamma_R",
    )
    if any(not isfinite(float(sample[field])) for field in finite_fields):
        raise ValueError("Sample contains non-finite physical parameters.")
    if not 0.0 <= float(sample["e"]) < 1.0:
        raise ValueError("Sample eccentricity must satisfy 0 <= e < 1.")
    if float(sample["a_km"]) * (1.0 - float(sample["e"])) <= float(
        sample["radius_km"]
    ):
        raise ValueError("Sample periapsis must be above the central-body surface.")
    if (
        sample["orbit_family"] in EARTH_SPECIFIC_FAMILIES
        and sample["central_body_name"] != "earth"
    ):
        raise ValueError("Earth-specific orbit family sampled for a non-Earth body.")
    if (
        "moon" in sample["third_bodies_enabled"]
        and sample["central_body_name"] != "earth"
    ):
        raise ValueError("The Moon third body is only supported for Earth.")
    if float(sample["A_over_m"]) <= 0.0:
        raise ValueError("Sample area-to-mass ratio must be positive.")
    if not np.isclose(sample["gamma_D"], sample["C_D"] * sample["A_over_m"]):
        raise ValueError("gamma_D must equal C_D * A_over_m.")
    if not np.isclose(sample["gamma_R"], sample["C_R"] * sample["A_over_m"]):
        raise ValueError("gamma_R must equal C_R * A_over_m.")


def sample_generation_parameters(
    recipe_name: str,
    rng: np.random.Generator,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Sample one complete Phase 7 parameter object from a Phase 6 recipe."""
    recipe = load_dataset_recipe(recipe_name, path)
    selected_body = sample_central_body(recipe, rng)
    planet = central_body_from_catalog(selected_body, path=path)
    forces = _force_config(recipe, planet)
    validate_force_model(planet, forces)

    family_name = sample_orbit_family(recipe, selected_body, rng)
    family = _family_config(
        family_name, selected_body, load_orbit_family_catalog(path)
    )
    config = load_data_generation_config(path)
    spacecraft = sample_spacecraft_parameters(
        rng, config["sampled_parameters"]["spacecraft_priors"]
    )
    third_bodies = [
        name
        for name in ("sun", "moon")
        if getattr(forces, f"third_body_{name}")
    ]
    sample: dict[str, Any] = {
        "recipe_name": recipe_name,
        "dataset_name": recipe["dataset_name"],
        "central_body_name": selected_body,
        "mu_km3_s2": planet.mu_m3_s2 / 1.0e9,
        "radius_km": planet.radius_m / 1.0e3,
        "J2": planet.j2,
        "J2_reference_radius_km": (
            planet.j2_reference_radius_m / 1.0e3
            if planet.j2_reference_radius_m is not None
            else None
        ),
        "rotation_rate_rad_s": planet.rotation_rate_rad_s,
        "atmosphere_model": planet.atmosphere_model,
        "atmosphere_density_sea_level_kg_m3": (
            planet.atmosphere_density_sea_level_kg_m3
        ),
        "atmosphere_scale_height_m": planet.atmosphere_scale_height_m,
        "heliocentric_distance_au": planet.heliocentric_distance_au,
        "third_bodies_enabled": third_bodies,
        "orbit_family": family_name,
        "forces": _resolved_force_flags(forces),
    }
    sample.update(sample_orbital_elements(family, planet, rng))
    sample.update(spacecraft)
    validate_sample(sample)
    return sample
