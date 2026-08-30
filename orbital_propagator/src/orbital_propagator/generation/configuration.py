from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import yaml


class DataGenerationConfigError(ValueError):
    """Raised when the unified data-generation configuration is invalid."""


def default_data_generation_config_path() -> Path:
    """Return the packaged unified configuration path."""
    return Path(str(files("orbital_propagator.configs").joinpath("data_generation.yaml")))


def load_data_generation_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the single data-generation YAML document."""
    config_path = Path(path) if path is not None else default_data_generation_config_path()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise DataGenerationConfigError(
            f"Could not load data-generation configuration {config_path}: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise DataGenerationConfigError(
            f"Data-generation configuration {config_path} must contain a mapping."
        )

    required_sections = {
        "fixed_parameters",
        "sampled_parameters",
        "derived_environment",
        "dataset_recipes",
        "export",
    }
    missing_sections = sorted(required_sections.difference(config))
    if missing_sections:
        raise DataGenerationConfigError(
            "Data-generation configuration is missing sections: "
            + ", ".join(missing_sections)
        )
    return config


def get_config_section(
    config: Mapping[str, Any], section_name: str
) -> Mapping[str, Any]:
    section = config.get(section_name)
    if not isinstance(section, Mapping):
        raise DataGenerationConfigError(
            f"Configuration section {section_name!r} must be a mapping."
        )
    return section


def load_orbit_family_catalog(
    path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Load generic and Earth-specific orbit-family definitions."""
    sampled = get_config_section(load_data_generation_config(path), "sampled_parameters")
    families = sampled.get("orbit_families")
    if not isinstance(families, Mapping):
        raise DataGenerationConfigError(
            "sampled_parameters.orbit_families must be a mapping."
        )
    return families


def load_dataset_recipe(
    recipe_name: str,
    path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Load one named dataset recipe from the unified configuration."""
    recipes = get_config_section(load_data_generation_config(path), "dataset_recipes")
    recipe = recipes.get(recipe_name)
    if not isinstance(recipe, Mapping):
        raise DataGenerationConfigError(
            f"Unknown dataset recipe {recipe_name!r}; expected one of "
            + ", ".join(sorted(recipes))
            + "."
        )
    return recipe
