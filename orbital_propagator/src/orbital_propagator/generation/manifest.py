from __future__ import annotations

import json
import re
from hashlib import sha256
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from math import degrees, isclose
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from orbital_propagator.config import (
    CentralBodyConfig,
    ForceModelConfig,
    IntegratorConfig,
    PropagationConfig,
    SimulationRequest,
    SpacecraftConfig,
    keplerian_orbit_state,
    validate_force_model,
)
from orbital_propagator.generation.sampling import (
    sample_generation_parameters,
    validate_sample,
)
from orbital_propagator.generation.configuration import load_data_generation_config
from orbital_propagator.io.artifacts import build_run_artifact, save_run_artifact
from orbital_propagator.propagation.runner import run_simulation


MANIFEST_SCHEMA_VERSION = "0.1"
TRAJECTORY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _start_epoch_utc(
    sampled: Mapping[str, Any],
    config: Mapping[str, Any],
    rng: np.random.Generator,
    override: str | None,
) -> str:
    if override is not None:
        return override
    prior = config["sampled_parameters"]["environment_priors"]["start_epoch_utc"]
    forces = sampled["forces"]
    if not any(bool(forces.get(name, False)) for name in prior["sample_for_forces"]):
        return str(prior["default"])
    start, end = (_parse_utc(str(value)) for value in prior["range"])
    if start >= end:
        raise ValueError("start_epoch_utc range must be strictly increasing.")
    sampled_timestamp = rng.uniform(start.timestamp(), end.timestamp())
    return datetime.fromtimestamp(sampled_timestamp, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def append_sampled_trajectories(
    manifest_path: Path,
    recipe_name: str,
    count: int,
    rng: np.random.Generator,
    *,
    duration_s: float = 5_400.0,
    sample_count: int = 181,
    start_epoch_utc: str | None = None,
    mass_kg: float = 1_000.0,
    integrator_backend: str = "auto",
    integrator_method: str = "DOP853",
    rtol: float = 1.0e-9,
    atol: float = 1.0e-9,
    max_step_s: float | None = None,
    config_path: str | Path | None = None,
    random_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Append fully specified sampled trajectories to a JSON Lines manifest."""
    if count <= 0:
        raise ValueError("Trajectory count must be strictly positive.")
    if duration_s <= 0.0 or sample_count < 2 or mass_kg <= 0.0:
        raise ValueError("Duration and mass must be positive; sample_count must be >= 2.")

    existing_count = len(load_manifest(manifest_path)) if manifest_path.exists() else 0
    config = load_data_generation_config(config_path)
    records: list[dict[str, Any]] = []
    for offset in range(count):
        sampled = sample_generation_parameters(recipe_name, rng, config_path)
        sampled.update(
            {
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "trajectory_id": f"{recipe_name}_{existing_count + offset:06d}",
                "sampling_seed": random_seed,
                "sampling_index": existing_count + offset,
                "mass_kg": mass_kg,
                "cross_section_area_m2": sampled["A_over_m"] * mass_kg,
                "propagation": {
                    "duration_s": duration_s,
                    "sample_count": sample_count,
                    "start_epoch_utc": _start_epoch_utc(
                        sampled, config, rng, start_epoch_utc
                    ),
                },
                "integrator": {
                    "backend": integrator_backend,
                    "method": integrator_method,
                    "rtol": rtol,
                    "atol": atol,
                    "max_step_s": max_step_s,
                },
            }
        )
        validate_manifest_record(sampled)
        records.append(sampled)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            handle.write("\n")
    return records


def iter_manifest(manifest_path: Path) -> Iterator[dict[str, Any]]:
    """Yield validated records while preserving useful JSONL line errors."""
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {manifest_path} at line {line_number}: {exc.msg}."
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Manifest line {line_number} must contain one JSON object."
                )
            try:
                validate_manifest_record(record)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid trajectory at {manifest_path}:{line_number}: {exc}"
                ) from exc
            yield record


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Trajectory manifest does not exist: {manifest_path}")
    records = list(iter_manifest(manifest_path))
    trajectory_ids = [record["trajectory_id"] for record in records]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("Manifest trajectory_id values must be unique.")
    return records


def _central_body_from_record(record: Mapping[str, Any]) -> CentralBodyConfig:
    reference_radius_km = record.get("J2_reference_radius_km")
    return CentralBodyConfig(
        name=str(record["central_body_name"]).title(),
        mu_m3_s2=float(record["mu_km3_s2"]) * 1.0e9,
        radius_m=float(record["radius_km"]) * 1.0e3,
        j2=float(record["J2"]) if record.get("J2") is not None else None,
        rotation_rate_rad_s=float(record["rotation_rate_rad_s"]),
        atmosphere_density_sea_level_kg_m3=float(
            record["atmosphere_density_sea_level_kg_m3"]
        ),
        atmosphere_scale_height_m=float(record["atmosphere_scale_height_m"]),
        atmosphere_model=str(record["atmosphere_model"]),
        heliocentric_distance_au=float(record["heliocentric_distance_au"]),
        j2_reference_radius_m=(
            float(reference_radius_km) * 1.0e3
            if reference_radius_km is not None
            else None
        ),
    )


def _forces_from_record(record: Mapping[str, Any]) -> ForceModelConfig:
    configured = record["forces"]
    if not isinstance(configured, Mapping):
        raise ValueError("forces must be a mapping.")
    third_bodies = record["third_bodies_enabled"]
    if not isinstance(third_bodies, list):
        raise ValueError("third_bodies_enabled must be a list.")
    return ForceModelConfig(
        central_gravity=bool(configured.get("two_body", False)),
        j2=bool(configured.get("J2", False)),
        drag=bool(configured.get("drag", False)),
        solar_radiation_pressure=bool(configured.get("SRP", False)),
        third_body_sun="sun" in third_bodies,
        third_body_moon="moon" in third_bodies,
    )


def validate_manifest_record(record: Mapping[str, Any]) -> None:
    validate_sample(record)
    trajectory_id = str(record["trajectory_id"])
    if TRAJECTORY_ID_PATTERN.fullmatch(trajectory_id) is None:
        raise ValueError(
            "trajectory_id may contain only letters, numbers, '.', '_', and '-'."
        )
    if record.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"manifest_schema_version must be {MANIFEST_SCHEMA_VERSION!r}."
        )
    if float(record["mass_kg"]) <= 0.0:
        raise ValueError("mass_kg must be strictly positive.")
    expected_area = float(record["A_over_m"]) * float(record["mass_kg"])
    if not isclose(float(record["cross_section_area_m2"]), expected_area):
        raise ValueError("cross_section_area_m2 must equal A_over_m * mass_kg.")
    propagation = record["propagation"]
    if not isinstance(propagation, Mapping):
        raise ValueError("propagation must be a mapping.")
    if float(propagation["duration_s"]) <= 0.0:
        raise ValueError("propagation.duration_s must be strictly positive.")
    if int(propagation["sample_count"]) < 2:
        raise ValueError("propagation.sample_count must be at least 2.")
    if not propagation.get("start_epoch_utc"):
        raise ValueError("propagation.start_epoch_utc must not be empty.")
    integrator = record["integrator"]
    if not isinstance(integrator, Mapping):
        raise ValueError("integrator must be a mapping.")
    if integrator.get("backend") not in {"auto", "scipy", "rk4"}:
        raise ValueError("integrator.backend must be auto, scipy, or rk4.")
    if float(integrator["rtol"]) <= 0.0 or float(integrator["atol"]) <= 0.0:
        raise ValueError("integrator tolerances must be strictly positive.")
    if integrator.get("max_step_s") is not None and float(
        integrator["max_step_s"]
    ) <= 0.0:
        raise ValueError("integrator.max_step_s must be positive when provided.")
    validate_force_model(_central_body_from_record(record), _forces_from_record(record))


def simulation_request_from_record(record: Mapping[str, Any]) -> SimulationRequest:
    """Convert one validated, self-contained manifest line to a simulation request."""
    validate_manifest_record(record)
    body = _central_body_from_record(record)
    propagation = record["propagation"]
    integrator = record["integrator"]
    initial_state = keplerian_orbit_state(
        central_body=body,
        semimajor_axis_m=float(record["a_km"]) * 1.0e3,
        eccentricity=float(record["e"]),
        inclination_deg=degrees(float(record["i_rad"])),
        raan_deg=degrees(float(record["raan_rad"])),
        argument_of_periapsis_deg=degrees(float(record["arg_perigee_rad"])),
        true_anomaly_deg=degrees(float(record["true_anomaly_rad"])),
    )
    return SimulationRequest(
        run_name=str(record["trajectory_id"]),
        producer="trajectory_manifest",
        central_body=body,
        initial_state_m_s=initial_state,
        propagation=PropagationConfig(
            duration_s=float(propagation["duration_s"]),
            sample_count=int(propagation["sample_count"]),
            start_epoch_utc=str(propagation["start_epoch_utc"]),
        ),
        integrator=IntegratorConfig(
            backend=str(integrator["backend"]),
            method=str(integrator["method"]),
            rtol=float(integrator["rtol"]),
            atol=float(integrator["atol"]),
            max_step_s=(
                float(integrator["max_step_s"])
                if integrator.get("max_step_s") is not None
                else None
            ),
        ),
        spacecraft=SpacecraftConfig(
            mass_kg=float(record["mass_kg"]),
            cross_section_area_m2=float(record["cross_section_area_m2"]),
            drag_coefficient=float(record["C_D"]),
            reflectivity_coefficient=float(record["C_R"]),
        ),
        forces=_forces_from_record(record),
    )


def build_manifest_dataset(
    manifest_path: Path,
    output_directory: Path,
    *,
    force_breakdown: bool = True,
    skip_existing: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Execute all manifest trajectories into existing JSON run artifacts."""
    records = load_manifest(manifest_path)
    if not records:
        raise ValueError("Cannot build a dataset from an empty trajectory manifest.")
    targets = [output_directory / f"{record['trajectory_id']}.json" for record in records]
    collisions = [path for path in targets if path.exists()]
    if collisions and not skip_existing:
        raise FileExistsError(
            f"Refusing to overwrite {len(collisions)} existing trajectory artifact(s); "
            "use skip_existing=True to resume."
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, (record, output_path) in enumerate(
        zip(records, targets, strict=True), start=1
    ):
        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as handle:
                existing_artifact = json.load(handle)
            if existing_artifact.get("manifest_parameters") != record:
                raise ValueError(
                    f"Existing artifact {output_path} does not match its manifest line."
                )
        else:
            request = simulation_request_from_record(record)
            artifact = build_run_artifact(
                request, run_simulation(request), force_breakdown=force_breakdown
            )
            artifact["run_id"] = record["trajectory_id"]
            artifact["manifest_parameters"] = record
            save_run_artifact(artifact, output_path)
            written.append(output_path)
        if progress_callback is not None:
            progress_callback(index, len(records))

    metadata = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_type": "trajectory_artifacts",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "trajectory_count": len(records),
        "force_breakdown": force_breakdown,
        "artifacts": [path.name for path in targets],
    }
    save_run_artifact(metadata, output_directory / "metadata.json")
    return written
