from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from orbital_propagator.config import SimulationRequest
from orbital_propagator.propagation.runner import SimulationResult


SCHEMA_VERSION = "0.4.0"


def _rounded_list(array: np.ndarray, decimals: int = 9) -> list[Any]:
    return np.round(array, decimals=decimals).tolist()


def build_run_artifact(
    request: SimulationRequest,
    result: SimulationResult,
) -> dict[str, Any]:
    metadata = dict(result.metadata)
    if "reference_vectors_m" in metadata:
        metadata["reference_vectors_m"] = {
            name: _rounded_list(values)
            for name, values in metadata["reference_vectors_m"].items()
        }
    if "reference_vector_tracks_m" in metadata:
        metadata["reference_vector_tracks_m"] = {
            name: {
                "times_s": _rounded_list(track["times_s"]),
                "vectors_m": _rounded_list(track["vectors_m"]),
            }
            for name, track in metadata["reference_vector_tracks_m"].items()
        }

    summary = {
        "sample_count": int(len(result.times_s)),
        "duration_s": float(result.times_s[-1] - result.times_s[0]),
        "min_radius_m": float(np.min(result.derived_series["radius_m"])),
        "max_radius_m": float(np.max(result.derived_series["radius_m"])),
        "min_altitude_m": float(
            np.min(result.derived_series["radius_m"]) - request.central_body.radius_m
        ),
        "max_altitude_m": float(
            np.max(result.derived_series["radius_m"]) - request.central_body.radius_m
        ),
        "min_speed_m_s": float(np.min(result.derived_series["speed_m_s"])),
        "max_speed_m_s": float(np.max(result.derived_series["speed_m_s"])),
        "specific_energy_span_j_kg": float(
            np.max(result.derived_series["specific_energy_j_kg"])
            - np.min(result.derived_series["specific_energy_j_kg"])
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid4()),
        "run_name": request.run_name,
        "producer": request.producer,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "central_body": request.central_body.name,
        "enabled_forces": list(result.accelerations_by_force_m_s2.keys()),
        "parameters": {
            "central_body": asdict(request.central_body),
            "propagation": asdict(request.propagation),
            "integrator": asdict(request.integrator),
            "spacecraft": asdict(request.spacecraft),
            "forces": asdict(request.forces),
        },
        "initial_conditions": {
            "state_vector_m_s": _rounded_list(request.initial_state_m_s),
        },
        "summary": summary,
        "metadata": metadata,
        "times_s": _rounded_list(result.times_s),
        "states_m_s": _rounded_list(result.states_m_s),
        "accelerations_total_m_s2": _rounded_list(result.accelerations_total_m_s2),
        "accelerations_by_force_m_s2": {
            force_name: _rounded_list(values)
            for force_name, values in result.accelerations_by_force_m_s2.items()
        },
        "derived_series": {
            name: _rounded_list(values)
            for name, values in result.derived_series.items()
        },
    }


def save_run_artifact(artifact: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")
