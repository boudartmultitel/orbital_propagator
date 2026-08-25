from __future__ import annotations

import math
import re
from pathlib import Path


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "ui_run"


def _unique_output_path(results_dir: Path, slug: str) -> Path:
    candidate = results_dir / f"{slug}.json"
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = results_dir / f"{slug}_{index:02d}.json"
        if not candidate.exists():
            return candidate
        index += 1


def estimate_sampling_parameters(
    *,
    orbit_definition: str,
    duration_s: float,
    samples_per_orbit: int,
    altitude_km: float | None = None,
    semimajor_axis_km: float | None = None,
) -> dict[str, float]:
    try:
        from orbital_propagator.bodies.earth import EARTH
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The visualization container cannot import orbital_propagator. "
            "Check the Docker volume mount and PYTHONPATH configuration."
        ) from exc

    if orbit_definition == "circular":
        if altitude_km is None:
            raise ValueError("Altitude is required for circular-orbit sampling estimates.")
        semimajor_axis_m = EARTH.radius_m + float(altitude_km) * 1_000.0
    elif orbit_definition == "keplerian":
        if semimajor_axis_km is None:
            raise ValueError("Semimajor axis is required for keplerian sampling estimates.")
        semimajor_axis_m = float(semimajor_axis_km) * 1_000.0
    else:
        raise ValueError(f"Unsupported orbit definition: {orbit_definition}")

    orbit_period_s = 2.0 * math.pi * math.sqrt(semimajor_axis_m**3 / EARTH.mu_m3_s2)
    orbit_count = max(float(duration_s), 0.0) / orbit_period_s if orbit_period_s > 0.0 else 0.0
    sample_count = max(
        2,
        int(math.ceil(max(int(samples_per_orbit), 1) * orbit_count)) + 1,
    )
    return {
        "orbit_period_s": orbit_period_s,
        "orbit_count": orbit_count,
        "sample_count": float(sample_count),
    }


def launch_simulation_from_ui(
    *,
    results_dir: Path,
    run_name: str,
    orbit_definition: str,
    altitude_km: float | None,
    semimajor_axis_km: float | None,
    eccentricity: float,
    inclination_deg: float,
    raan_deg: float,
    argument_of_periapsis_deg: float,
    true_anomaly_deg: float,
    duration_s: float,
    samples_per_orbit: int,
    start_epoch_utc: str,
    integrator_backend: str,
    mass_kg: float,
    cross_section_area_m2: float,
    drag_coefficient: float,
    reflectivity_coefficient: float,
    atmosphere_model: str,
    corotating_atmosphere: bool,
    enable_j2: bool,
    enable_drag: bool,
    enable_solar_radiation_pressure: bool,
    enable_third_body_sun: bool,
    enable_third_body_moon: bool,
) -> Path:
    try:
        from orbital_propagator.bodies.earth import EARTH
        from orbital_propagator.config import (
            ForceModelConfig,
            IntegratorConfig,
            PropagationConfig,
            SimulationRequest,
            SpacecraftConfig,
            circular_orbit_state,
            keplerian_orbit_state,
        )
        from orbital_propagator.io.artifacts import build_run_artifact, save_run_artifact
        from orbital_propagator.propagation.runner import run_simulation
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The visualization container cannot import orbital_propagator. "
            "Check the Docker volume mount and PYTHONPATH configuration."
        ) from exc

    clean_name = run_name.strip() or "UI Run"
    slug = _safe_slug(clean_name)
    output_path = _unique_output_path(results_dir, slug)
    sampling = estimate_sampling_parameters(
        orbit_definition=orbit_definition,
        duration_s=duration_s,
        samples_per_orbit=samples_per_orbit,
        altitude_km=altitude_km,
        semimajor_axis_km=semimajor_axis_km,
    )
    if orbit_definition == "circular":
        if altitude_km is None:
            raise ValueError("Altitude is required for circular orbit initialization.")
        initial_state_m_s = circular_orbit_state(
            central_body=EARTH,
            altitude_m=altitude_km * 1_000.0,
            inclination_deg=inclination_deg,
            raan_deg=raan_deg,
            true_anomaly_deg=true_anomaly_deg,
        )
    elif orbit_definition == "keplerian":
        if semimajor_axis_km is None:
            raise ValueError("Semimajor axis is required for keplerian orbit initialization.")
        initial_state_m_s = keplerian_orbit_state(
            central_body=EARTH,
            semimajor_axis_m=semimajor_axis_km * 1_000.0,
            eccentricity=eccentricity,
            inclination_deg=inclination_deg,
            raan_deg=raan_deg,
            argument_of_periapsis_deg=argument_of_periapsis_deg,
            true_anomaly_deg=true_anomaly_deg,
        )
    else:
        raise ValueError(f"Unsupported orbit definition: {orbit_definition}")
    request = SimulationRequest(
        run_name=clean_name,
        producer="ui",
        central_body=EARTH,
        initial_state_m_s=initial_state_m_s,
        propagation=PropagationConfig(
            duration_s=duration_s,
            sample_count=int(sampling["sample_count"]),
            start_epoch_utc=start_epoch_utc,
        ),
        integrator=IntegratorConfig(
            backend=integrator_backend,
        ),
        spacecraft=SpacecraftConfig(
            mass_kg=mass_kg,
            cross_section_area_m2=cross_section_area_m2,
            drag_coefficient=drag_coefficient,
            reflectivity_coefficient=reflectivity_coefficient,
        ),
        forces=ForceModelConfig(
            central_gravity=True,
            j2=enable_j2,
            drag=enable_drag,
            atmosphere_model=atmosphere_model,
            corotating_atmosphere=corotating_atmosphere,
            solar_radiation_pressure=enable_solar_radiation_pressure,
            third_body_sun=enable_third_body_sun,
            third_body_moon=enable_third_body_moon,
        ),
    )
    result = run_simulation(request)
    # The dashboard displays acceleration diagnostics, so its artifacts need
    # the optional force-component time series.
    artifact = build_run_artifact(request, result, force_breakdown=True)
    save_run_artifact(artifact, output_path)
    return output_path
