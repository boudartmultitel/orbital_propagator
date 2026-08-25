from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    SRC_PATH = Path(__file__).resolve().parents[1]
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a two-body orbital propagation and save the result artifact."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the JSON artifact to create.",
    )
    parser.add_argument(
        "--run-name",
        default="two_body_earth",
        help="Logical name of the run stored in the artifact.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=5400.0,
        help="Propagation duration in seconds.",
    )
    parser.add_argument(
        "--start-epoch-utc",
        default="2026-01-01T00:00:00Z",
        help="UTC epoch used for time-dependent ephemerides, in ISO 8601 format.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=181,
        help="Number of stored samples including the initial state.",
    )
    parser.add_argument(
        "--force-breakdown",
        "--force_breakdown",
        action="store_true",
        help=(
            "Include total and per-force acceleration vectors in the JSON artifact. "
            "The default artifact contains state vectors only."
        ),
    )
    parser.add_argument(
        "--orbit-definition",
        choices=("circular", "keplerian"),
        default="circular",
        help="How to build the initial orbit state.",
    )
    parser.add_argument(
        "--altitude-km",
        type=float,
        default=621.8637,
        help="Initial circular orbit altitude above Earth's radius in kilometers.",
    )
    parser.add_argument(
        "--semimajor-axis-km",
        type=float,
        default=None,
        help="Semimajor axis in kilometers when --orbit-definition keplerian is used.",
    )
    parser.add_argument(
        "--eccentricity",
        type=float,
        default=0.0,
        help="Initial orbital eccentricity when --orbit-definition keplerian is used.",
    )
    parser.add_argument(
        "--argument-of-periapsis-deg",
        type=float,
        default=0.0,
        help="Initial argument of periapsis in degrees for keplerian initialization.",
    )
    parser.add_argument(
        "--inclination-deg",
        type=float,
        default=0.0,
        help="Initial orbit inclination in degrees.",
    )
    parser.add_argument(
        "--raan-deg",
        type=float,
        default=0.0,
        help="Initial right ascension of ascending node in degrees.",
    )
    parser.add_argument(
        "--true-anomaly-deg",
        type=float,
        default=0.0,
        help="Initial true anomaly in degrees.",
    )
    parser.add_argument(
        "--integrator-backend",
        choices=("auto", "scipy", "rk4"),
        default="auto",
        help="Integrator backend to use.",
    )
    parser.add_argument(
        "--integrator-method",
        default="DOP853",
        help="SciPy solve_ivp method to use when the scipy backend is active.",
    )
    parser.add_argument(
        "--max-step-s",
        type=float,
        default=None,
        help=(
            "Optional maximum internal SciPy integration step in seconds. "
            "Leave unset to use tolerance-driven adaptive steps only."
        ),
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
        help="Relative tolerance for the SciPy backend.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for the SciPy backend.",
    )
    parser.add_argument(
        "--enable-j2",
        action="store_true",
        help="Enable the J2 perturbation force model.",
    )
    parser.add_argument(
        "--enable-drag",
        action="store_true",
        help="Enable the atmospheric drag force model.",
    )
    parser.add_argument(
        "--atmosphere-model",
        choices=("piecewise_exponential", "pymsis"),
        default="piecewise_exponential",
        help="Atmosphere density model to use when drag is enabled.",
    )
    parser.add_argument(
        "--disable-atmosphere-corotation",
        action="store_true",
        help="Disable the default corotating-atmosphere assumption in the drag model.",
    )
    parser.add_argument(
        "--enable-srp",
        action="store_true",
        help="Enable the solar radiation pressure force model.",
    )
    parser.add_argument(
        "--enable-third-body-sun",
        action="store_true",
        help="Enable the Sun third-body gravity force model.",
    )
    parser.add_argument(
        "--enable-third-body-moon",
        action="store_true",
        help="Enable the Moon third-body gravity force model.",
    )
    parser.add_argument(
        "--mass-kg",
        type=float,
        default=1000.0,
        help="Spacecraft mass in kilograms.",
    )
    parser.add_argument(
        "--cross-section-area-m2",
        type=float,
        default=10.0,
        help="Spacecraft cross-sectional area in square meters.",
    )
    parser.add_argument(
        "--drag-coefficient",
        type=float,
        default=2.2,
        help="Spacecraft drag coefficient.",
    )
    parser.add_argument(
        "--reflectivity-coefficient",
        type=float,
        default=1.2,
        help="Spacecraft reflectivity coefficient for SRP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.orbit_definition == "circular":
        initial_state_m_s = circular_orbit_state(
            central_body=EARTH,
            altitude_m=args.altitude_km * 1_000.0,
            inclination_deg=args.inclination_deg,
            raan_deg=args.raan_deg,
            true_anomaly_deg=args.true_anomaly_deg,
        )
    else:
        if args.semimajor_axis_km is None:
            raise ValueError(
                "--semimajor-axis-km is required when --orbit-definition keplerian is used."
            )
        initial_state_m_s = keplerian_orbit_state(
            central_body=EARTH,
            semimajor_axis_m=args.semimajor_axis_km * 1_000.0,
            eccentricity=args.eccentricity,
            inclination_deg=args.inclination_deg,
            raan_deg=args.raan_deg,
            argument_of_periapsis_deg=args.argument_of_periapsis_deg,
            true_anomaly_deg=args.true_anomaly_deg,
        )
    request = SimulationRequest(
        run_name=args.run_name,
        producer="simulation",
        central_body=EARTH,
        initial_state_m_s=initial_state_m_s,
        propagation=PropagationConfig(
            duration_s=args.duration_s,
            sample_count=args.sample_count,
            start_epoch_utc=args.start_epoch_utc,
        ),
        integrator=IntegratorConfig(
            backend=args.integrator_backend,
            method=args.integrator_method,
            rtol=args.rtol,
            atol=args.atol,
            max_step_s=args.max_step_s,
        ),
        spacecraft=SpacecraftConfig(
            mass_kg=args.mass_kg,
            cross_section_area_m2=args.cross_section_area_m2,
            drag_coefficient=args.drag_coefficient,
            reflectivity_coefficient=args.reflectivity_coefficient,
        ),
        forces=ForceModelConfig(
            central_gravity=True,
            j2=args.enable_j2,
            drag=args.enable_drag,
            atmosphere_model=args.atmosphere_model,
            corotating_atmosphere=not args.disable_atmosphere_corotation,
            solar_radiation_pressure=args.enable_srp,
            third_body_sun=args.enable_third_body_sun,
            third_body_moon=args.enable_third_body_moon,
        ),
    )
    result = run_simulation(request)
    artifact = build_run_artifact(
        request,
        result,
        force_breakdown=args.force_breakdown,
    )
    save_run_artifact(artifact, args.output)
    print(f"Wrote run artifact to {args.output}")


if __name__ == "__main__":
    main()
