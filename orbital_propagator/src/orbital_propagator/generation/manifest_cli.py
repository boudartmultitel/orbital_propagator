from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from orbital_propagator.generation.manifest import (
    append_sampled_trajectories,
    build_manifest_dataset,
    load_manifest,
)
from orbital_propagator.generation.configuration import load_data_generation_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orbital_propagator.cli manifest",
        description="Create, validate, and execute JSONL trajectory manifests.",
    )
    commands = parser.add_subparsers(dest="manifest_command", required=True)

    commands.add_parser("recipes", help="List available dataset recipe names.")

    append = commands.add_parser(
        "append", help="Append reproducibly sampled trajectories to a manifest."
    )
    append.add_argument("--manifest", type=Path, required=True)
    append.add_argument("--recipe", required=True)
    append.add_argument("--count", type=int, default=1)
    append.add_argument("--seed", type=int, default=42)
    append.add_argument("--duration-s", type=float, default=5_400.0)
    append.add_argument("--sample-count", type=int, default=181)
    append.add_argument(
        "--start-epoch-utc",
        default=None,
        help=(
            "Fixed epoch for every trajectory. By default, time-dependent recipes "
            "sample the configured epoch range."
        ),
    )
    append.add_argument("--mass-kg", type=float, default=1_000.0)
    append.add_argument(
        "--integrator-backend", choices=("auto", "scipy", "rk4"), default="auto"
    )
    append.add_argument("--integrator-method", default="DOP853")
    append.add_argument("--rtol", type=float, default=1.0e-9)
    append.add_argument("--atol", type=float, default=1.0e-9)
    append.add_argument("--max-step-s", type=float, default=None)
    append.add_argument("--config", type=Path, default=None)

    validate = commands.add_parser(
        "validate", help="Validate every non-empty manifest line."
    )
    validate.add_argument("--manifest", type=Path, required=True)

    build = commands.add_parser(
        "build", help="Propagate every manifest line into a trajectory dataset."
    )
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument(
        "--state-only",
        action="store_true",
        help="Omit per-force acceleration arrays from trajectory artifacts.",
    )
    build.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume by retaining trajectory artifacts that already exist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.manifest_command == "recipes":
        recipes = load_data_generation_config()["dataset_recipes"]
        print("\n".join(sorted(recipes)))
    elif args.manifest_command == "append":
        records = append_sampled_trajectories(
            args.manifest,
            args.recipe,
            args.count,
            np.random.default_rng(args.seed),
            duration_s=args.duration_s,
            sample_count=args.sample_count,
            start_epoch_utc=args.start_epoch_utc,
            mass_kg=args.mass_kg,
            integrator_backend=args.integrator_backend,
            integrator_method=args.integrator_method,
            rtol=args.rtol,
            atol=args.atol,
            max_step_s=args.max_step_s,
            config_path=args.config,
            random_seed=args.seed,
        )
        print(f"Appended {len(records)} trajectories to {args.manifest}")
    elif args.manifest_command == "validate":
        records = load_manifest(args.manifest)
        print(f"Validated {len(records)} trajectories in {args.manifest}")
    else:
        def show_progress(completed: int, total: int) -> None:
            width = 30
            filled = width * completed // total
            bar = "#" * filled + "-" * (width - filled)
            end = "\n" if completed == total else "\r"
            print(
                f"[{bar}] {completed}/{total} trajectories",
                end=end,
                file=sys.stderr,
                flush=True,
            )

        written = build_manifest_dataset(
            args.manifest,
            args.output_dir,
            force_breakdown=not args.state_only,
            skip_existing=args.skip_existing,
            progress_callback=show_progress,
        )
        print(f"Wrote {len(written)} trajectory artifacts to {args.output_dir}")
