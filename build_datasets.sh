#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${BASE_DIR}/data"

# Container-internal paths (docker-compose mounts ./data to /shared/data)
CONTAINER_DATA_DIR="/shared/data"
MANIFEST_DIR="${CONTAINER_DATA_DIR}/manifests"
DATASET_DIR="${CONTAINER_DATA_DIR}/datasets"

declare -A RECIPES=(
  [multi_planet_two_body]="multi_planet_two_body_1000_1day_60s.jsonl"
  [multi_planet_j2]="multi_planet_j2_1000_1day_60s.jsonl"
  [earth_j2_drag]="earth_j2_drag_1000_1day_60s.jsonl"
  [multi_planet_j2_third_body]="multi_planet_j2_third_body_1000_1day_60s.jsonl"
  [all_body_full_perturbations]="all_body_full_perturbations_1000_1day_60s.jsonl"
)

# Phase 1: Validate all manifests
echo "========================================="
echo " Phase 1: Validating manifests"
echo "========================================="
for recipe in "${!RECIPES[@]}"; do
  fname="${RECIPES[$recipe]}"
  manifest="${MANIFEST_DIR}/${fname}"

  if [[ ! -f "${manifest}" ]]; then
    echo "SKIP: ${manifest} not found (run generate_manifests.sh first)"
    continue
  fi

  echo "Validating: ${recipe}"
  docker compose run --rm orbital_propagator manifest validate \
    --manifest "${manifest}"
  echo ""
done

# Phase 2: Build trajectories from each manifest
echo "========================================="
echo " Phase 2: Building trajectory datasets"
echo "========================================="
for recipe in "${!RECIPES[@]}"; do
  fname="${RECIPES[$recipe]}"
  manifest="${MANIFEST_DIR}/${fname}"
  output_dir="${DATASET_DIR}/${fname%.jsonl}"

  if [[ ! -f "${manifest}" ]]; then
    echo "SKIP: ${manifest} not found"
    continue
  fi

  mkdir -p "${output_dir}"

  echo "Building: ${recipe} -> ${output_dir}"
  docker compose run --rm orbital_propagator manifest build \
    --manifest "${manifest}" \
    --output-dir "${output_dir}" \
    --skip-existing
  echo ""
done

echo "========================================="
echo " Done."
echo "========================================="
echo "Trajectory datasets:"
ls -lhR "${DATASET_DIR}/" 2>/dev/null || echo "(none found)"
