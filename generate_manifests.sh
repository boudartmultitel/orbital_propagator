#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${BASE_DIR}/data"

# Container-internal paths (docker-compose mounts ./data to /shared/data)
CONTAINER_DATA_DIR="/shared/data"
MANIFEST_DIR="${CONTAINER_DATA_DIR}/manifests"

SEED=42
DURATION_S=86400
SAMPLE_COUNT=1441
START_EPOCH="2026-01-01T00:00:00Z"

# recipe -> manifest filename
declare -A RECIPES=(
  [multi_planet_two_body]="multi_planet_two_body_1000_1day_60s.jsonl"
  [multi_planet_j2]="multi_planet_j2_1000_1day_60s.jsonl"
  [earth_j2_drag]="earth_j2_drag_1000_1day_60s.jsonl"
  [multi_planet_j2_third_body]="multi_planet_j2_third_body_1000_1day_60s.jsonl"
  [all_body_full_perturbations]="all_body_full_perturbations_1000_1day_60s.jsonl"
)

for recipe in "${!RECIPES[@]}"; do
  fname="${RECIPES[$recipe]}"
  manifest="${MANIFEST_DIR}/${fname}"
  mkdir -p "${BASE_DIR}/data/manifests"

  echo "=== Generating: ${recipe} ==="
  docker compose run --rm orbital_propagator manifest append \
    --manifest "${manifest}" \
    --recipe "${recipe}" \
    --count 1000 \
    --seed "${SEED}" \
    --duration-s "${DURATION_S}" \
    --sample-count "${SAMPLE_COUNT}" \
    --start-epoch-utc "${START_EPOCH}"
  echo ""
done

echo "Done. Manifests:"
ls -lh "${MANIFEST_DIR}/"
