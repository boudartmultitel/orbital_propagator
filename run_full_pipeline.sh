#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${BASE_DIR}/data"

# Container-internal paths (docker-compose mounts ./data to /shared/data)
CONTAINER_DATA_DIR="/shared/data"
MANIFEST_DIR="${CONTAINER_DATA_DIR}/manifests"
DATASET_DIR="${CONTAINER_DATA_DIR}/datasets"

SEED=42
DURATION_S=86400
SAMPLE_COUNT=1441
START_EPOCH="2026-01-01T00:00:00Z"
COUNT=1000

# Build recipe list from the YAML config
RECIPES=()
while IFS= read -r line; do
  RECIPES+=("$line")
done < <(python3 -c "
import yaml, sys
with open('${BASE_DIR}/orbital_propagator/src/orbital_propagator/configs/data_generation.yaml') as f:
    config = yaml.safe_load(f)
for name in sorted(config['dataset_recipes']):
    print(name)
")

echo "========================================================"
echo " Orbital Propagator - Full Dataset Generation Pipeline"
echo "========================================================"
echo "Recipes: ${#RECIPES[@]}"
echo "Seed: ${SEED}"
echo "Duration: ${DURATION_S}s (${SAMPLE_COUNT} samples)"
echo "Count per recipe: ${COUNT}"
echo "Manifest dir: ${MANIFEST_DIR}"
echo "Dataset dir:  ${DATASET_DIR}"
echo "========================================================"
echo ""

# Ensure directories exist
mkdir -p "${BASE_DIR}/manifests" "${BASE_DIR}/datasets"

run_command() {
  docker compose run --rm orbital_propagator "$@"
}

# Phase 1: Generate manifests
echo "========================================================"
echo " Phase 1: Generating manifests"
echo "========================================================"
for recipe in "${RECIPES[@]}"; do
  fname="${recipe}_${COUNT}_${DURATION_S}s_${SAMPLE_COUNT}pts.jsonl"
  host_manifest="${BASE_DIR}/data/manifests/${fname}"
  container_manifest="${MANIFEST_DIR}/${fname}"

  # Check if manifest already exists with correct count
  if [[ -f "${host_manifest}" ]]; then
    existing_lines=$(wc -l < "${host_manifest}")
    if [[ "${existing_lines}" -ge "${COUNT}" ]]; then
      echo "SKIP: ${recipe} (manifest exists with ${existing_lines} lines)"
      continue
    fi
    echo "RESUME: ${recipe} (manifest exists with ${existing_lines} lines, appending)"
  fi

  echo "GENERATING: ${recipe} -> ${fname}"
  run_command manifest append \
    --manifest "${container_manifest}" \
    --recipe "${recipe}" \
    --count "${COUNT}" \
    --seed "${SEED}" \
    --duration-s "${DURATION_S}" \
    --sample-count "${SAMPLE_COUNT}" \
    --start-epoch-utc "${START_EPOCH}"
  echo ""
done

# Phase 2: Validate all manifests
echo "========================================================"
echo " Phase 2: Validating manifests"
echo "========================================================"
for recipe in "${RECIPES[@]}"; do
  fname="${recipe}_${COUNT}_${DURATION_S}s_${SAMPLE_COUNT}pts.jsonl"
  host_manifest="${BASE_DIR}/data/manifests/${fname}"
  container_manifest="${MANIFEST_DIR}/${fname}"

  if [[ ! -f "${host_manifest}" ]]; then
    echo "SKIP: ${recipe} (manifest not found)"
    continue
  fi

  echo "VALIDATING: ${recipe}"
  run_command manifest validate --manifest "${container_manifest}"
  echo ""
done

# Phase 3: Build trajectories
echo "========================================================"
echo " Phase 3: Building trajectory datasets"
echo "========================================================"
for recipe in "${RECIPES[@]}"; do
  fname="${recipe}_${COUNT}_${DURATION_S}s_${SAMPLE_COUNT}pts.jsonl"
  host_manifest="${BASE_DIR}/data/manifests/${fname}"
  container_manifest="${MANIFEST_DIR}/${fname}"
  container_output="${DATASET_DIR}/${recipe}"

  if [[ ! -f "${host_manifest}" ]]; then
    echo "SKIP: ${recipe} (manifest not found)"
    continue
  fi

  mkdir -p "${container_output}"

  echo "BUILDING: ${recipe} -> ${container_output}"
  run_command manifest build \
    --manifest "${container_manifest}" \
    --output-dir "${container_output}" \
    --skip-existing
  echo ""
done

echo "========================================================"
echo " Done."
echo "========================================================"
echo ""
echo "Manifests:"
ls -lh "${BASE_DIR}/data/manifests/" 2>/dev/null || echo "(none)"
echo ""
echo "Trajectory datasets:"
ls -lhR "${BASE_DIR}/data/datasets/" 2>/dev/null || echo "(none found)"
