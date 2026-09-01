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
echo "========================================================"
echo ""

# Ensure directories exist
mkdir -p "${BASE_DIR}/data/manifests" "${BASE_DIR}/data/datasets"

# Phase 1: Generate manifests (skip if already complete)
echo "========================================================"
echo " Phase 1: Generating manifests"
echo "========================================================"
for recipe in "${RECIPES[@]}"; do
  fname="${recipe}_${COUNT}_${DURATION_S}s_${SAMPLE_COUNT}pts.jsonl"
  host_manifest="${BASE_DIR}/data/manifests/${fname}"

  if [[ -f "${host_manifest}" ]]; then
    existing_lines=$(wc -l < "${host_manifest}")
    if [[ "${existing_lines}" -ge "${COUNT}" ]]; then
      echo "SKIP: ${recipe} (already has ${existing_lines} lines)"
      continue
    fi
    # Check for duplicate trajectory IDs - if found, regenerate
    duplicate_check=$(python3 -c "
lines = open('${host_manifest}').readlines()
ids = [l.strip().split('\"trajectory_id\":\"')[1].split('\"')[0] for l in lines if '\"trajectory_id\":\"' in l]
if len(ids) != len(set(ids)):
    print('duplicate')
else:
    print('ok')
" 2>/dev/null || echo "error")
    if [[ "${duplicate_check}" == "duplicate" ]]; then
      echo "REGEN: ${recipe} (duplicate IDs found, regenerating)"
      rm -f "${host_manifest}"
    else
      echo "RESUME: ${recipe} (has ${existing_lines} lines, appending)"
    fi
  fi

  echo "GENERATING: ${recipe} -> ${fname}"
  docker compose run --rm orbital_propagator manifest append \
    --manifest "${MANIFEST_DIR}/${fname}" \
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

  if [[ ! -f "${host_manifest}" ]]; then
    echo "SKIP: ${recipe} (manifest not found)"
    continue
  fi

  echo "VALIDATING: ${recipe}"
  docker compose run --rm orbital_propagator manifest validate \
    --manifest "${MANIFEST_DIR}/${fname}"
  echo ""
done

# Phase 3: Build trajectories (skip if already complete)
echo "========================================================"
echo " Phase 3: Building trajectory datasets"
echo "========================================================"
for recipe in "${RECIPES[@]}"; do
  fname="${recipe}_${COUNT}_${DURATION_S}s_${SAMPLE_COUNT}pts.jsonl"
  host_manifest="${BASE_DIR}/data/manifests/${fname}"
  container_output="${DATASET_DIR}/${recipe}"
  host_output="${BASE_DIR}/data/datasets/${recipe}"

  if [[ ! -f "${host_manifest}" ]]; then
    echo "SKIP: ${recipe} (manifest not found)"
    continue
  fi

  # Check if trajectories are already complete
  if [[ -d "${host_output}" ]] && [[ -f "${host_output}/metadata.json" ]]; then
    existing_count=$(python3 -c "
import json
with open('${host_output}/metadata.json') as f:
    meta = json.load(f)
print(meta.get('trajectory_count', 0))
" 2>/dev/null || echo "0")
    if [[ "${existing_count}" -ge "${COUNT}" ]]; then
      echo "SKIP: ${recipe} (already has ${existing_count} trajectories)"
      continue
    fi
    echo "RESUME: ${recipe} (has ${existing_count} trajectories, continuing)"
  fi

  mkdir -p "${host_output}"

  echo "BUILDING: ${recipe} -> ${recipe}/"
  docker compose run --rm orbital_propagator manifest build \
    --manifest "${MANIFEST_DIR}/${fname}" \
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
