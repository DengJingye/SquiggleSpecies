#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/mnt/zzbnew/rnamodel/dengjingye/tools/conda/envs/bonito090_py39/bin/python}"
MANIFEST="${MANIFEST:?Set MANIFEST to a portable raw benchmark manifest}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/artifacts/runs/v3_stone_weekend/pft_c2lr_mimix_blocks3/model.pth}"
BONITO_MODEL_DIR="${BONITO_MODEL_DIR:-/mnt/zzbnew/rnamodel/dengjingye/bacteria/data/models}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/output/toolkit_benchmark_eval}"
DEVICE="${DEVICE:-cuda:0}"
DATASET_ROLE="${DATASET_ROLE:-development-benchmark}"
PREPROCESSING_PROFILE_ID="${PREPROCESSING_PROFILE_ID:-legacy-stone-v1}"

for path in "${PY}" "${MANIFEST}" "${CHECKPOINT}" "${BONITO_MODEL_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required path does not exist: ${path}" >&2
    exit 1
  fi
done

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${DATASET_ROLE}" > "${OUTPUT_DIR}/DATASET_ROLE.txt"
echo "[benchmark] role=${DATASET_ROLE} manifest=${MANIFEST}"

for split in val test; do
  "${PY}" -m squiggle_species predict-raw-cache \
    --manifest "${MANIFEST}" \
    --checkpoint "${CHECKPOINT}" \
    --bonito-model-dir "${BONITO_MODEL_DIR}" \
    --split "${split}" \
    --device "${DEVICE}" \
    --expected-preprocessing-profile "${PREPROCESSING_PROFILE_ID}" \
    --output "${OUTPUT_DIR}/${split}_predictions.csv"
done

"${PY}" -m squiggle_species calibrate \
  --predictions "${OUTPUT_DIR}/val_predictions.csv" \
  --target-accuracy 0.90 \
  --min-coverage 0.50 \
  --min-per-class-coverage 0.50 \
  --min-accuracy-gain 0.01 \
  --output-dir "${OUTPUT_DIR}/calibration"

"${PY}" -m squiggle_species report \
  --predictions "${OUTPUT_DIR}/test_predictions.csv" \
  --threshold-json "${OUTPUT_DIR}/calibration/calibration.json" \
  --output-dir "${OUTPUT_DIR}/test_report"

echo "[benchmark] Complete: ${OUTPUT_DIR}"
