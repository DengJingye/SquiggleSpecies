#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715
PY=${PY:-/mnt/zzbnew/rnamodel/dengjingye/tools/conda/envs/bonito090_py39/bin/python}
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

RESOURCES=${ROOT}/config/resources.json
EXPERIMENT=${ROOT}/config/experiment_small.json
AUDIT_DIR=${ROOT}/artifacts/audits/v0_legacy_protocol
MANIFEST_DIR=${ROOT}/artifacts/manifests/v1_groupheldout_3000_seed42

mkdir -p "${ROOT}/logs" "${AUDIT_DIR}" "${MANIFEST_DIR}"

echo "[$(date '+%F %T')] new0715 v0 audit started."
"${PY}" "${ROOT}/scripts/00_preflight.py" \
  --resources "${RESOURCES}" \
  --output "${AUDIT_DIR}/preflight.json" \
  --metadata-only

"${PY}" "${ROOT}/scripts/01_audit_legacy_protocol.py" \
  --resources "${RESOURCES}" \
  --output-dir "${AUDIT_DIR}"

"${PY}" "${ROOT}/scripts/02_build_group_split.py" \
  --resources "${RESOURCES}" \
  --experiment "${EXPERIMENT}" \
  --output-dir "${MANIFEST_DIR}"

echo "[$(date '+%F %T')] v0 audit complete."
echo "Group manifest: ${MANIFEST_DIR}/group_split_manifest.csv"
