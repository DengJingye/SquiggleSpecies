#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715
PY=${PY:-/mnt/zzbnew/rnamodel/dengjingye/tools/conda/envs/bonito090_py39/bin/python}
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
RESOURCES=${ROOT}/config/resources.json
EXPERIMENT=${ROOT}/config/experiment_smoke.json
MANIFEST_DIR=${ROOT}/artifacts/manifests/smoke_groupheldout
GROUP_MANIFEST=${MANIFEST_DIR}/group_split_manifest.csv
TEACHER_DIR=${ROOT}/artifacts/teacher/smoke_sequence_teacher
RUN_ROOT=${ROOT}/artifacts/runs/smoke_groupheldout
SUMMARY_DIR=${ROOT}/artifacts/summaries/smoke_groupheldout

mkdir -p "${MANIFEST_DIR}" "${TEACHER_DIR}" "${RUN_ROOT}/signal_ce" "${RUN_ROOT}/signal_kd" "${SUMMARY_DIR}" "${ROOT}/logs"

"${PY}" "${ROOT}/scripts/02_build_group_split.py" \
  --resources "${RESOURCES}" \
  --experiment "${EXPERIMENT}" \
  --output-dir "${MANIFEST_DIR}"

"${PY}" -u "${ROOT}/scripts/03_train_group_sequence_teacher.py" \
  --resources "${RESOURCES}" \
  --experiment "${EXPERIMENT}" \
  --group-manifest "${GROUP_MANIFEST}" \
  --output-dir "${TEACHER_DIR}" \
  --device cpu \
  --epochs 2 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --patience 2 \
  --num-workers 0

"${PY}" -u "${ROOT}/scripts/04_train_small_signal_student.py" \
  --experiment "${EXPERIMENT}" \
  --group-manifest "${GROUP_MANIFEST}" \
  --output-dir "${RUN_ROOT}/signal_ce" \
  --mode ce \
  --device cpu \
  --num-workers 0

"${PY}" -u "${ROOT}/scripts/04_train_small_signal_student.py" \
  --experiment "${EXPERIMENT}" \
  --group-manifest "${GROUP_MANIFEST}" \
  --output-dir "${RUN_ROOT}/signal_kd" \
  --mode kd \
  --teacher-train "${TEACHER_DIR}/teacher_train.npz" \
  --device cpu \
  --num-workers 0

"${PY}" "${ROOT}/scripts/05_compare_small_results.py" \
  --experiment "${EXPERIMENT}" \
  --ce-summary "${RUN_ROOT}/signal_ce/summary.json" \
  --kd-summary "${RUN_ROOT}/signal_kd/summary.json" \
  --teacher-metrics "${TEACHER_DIR}/teacher_metrics.json" \
  --output-dir "${SUMMARY_DIR}"

echo "Smoke test complete: ${SUMMARY_DIR}/decision_summary.json"
