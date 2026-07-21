#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715
PY=${PY:-/mnt/zzbnew/rnamodel/dengjingye/tools/conda/envs/bonito090_py39/bin/python}
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

RESOURCES=${ROOT}/config/resources.json
EXPERIMENT=${ROOT}/config/experiment_small.json
MANIFEST_DIR=${ROOT}/artifacts/manifests/v1_groupheldout_3000_seed42
GROUP_MANIFEST=${MANIFEST_DIR}/group_split_manifest.csv
TEACHER_DIR=${ROOT}/artifacts/teacher/v1_groupheldout_sequence_teacher
RUN_ROOT=${ROOT}/artifacts/runs/v1_groupheldout_small
CE_DIR=${RUN_ROOT}/signal_ce
KD_DIR=${RUN_ROOT}/signal_crossmodal_kd
SUMMARY_DIR=${ROOT}/artifacts/summaries/v1_groupheldout_small
RUN_AUDIT=${RUN_AUDIT:-1}
RUN_TEACHER=${RUN_TEACHER:-1}
RUN_CE=${RUN_CE:-1}
RUN_KD=${RUN_KD:-1}
FORCE=${FORCE:-0}

mkdir -p "${ROOT}/logs" "${TEACHER_DIR}" "${CE_DIR}" "${KD_DIR}" "${SUMMARY_DIR}"

echo "[$(date '+%F %T')] new0715 v1 group-held-out CE/KD started."
echo "Python: ${PY}"

"${PY}" -c 'import sklearn, torch; print(f"torch={torch.__version__} sklearn={sklearn.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}"); assert torch.cuda.is_available() and torch.cuda.device_count() >= 2'

if [[ "${RUN_AUDIT}" == "1" ]]; then
  bash "${ROOT}/scripts/run_v0_audit.sh"
fi

if [[ ! -s "${GROUP_MANIFEST}" ]]; then
  echo "ERROR: group manifest missing: ${GROUP_MANIFEST}" >&2
  exit 2
fi

if [[ "${RUN_TEACHER}" == "1" ]]; then
  if [[ "${FORCE}" == "1" || ! -s "${TEACHER_DIR}/teacher_metrics.json" ]]; then
    echo "[$(date '+%F %T')] Training group-held-out sequence teacher on GPU0."
    "${PY}" -u "${ROOT}/scripts/03_train_group_sequence_teacher.py" \
      --resources "${RESOURCES}" \
      --experiment "${EXPERIMENT}" \
      --group-manifest "${GROUP_MANIFEST}" \
      --output-dir "${TEACHER_DIR}" \
      --device cuda:0 \
      > "${ROOT}/logs/v1_sequence_teacher_gpu0.log" 2>&1
  else
    echo "[$(date '+%F %T')] Reusing completed sequence teacher."
  fi
fi

if [[ ! -s "${TEACHER_DIR}/teacher_train.npz" ]]; then
  echo "ERROR: teacher_train.npz missing: ${TEACHER_DIR}/teacher_train.npz" >&2
  exit 2
fi

pids=()
names=()
if [[ "${RUN_CE}" == "1" && ( "${FORCE}" == "1" || ! -s "${CE_DIR}/summary.json" ) ]]; then
  echo "[$(date '+%F %T')] Starting CE student on GPU0."
  "${PY}" -u "${ROOT}/scripts/04_train_small_signal_student.py" \
    --experiment "${EXPERIMENT}" \
    --group-manifest "${GROUP_MANIFEST}" \
    --output-dir "${CE_DIR}" \
    --mode ce \
    --device cuda:0 \
    > "${ROOT}/logs/v1_ce_gpu0.log" 2>&1 &
  pids+=("$!")
  names+=("ce")
fi

if [[ "${RUN_KD}" == "1" && ( "${FORCE}" == "1" || ! -s "${KD_DIR}/summary.json" ) ]]; then
  echo "[$(date '+%F %T')] Starting cross-modal KD student on GPU1."
  "${PY}" -u "${ROOT}/scripts/04_train_small_signal_student.py" \
    --experiment "${EXPERIMENT}" \
    --group-manifest "${GROUP_MANIFEST}" \
    --output-dir "${KD_DIR}" \
    --mode kd \
    --teacher-train "${TEACHER_DIR}/teacher_train.npz" \
    --device cuda:1 \
    > "${ROOT}/logs/v1_kd_gpu1.log" 2>&1 &
  pids+=("$!")
  names+=("kd")
fi

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "ERROR: ${names[$index]} worker failed; inspect its log." >&2
    status=1
  fi
done
if [[ "${status}" != "0" ]]; then
  exit "${status}"
fi

if [[ ! -s "${CE_DIR}/summary.json" || ! -s "${KD_DIR}/summary.json" ]]; then
  echo "ERROR: CE/KD summaries are incomplete." >&2
  exit 2
fi

"${PY}" "${ROOT}/scripts/05_compare_small_results.py" \
  --experiment "${EXPERIMENT}" \
  --ce-summary "${CE_DIR}/summary.json" \
  --kd-summary "${KD_DIR}/summary.json" \
  --teacher-metrics "${TEACHER_DIR}/teacher_metrics.json" \
  --output-dir "${SUMMARY_DIR}"

echo "[$(date '+%F %T')] v1 complete."
cat "${SUMMARY_DIR}/decision_summary.json"
