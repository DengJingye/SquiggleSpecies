#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715
BONITO_PY=${BONITO_PY:-/mnt/zzbnew/rnamodel/dengjingye/tools/conda/envs/bonito090_py39/bin/python}
GPU=${GPU:-0}
DEVICE=cuda:${GPU}
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

CONFIG=${ROOT}/config/experiment_stone_pft_weekend.json
RAW_MANIFEST=${ROOT}/artifacts/cache/v3_stone_raw_v1_3000/raw_chunk_manifest.csv
MODEL_DIR=${MODEL_DIR:-/mnt/zzbnew/rnamodel/dengjingye/bacteria/data/models}
V3_ROOT=${ROOT}/artifacts/runs/v3_stone_weekend
BASELINE_DIR=${V3_ROOT}/selected_pft_eval
BASELINE_SUMMARY=${BASELINE_DIR}/summary.json
HEAD_SELECTION=${ROOT}/artifacts/summaries/v3_stone_weekend/head_selection/selection.json
DEPTH_SELECTION=${ROOT}/artifacts/summaries/v3_stone_weekend/depth_selection/selection.json
RUN_ROOT=${ROOT}/artifacts/runs/v4_closure
SUMMARY_ROOT=${ROOT}/artifacts/summaries/v4_closure
LOG_ROOT=${ROOT}/logs/v4_closure
SMOKE_MANIFEST=${ROOT}/artifacts/smoke/v4_crossfile_raw_manifest.csv

RUN_SMOKE=${RUN_SMOKE:-1}
SMOKE_ONLY=${SMOKE_ONLY:-0}
RUN_REPEATS=${RUN_REPEATS:-1}
RUN_ROBUST=${RUN_ROBUST:-1}
RUN_SUMMARY=${RUN_SUMMARY:-1}
FORCE=${FORCE:-0}

mkdir -p "${RUN_ROOT}" "${SUMMARY_ROOT}" "${LOG_ROOT}" "$(dirname "${SMOKE_MANIFEST}")"
log() { echo "[$(date '+%F %T')] $*"; }
trap 'status=$?; log "ERROR line=${LINENO} status=${status}. Inspect ${LOG_ROOT}."; exit ${status}' ERR

log "v4 bounded closure run started on ${DEVICE}."
log "Scope: two fixed v3 seed repeats plus one cross-file GroupDRO candidate; no architecture search."
"${BONITO_PY}" -c "import torch, bonito; assert torch.cuda.is_available(); print('cuda=', torch.cuda.get_device_name(${GPU}))"
test -s "${RAW_MANIFEST}"
test -s "${BASELINE_SUMMARY}"
test -s "${MODEL_DIR}/weights_0.tar"

BEST_HEAD=$("${BONITO_PY}" -c "import json; print(json.load(open('${HEAD_SELECTION}'))['best']['checkpoint'])")
HARD_PAIRS=$("${BONITO_PY}" -c "import json; print(json.load(open('${DEPTH_SELECTION}'))['hard_pairs_from_best_validation_confusion'])")
test -s "${BEST_HEAD}"

if [[ ! -s "${SMOKE_MANIFEST}" ]]; then
  "${BONITO_PY}" "${ROOT}/scripts/15_build_v4_smoke_manifest.py" \
    --input "${RAW_MANIFEST}" --output "${SMOKE_MANIFEST}" \
    > "${LOG_ROOT}/00_build_smoke_manifest.log" 2>&1
fi

SMOKE_OUT=${RUN_ROOT}/smoke_crossfile_groupdro
if [[ "${RUN_SMOKE}" == "1" && ( "${FORCE}" == "1" || ! -s "${SMOKE_OUT}/summary.json" ) ]]; then
  log "Running v4 cross-file sampler/loss GPU smoke."
  "${BONITO_PY}" -u "${ROOT}/scripts/11_train_bonito_stone_objective.py" \
    --experiment "${CONFIG}" --raw-manifest "${SMOKE_MANIFEST}" \
    --model-dir "${MODEL_DIR}" --initial-student-checkpoint "${BEST_HEAD}" \
    --output-dir "${SMOKE_OUT}" --objective crossfile_groupdro \
    --trainable-blocks 3 --aggregation transformer --seed-override 42 \
    --epochs 1 --max-chunks 2 --batch-size 4 --eval-batch-size 6 --chunk-microbatch 4 \
    --num-workers 0 --device "${DEVICE}" \
    > "${LOG_ROOT}/01_smoke_crossfile_groupdro.log" 2>&1
fi
test -s "${SMOKE_OUT}/summary.json"
log "GPU smoke complete."
if [[ "${SMOKE_ONLY}" == "1" ]]; then
  log "Smoke-only mode finished."
  exit 0
fi

REPEAT_SUMMARIES=()
for seed in 3407 2026; do
  out=${RUN_ROOT}/v3_c2lr_mimix_blocks3_seed${seed}
  REPEAT_SUMMARIES+=("${out}/summary.json")
  if [[ "${RUN_REPEATS}" == "1" && ( "${FORCE}" == "1" || ! -s "${out}/summary.json" ) ]]; then
    log "Running fixed v3 reproducibility seed=${seed}."
    "${BONITO_PY}" -u "${ROOT}/scripts/11_train_bonito_stone_objective.py" \
      --experiment "${CONFIG}" --raw-manifest "${RAW_MANIFEST}" \
      --model-dir "${MODEL_DIR}" --initial-student-checkpoint "${BEST_HEAD}" \
      --output-dir "${out}" --objective c2lr_mimix --trainable-blocks 3 \
      --aggregation transformer --hard-pairs "${HARD_PAIRS}" --seed-override "${seed}" \
      --device "${DEVICE}" --evaluate-test \
      > "${LOG_ROOT}/02_repeat_seed${seed}.log" 2>&1
  fi
  test -s "${out}/summary.json"
done

ROBUST_OUT=${RUN_ROOT}/crossfile_groupdro_blocks3_seed42
if [[ "${RUN_ROBUST}" == "1" && ( "${FORCE}" == "1" || ! -s "${ROBUST_OUT}/summary.json" ) ]]; then
  log "Training the single v4 cross-file GroupDRO candidate; validation only."
  "${BONITO_PY}" -u "${ROOT}/scripts/11_train_bonito_stone_objective.py" \
    --experiment "${CONFIG}" --raw-manifest "${RAW_MANIFEST}" \
    --model-dir "${MODEL_DIR}" --initial-student-checkpoint "${BEST_HEAD}" \
    --output-dir "${ROBUST_OUT}" --objective crossfile_groupdro --trainable-blocks 3 \
    --aggregation transformer --seed-override 42 --device "${DEVICE}" \
    > "${LOG_ROOT}/03_crossfile_groupdro_seed42.log" 2>&1
fi
test -s "${ROBUST_OUT}/summary.json"

summarize() {
  local extra=()
  if [[ -s "${RUN_ROOT}/crossfile_groupdro_selected_eval/summary.json" ]]; then
    extra+=(--robust-eval-summary "${RUN_ROOT}/crossfile_groupdro_selected_eval/summary.json")
  fi
  "${BONITO_PY}" "${ROOT}/scripts/14_summarize_v4_closure.py" \
    --baseline-summary "${BASELINE_SUMMARY}" --baseline-dir "${BASELINE_DIR}" \
    --repeat-summaries "${REPEAT_SUMMARIES[@]}" \
    --robust-summary "${ROBUST_OUT}/summary.json" \
    --raw-manifest "${RAW_MANIFEST}" --experiment "${CONFIG}" \
    --output-dir "${SUMMARY_ROOT}" "${extra[@]}"
}

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  summarize > "${LOG_ROOT}/04_v4_gate_summary.log" 2>&1
fi
test -s "${SUMMARY_ROOT}/gate_decision.json"
PROMOTE=$("${BONITO_PY}" -c "import json; print(int(json.load(open('${SUMMARY_ROOT}/gate_decision.json'))['promote_to_test']))")

ROBUST_EVAL=${RUN_ROOT}/crossfile_groupdro_selected_eval
if [[ "${PROMOTE}" == "1" && ( "${FORCE}" == "1" || ! -s "${ROBUST_EVAL}/summary.json" ) ]]; then
  log "Both validation gates passed; evaluating the frozen robust candidate on test once."
  "${BONITO_PY}" -u "${ROOT}/scripts/11_train_bonito_stone_objective.py" \
    --experiment "${CONFIG}" --raw-manifest "${RAW_MANIFEST}" \
    --model-dir "${MODEL_DIR}" --initial-student-checkpoint "${BEST_HEAD}" \
    --output-dir "${ROBUST_EVAL}" --objective crossfile_groupdro --trainable-blocks 3 \
    --aggregation transformer --seed-override 42 --device "${DEVICE}" \
    --eval-only-checkpoint "${ROBUST_OUT}/model.pth" \
    > "${LOG_ROOT}/05_eval_crossfile_groupdro.log" 2>&1
else
  log "Cross-file candidate did not pass both validation gates, or its frozen test evaluation already exists."
fi

summarize > "${LOG_ROOT}/06_v4_final_summary.log" 2>&1
log "v4 bounded closure run complete."
cat "${SUMMARY_ROOT}/v4_final_summary.json"
