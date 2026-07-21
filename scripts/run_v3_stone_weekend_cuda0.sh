#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715
EXTRACT_PY=${EXTRACT_PY:-/usr/bin/python3}
BONITO_PY=${BONITO_PY:-/mnt/zzbnew/rnamodel/dengjingye/tools/conda/envs/bonito090_py39/bin/python}
GPU=${GPU:-0}
DEVICE=cuda:${GPU}
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

MANIFEST=${ROOT}/artifacts/manifests/v1_groupheldout_3000_seed42/group_split_manifest.csv
FROZEN_CONFIG=${ROOT}/config/experiment_stone_frozen_small.json
PFT_CONFIG=${ROOT}/config/experiment_stone_pft_weekend.json
SOURCE_CCF=${SOURCE_CCF:-/mnt/zzbnew/rnamodel/dengjingye/bacteria/zymo_10_0616/extract_signal}
MODEL_DIR=${MODEL_DIR:-/mnt/zzbnew/rnamodel/dengjingye/bacteria/data/models}
STONE_MODULE=${STONE_MODULE:-/mnt/zzbnew/rnamodel/shenhaojie/poregpt/poregpt/utils/signal.py}
RAW_CACHE=${RAW_CACHE:-${ROOT}/artifacts/cache/v3_stone_raw_v1_3000}
EMBED_CACHE=${EMBED_CACHE:-${ROOT}/artifacts/cache/v3_stone_bonito768_v1_3000}
RUN_ROOT=${RUN_ROOT:-${ROOT}/artifacts/runs/v3_stone_weekend}
SUMMARY_ROOT=${SUMMARY_ROOT:-${ROOT}/artifacts/summaries/v3_stone_weekend}
LOG_ROOT=${ROOT}/logs/v3_stone_weekend

RUN_CACHE=${RUN_CACHE:-1}
RUN_EMBED=${RUN_EMBED:-1}
RUN_HEADS=${RUN_HEADS:-1}
RUN_DEPTHS=${RUN_DEPTHS:-1}
RUN_OBJECTIVES=${RUN_OBJECTIVES:-1}
RUN_FINAL_EVAL=${RUN_FINAL_EVAL:-1}
FORCE=${FORCE:-0}

mkdir -p "${RAW_CACHE}" "${EMBED_CACHE}" "${RUN_ROOT}" "${SUMMARY_ROOT}" "${LOG_ROOT}"

log() { echo "[$(date '+%F %T')] $*"; }
trap 'status=$?; log "ERROR line=${LINENO} status=${status}. Inspect ${LOG_ROOT}."; exit ${status}' ERR

log "v3 stone weekend ablation started on ${DEVICE}."
log "Stone implementation: ${STONE_MODULE}"
log "Protocol: CCF-file-held-out 1800/600/600 per species; test reserved for selected models."
"${EXTRACT_PY}" -c 'import numpy, pyccf5, scipy; print("extract preflight: ok")'
"${BONITO_PY}" -c "import torch, bonito; assert torch.cuda.is_available(); print('cuda=', torch.cuda.get_device_name(${GPU}))"
test -s "${STONE_MODULE}"
test -s "${MODEL_DIR}/weights_0.tar"

if [[ "${RUN_CACHE}" == "1" && ( "${FORCE}" == "1" || ! -s "${RAW_CACHE}/cache_summary.json" ) ]]; then
  log "Building resumable legacy-stone raw chunk cache."
  "${EXTRACT_PY}" -u "${ROOT}/scripts/07_cache_groupheldout_raw_chunks.py" \
    --group-manifest "${MANIFEST}" \
    --output-dir "${RAW_CACHE}" \
    --source-ccf-root "${SOURCE_CCF}" \
    --signal-module-file "${STONE_MODULE}" \
    --signal-strategy stone \
    --max-workers 4 \
    > "${LOG_ROOT}/01_cache_stone_raw.log" 2>&1
fi
test -s "${RAW_CACHE}/cache_summary.json"
test -s "${RAW_CACHE}/raw_chunk_manifest.csv"

if [[ "${RUN_EMBED}" == "1" && ( "${FORCE}" == "1" || ! -s "${EMBED_CACHE}/cache_summary.json" ) ]]; then
  log "Encoding fresh stone Bonito 768D cache on ${DEVICE}."
  "${BONITO_PY}" -u "${ROOT}/scripts/08_cache_fresh_bonito_embeddings.py" \
    --raw-manifest "${RAW_CACHE}/raw_chunk_manifest.csv" \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${EMBED_CACHE}" \
    --device "${DEVICE}" \
    --num-shards 1 --shard-index 0 \
    > "${LOG_ROOT}/02_cache_stone_bonito768.log" 2>&1
  "${BONITO_PY}" "${ROOT}/scripts/08_cache_fresh_bonito_embeddings.py" \
    --raw-manifest "${RAW_CACHE}/raw_chunk_manifest.csv" \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${EMBED_CACHE}" \
    --merge-only \
    >> "${LOG_ROOT}/02_cache_stone_bonito768.log" 2>&1
fi
test -s "${EMBED_CACHE}/cache_summary.json"
test -s "${EMBED_CACHE}/bag_manifest.csv"

HEAD_SUMMARIES=()
for aggregation in mean attention transformer; do
  out="${RUN_ROOT}/frozen_${aggregation}"
  HEAD_SUMMARIES+=("${out}/summary.json")
  if [[ "${RUN_HEADS}" == "1" && ( "${FORCE}" == "1" || ! -s "${out}/summary.json" ) ]]; then
    log "Training frozen stone head: ${aggregation}."
    "${BONITO_PY}" -u "${ROOT}/scripts/04_train_small_signal_student.py" \
      --experiment "${FROZEN_CONFIG}" \
      --group-manifest "${EMBED_CACHE}/bag_manifest.csv" \
      --output-dir "${out}" \
      --mode ce --aggregation "${aggregation}" --max-chunks 64 \
      --device "${DEVICE}" --num-workers 2 --skip-test \
      > "${LOG_ROOT}/03_frozen_${aggregation}.log" 2>&1
  fi
done
"${BONITO_PY}" "${ROOT}/scripts/12_select_stone_candidates.py" \
  --summaries "${HEAD_SUMMARIES[@]}" \
  --output-dir "${SUMMARY_ROOT}/head_selection" \
  --selection-name frozen_aggregation \
  > "${LOG_ROOT}/04_select_head.log" 2>&1

HEAD_SELECTION=${SUMMARY_ROOT}/head_selection/selection.json
BEST_AGG=$("${BONITO_PY}" -c "import json; print(json.load(open('${HEAD_SELECTION}'))['best']['aggregation'])")
BEST_HEAD=$("${BONITO_PY}" -c "import json; print(json.load(open('${HEAD_SELECTION}'))['best']['checkpoint'])")
log "Selected frozen aggregation by validation: ${BEST_AGG}."

if [[ "${RUN_FINAL_EVAL}" == "1" && ( "${FORCE}" == "1" || ! -s "${RUN_ROOT}/selected_frozen_eval/summary.json" ) ]]; then
  "${BONITO_PY}" -u "${ROOT}/scripts/04_train_small_signal_student.py" \
    --experiment "${FROZEN_CONFIG}" \
    --group-manifest "${EMBED_CACHE}/bag_manifest.csv" \
    --output-dir "${RUN_ROOT}/selected_frozen_eval" \
    --mode ce --aggregation "${BEST_AGG}" --max-chunks 64 \
    --device "${DEVICE}" --num-workers 2 \
    --eval-only-checkpoint "${BEST_HEAD}" \
    > "${LOG_ROOT}/05_eval_selected_frozen.log" 2>&1
fi

DEPTH_SUMMARIES=()
for blocks in 1 2 3 5; do
  out="${RUN_ROOT}/pft_ce_blocks${blocks}"
  DEPTH_SUMMARIES+=("${out}/summary.json")
  if [[ "${RUN_DEPTHS}" == "1" && ( "${FORCE}" == "1" || ! -s "${out}/summary.json" ) ]]; then
    log "Training stone PFT depth=${blocks}, aggregation=${BEST_AGG}, objective=CE."
    "${BONITO_PY}" -u "${ROOT}/scripts/11_train_bonito_stone_objective.py" \
      --experiment "${PFT_CONFIG}" \
      --raw-manifest "${RAW_CACHE}/raw_chunk_manifest.csv" \
      --model-dir "${MODEL_DIR}" \
      --initial-student-checkpoint "${BEST_HEAD}" \
      --output-dir "${out}" \
      --objective ce --trainable-blocks "${blocks}" --aggregation "${BEST_AGG}" \
      --device "${DEVICE}" \
      > "${LOG_ROOT}/06_pft_ce_blocks${blocks}.log" 2>&1
  fi
done
"${BONITO_PY}" "${ROOT}/scripts/12_select_stone_candidates.py" \
  --summaries "${DEPTH_SUMMARIES[@]}" \
  --output-dir "${SUMMARY_ROOT}/depth_selection" \
  --selection-name pft_depth \
  > "${LOG_ROOT}/07_select_depth.log" 2>&1

DEPTH_SELECTION=${SUMMARY_ROOT}/depth_selection/selection.json
BEST_BLOCKS=$("${BONITO_PY}" -c "import json; print(json.load(open('${DEPTH_SELECTION}'))['best']['trainable_lstm_blocks'])")
BEST_DEPTH_SUMMARY=$("${BONITO_PY}" -c "import json; print(json.load(open('${DEPTH_SELECTION}'))['best']['summary'])")
HARD_PAIRS=$("${BONITO_PY}" -c "import json; print(json.load(open('${DEPTH_SELECTION}'))['hard_pairs_from_best_validation_confusion'])")
log "Selected PFT depth=${BEST_BLOCKS}; validation-derived hard pairs=${HARD_PAIRS}."

OBJECTIVE_SUMMARIES=("${BEST_DEPTH_SUMMARY}")
for objective in supcon mixup c2lr_mimix; do
  out="${RUN_ROOT}/pft_${objective}_blocks${BEST_BLOCKS}"
  OBJECTIVE_SUMMARIES+=("${out}/summary.json")
  if [[ "${RUN_OBJECTIVES}" == "1" && ( "${FORCE}" == "1" || ! -s "${out}/summary.json" ) ]]; then
    log "Training stone PFT objective=${objective}, blocks=${BEST_BLOCKS}."
    "${BONITO_PY}" -u "${ROOT}/scripts/11_train_bonito_stone_objective.py" \
      --experiment "${PFT_CONFIG}" \
      --raw-manifest "${RAW_CACHE}/raw_chunk_manifest.csv" \
      --model-dir "${MODEL_DIR}" \
      --initial-student-checkpoint "${BEST_HEAD}" \
      --output-dir "${out}" \
      --objective "${objective}" --trainable-blocks "${BEST_BLOCKS}" --aggregation "${BEST_AGG}" \
      --hard-pairs "${HARD_PAIRS}" --device "${DEVICE}" \
      > "${LOG_ROOT}/08_pft_${objective}_blocks${BEST_BLOCKS}.log" 2>&1
  fi
done
"${BONITO_PY}" "${ROOT}/scripts/12_select_stone_candidates.py" \
  --summaries "${OBJECTIVE_SUMMARIES[@]}" \
  --output-dir "${SUMMARY_ROOT}/objective_selection" \
  --selection-name pft_objective \
  > "${LOG_ROOT}/09_select_objective.log" 2>&1

OBJECTIVE_SELECTION=${SUMMARY_ROOT}/objective_selection/selection.json
BEST_OBJECTIVE=$("${BONITO_PY}" -c "import json; print(json.load(open('${OBJECTIVE_SELECTION}'))['best']['objective'])")
BEST_FINAL_BLOCKS=$("${BONITO_PY}" -c "import json; print(json.load(open('${OBJECTIVE_SELECTION}'))['best']['trainable_lstm_blocks'])")
BEST_FINAL_CHECKPOINT=$("${BONITO_PY}" -c "import json; print(json.load(open('${OBJECTIVE_SELECTION}'))['best']['checkpoint'])")
log "Selected final PFT by validation: objective=${BEST_OBJECTIVE}, blocks=${BEST_FINAL_BLOCKS}."

FINAL_EVAL=${RUN_ROOT}/selected_pft_eval
if [[ "${RUN_FINAL_EVAL}" == "1" && ( "${FORCE}" == "1" || ! -s "${FINAL_EVAL}/summary.json" ) ]]; then
  "${BONITO_PY}" -u "${ROOT}/scripts/11_train_bonito_stone_objective.py" \
    --experiment "${PFT_CONFIG}" \
    --raw-manifest "${RAW_CACHE}/raw_chunk_manifest.csv" \
    --model-dir "${MODEL_DIR}" \
    --initial-student-checkpoint "${BEST_HEAD}" \
    --output-dir "${FINAL_EVAL}" \
    --objective "${BEST_OBJECTIVE}" --trainable-blocks "${BEST_FINAL_BLOCKS}" --aggregation "${BEST_AGG}" \
    --hard-pairs "${HARD_PAIRS}" --device "${DEVICE}" \
    --eval-only-checkpoint "${BEST_FINAL_CHECKPOINT}" \
    > "${LOG_ROOT}/10_eval_selected_pft.log" 2>&1
fi
test -s "${FINAL_EVAL}/summary.json"

"${BONITO_PY}" "${ROOT}/scripts/13_summarize_stone_weekend.py" \
  --head-selection "${HEAD_SELECTION}" \
  --depth-selection "${DEPTH_SELECTION}" \
  --objective-selection "${OBJECTIVE_SELECTION}" \
  --final-eval "${FINAL_EVAL}/summary.json" \
  --output-dir "${SUMMARY_ROOT}/final" \
  --softcopyright-threshold 0.82 \
  > "${LOG_ROOT}/11_final_summary.log" 2>&1

if [[ ! -s "${SUMMARY_ROOT}/final/calibration/calibration.json" ]]; then
  "${BONITO_PY}" -m squiggle_species.cli calibrate \
    --predictions "${FINAL_EVAL}/val_predictions.csv" \
    --target-accuracy 0.90 \
    --min-coverage 0.50 \
    --min-per-class-coverage 0.50 \
    --min-accuracy-gain 0.01 \
    --output-dir "${SUMMARY_ROOT}/final/calibration" \
    > "${LOG_ROOT}/12_calibration.log" 2>&1
fi
if [[ ! -s "${SUMMARY_ROOT}/final/test_report/report_summary.json" ]]; then
  "${BONITO_PY}" -m squiggle_species.cli report \
    --predictions "${FINAL_EVAL}/test_predictions.csv" \
    --threshold-json "${SUMMARY_ROOT}/final/calibration/calibration.json" \
    --output-dir "${SUMMARY_ROOT}/final/test_report" \
    > "${LOG_ROOT}/13_test_report.log" 2>&1
fi

log "v3 stone weekend ablation complete."
cat "${SUMMARY_ROOT}/final/final_summary.json"
