#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715
PY=${PY:-/mnt/zzbnew/rnamodel/dengjingye/tools/conda/envs/bonito090_py39/bin/python}
GPU=${GPU:-0}
DEVICE=cuda:${GPU}
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

RAW_MANIFEST=${ROOT}/artifacts/cache/v3_stone_raw_v1_3000/raw_chunk_manifest.csv
BAG_MANIFEST=${ROOT}/artifacts/cache/v3_stone_bonito768_v1_3000/bag_manifest.csv
DISTANCE_MATRIX=/mnt/zzbnew/rnamodel/dengjingye/bacteria/zymo_10_0616/zymo9_work/08_figure_data/signal_space_diagnostics/reference_genome_5mer_cosine_distance.csv
FROZEN_TEMPLATE=${ROOT}/config/experiment_stone_frozen_small.json
PFT_TEMPLATE=${ROOT}/config/experiment_stone_pft_weekend.json
SUBSET_ROOT=${ROOT}/artifacts/manifests/v5_class_count_scaling
RUN_ROOT=${ROOT}/artifacts/runs/v5_class_count_scaling
SUMMARY_ROOT=${ROOT}/artifacts/summaries/v5_class_count_scaling
LOG_ROOT=${ROOT}/logs/v5_class_count_scaling
MODEL_DIR=${MODEL_DIR:-/mnt/zzbnew/rnamodel/dengjingye/bacteria/data/models}
K9_SUMMARY=${ROOT}/artifacts/runs/v3_stone_weekend/selected_pft_eval/summary.json
CLASS_COUNTS=${CLASS_COUNTS:-"5 6 7 8"}
FORCE=${FORCE:-0}
ALLOW_CONCURRENT=${ALLOW_CONCURRENT:-0}

mkdir -p "${SUBSET_ROOT}" "${RUN_ROOT}" "${SUMMARY_ROOT}" "${LOG_ROOT}"
log() { echo "[$(date '+%F %T')] $*"; }
trap 'status=$?; log "ERROR line=${LINENO} status=${status}. Inspect ${LOG_ROOT}."; exit ${status}' ERR

if [[ "${GPU}" == "0" && "${ALLOW_CONCURRENT}" != "1" ]] && pgrep -f "run_v4_closure_cuda0.sh|v4_closure/.+11_train_bonito_stone_objective" >/dev/null; then
  log "v4 is still using cuda:0. Select GPU=1 or wait for v4 to finish."
  exit 3
fi

"${PY}" -c "import torch, bonito; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(${GPU}))"
test -s "${MODEL_DIR}/weights_0.tar"
test -s "${K9_SUMMARY}"

if [[ "${FORCE}" == "1" || ! -s "${SUBSET_ROOT}/class_count_protocol.json" ]]; then
  log "Building pre-registered nested class subsets without using validation/test predictions."
  "${PY}" "${ROOT}/scripts/19_build_class_count_subsets.py" \
    --raw-manifest "${RAW_MANIFEST}" \
    --bag-manifest "${BAG_MANIFEST}" \
    --distance-matrix "${DISTANCE_MATRIX}" \
    --frozen-experiment-template "${FROZEN_TEMPLATE}" \
    --pft-experiment-template "${PFT_TEMPLATE}" \
    --output-dir "${SUBSET_ROOT}" \
    > "${LOG_ROOT}/01_build_subsets.log" 2>&1
fi

for k in ${CLASS_COUNTS}; do
  subset=${SUBSET_ROOT}/k${k}
  frozen=${RUN_ROOT}/k${k}/frozen_transformer
  final=${RUN_ROOT}/k${k}/pft_best
  mkdir -p "${frozen}" "${final}"
  hard_pairs=LB01:LB12
  if (( k >= 7 )); then
    hard_pairs=${hard_pairs},LB11:LB02
  fi
  if (( k >= 8 )); then
    hard_pairs=${hard_pairs},LB07:LB09
  fi

  if [[ "${FORCE}" == "1" || ! -s "${frozen}/summary.json" ]]; then
    log "k=${k}: training class-specific frozen Transformer head."
    "${PY}" -u "${ROOT}/scripts/04_train_small_signal_student.py" \
      --experiment "${subset}/frozen_experiment.json" \
      --group-manifest "${subset}/bag_manifest.csv" \
      --output-dir "${frozen}" \
      --mode ce --aggregation transformer --max-chunks 64 \
      --device "${DEVICE}" --num-workers 2 --skip-test \
      > "${LOG_ROOT}/02_k${k}_frozen.log" 2>&1
  fi

  if [[ "${FORCE}" == "1" || ! -s "${final}/summary.json" ]]; then
    log "k=${k}: training the fixed best route, Stone + Bonito PFT-3 + Transformer + C2LR/MI-Mix."
    "${PY}" -u "${ROOT}/scripts/11_train_bonito_stone_objective.py" \
      --experiment "${subset}/pft_experiment.json" \
      --raw-manifest "${subset}/raw_chunk_manifest.csv" \
      --model-dir "${MODEL_DIR}" \
      --initial-student-checkpoint "${frozen}/model.pth" \
      --output-dir "${final}" \
      --objective c2lr_mimix --trainable-blocks 3 --aggregation transformer \
      --hard-pairs "${hard_pairs}" --device "${DEVICE}" --evaluate-test \
      > "${LOG_ROOT}/03_k${k}_pft_best.log" 2>&1
  fi
done

log "Summarizing separately trained k=5..9 models."
"${PY}" "${ROOT}/scripts/20_summarize_class_count_scaling.py" \
  --protocol "${SUBSET_ROOT}/class_count_protocol.json" \
  --run-root "${RUN_ROOT}" \
  --k9-summary "${K9_SUMMARY}" \
  --output-dir "${SUMMARY_ROOT}" \
  > "${LOG_ROOT}/04_summary.log" 2>&1

log "v5 class-count scaling complete."
cat "${SUMMARY_ROOT}/class_count_scaling_summary.json"
