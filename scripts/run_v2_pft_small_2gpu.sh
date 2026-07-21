#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715
EXTRACT_PY=${EXTRACT_PY:-/usr/bin/python3}
BONITO_PY=${BONITO_PY:-/mnt/zzbnew/rnamodel/dengjingye/tools/conda/envs/bonito090_py39/bin/python}
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

MANIFEST=${ROOT}/artifacts/manifests/v1_groupheldout_3000_seed42/group_split_manifest.csv
PFT_EXPERIMENT=${ROOT}/config/experiment_pft_small.json
FROZEN_EXPERIMENT=${ROOT}/config/experiment_apple_frozen_small.json
RAW_CACHE=${RAW_CACHE:-${ROOT}/artifacts/cache/v2_pft_raw_chunks_v1_3000_apple}
EMBED_CACHE=${EMBED_CACHE:-${ROOT}/artifacts/cache/v2_pft_bonito768_v1_3000_apple}
MODEL_DIR=${MODEL_DIR:-/mnt/zzbnew/rnamodel/dengjingye/bacteria/data/models}
SOURCE_CCF=${SOURCE_CCF:-/mnt/zzbnew/rnamodel/dengjingye/bacteria/zymo_10_0616/extract_signal}
SIGNAL_MODULE=${SIGNAL_MODULE:-/mnt/zzbnew/poregpt/shared/signal.py}
RUN_ROOT=${ROOT}/artifacts/runs/v2_pft_apple_small
FROZEN_DIR=${RUN_ROOT}/apple_frozen_ce
SUMMARY_ROOT=${ROOT}/artifacts/summaries/v2_pft_apple_small
RUN_CACHE=${RUN_CACHE:-1}
RUN_EMBED=${RUN_EMBED:-1}
RUN_FROZEN=${RUN_FROZEN:-1}
RUN_TRAIN=${RUN_TRAIN:-1}
FORCE=${FORCE:-0}

mkdir -p "${ROOT}/logs" "${RUN_ROOT}" "${SUMMARY_ROOT}" "${RAW_CACHE}" "${EMBED_CACHE}"
echo "[$(date '+%F %T')] new0715 v2 latest-apple Bonito PFT started."
echo "Standardization: /mnt/zzbnew/poregpt/shared/signal.py + apple"
echo "Protocol: file-held-out train/val/test=1800/600/600 per species; max_chunks=8"
"${EXTRACT_PY}" -c 'import numpy, pyccf5, scipy; print("extract preflight: ok")'
"${BONITO_PY}" -c 'import bonito, sklearn, torch; print(f"bonito preflight: torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}"); assert torch.cuda.is_available() and torch.cuda.device_count() >= 2'

if [[ "${RUN_CACHE}" == "1" ]]; then
  echo "[$(date '+%F %T')] Extracting/resuming latest-apple raw chunks."
  "${EXTRACT_PY}" -u "${ROOT}/scripts/07_cache_groupheldout_raw_chunks.py" \
    --group-manifest "${MANIFEST}" \
    --output-dir "${RAW_CACHE}" \
    --source-ccf-root "${SOURCE_CCF}" \
    --signal-module-file "${SIGNAL_MODULE}" \
    --signal-strategy apple \
    --max-workers 4 \
    > "${ROOT}/logs/v2_pft_apple_raw_cache.log" 2>&1
fi
if [[ ! -s "${RAW_CACHE}/cache_summary.json" || ! -s "${RAW_CACHE}/raw_chunk_manifest.csv" ]]; then
  echo "ERROR: latest-apple raw cache is incomplete." >&2
  exit 2
fi

if [[ "${RUN_EMBED}" == "1" ]]; then
  echo "[$(date '+%F %T')] Building fresh latest-apple Bonito 768D cache on two GPUs."
  pids=()
  for shard in 0 1; do
    "${BONITO_PY}" -u "${ROOT}/scripts/08_cache_fresh_bonito_embeddings.py" \
      --raw-manifest "${RAW_CACHE}/raw_chunk_manifest.csv" \
      --model-dir "${MODEL_DIR}" \
      --output-dir "${EMBED_CACHE}" \
      --device "cuda:${shard}" \
      --num-shards 2 --shard-index "${shard}" \
      > "${ROOT}/logs/v2_pft_apple_embed_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then status=1; fi
  done
  if [[ "${status}" != "0" ]]; then
    echo "ERROR: fresh latest-apple Bonito cache worker failed." >&2
    exit 1
  fi
  "${BONITO_PY}" "${ROOT}/scripts/08_cache_fresh_bonito_embeddings.py" \
    --raw-manifest "${RAW_CACHE}/raw_chunk_manifest.csv" \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${EMBED_CACHE}" \
    --merge-only
fi
if [[ ! -s "${EMBED_CACHE}/cache_summary.json" || ! -s "${EMBED_CACHE}/bag_manifest.csv" ]]; then
  echo "ERROR: fresh latest-apple Bonito 768D cache is incomplete." >&2
  exit 2
fi

if [[ "${RUN_FROZEN}" == "1" && ( "${FORCE}" == "1" || ! -s "${FROZEN_DIR}/summary.json" ) ]]; then
  echo "[$(date '+%F %T')] Training fresh latest-apple frozen CE baseline on GPU0."
  "${BONITO_PY}" -u "${ROOT}/scripts/04_train_small_signal_student.py" \
    --experiment "${FROZEN_EXPERIMENT}" \
    --group-manifest "${EMBED_CACHE}/bag_manifest.csv" \
    --output-dir "${FROZEN_DIR}" \
    --mode ce --device cuda:0 --num-workers 2 \
    > "${ROOT}/logs/v2_pft_apple_frozen_gpu0.log" 2>&1
fi
if [[ ! -s "${FROZEN_DIR}/model.pth" || ! -s "${FROZEN_DIR}/summary.json" ]]; then
  echo "ERROR: fresh latest-apple frozen CE baseline is incomplete." >&2
  exit 2
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  pids=()
  names=()
  if [[ "${FORCE}" == "1" || ! -s "${RUN_ROOT}/pft_a/summary.json" ]]; then
    echo "[$(date '+%F %T')] Starting latest-apple PFT-A on GPU0."
    "${BONITO_PY}" -u "${ROOT}/scripts/09_train_bonito_partial_finetune.py" \
      --experiment "${PFT_EXPERIMENT}" --raw-manifest "${RAW_CACHE}/raw_chunk_manifest.csv" \
      --model-dir "${MODEL_DIR}" --initial-student-checkpoint "${FROZEN_DIR}/model.pth" \
      --output-dir "${RUN_ROOT}/pft_a" --mode pft_a --device cuda:0 \
      > "${ROOT}/logs/v2_pft_apple_a_gpu0.log" 2>&1 &
    pids+=("$!"); names+=("pft_a")
  fi
  if [[ "${FORCE}" == "1" || ! -s "${RUN_ROOT}/pft_b/summary.json" ]]; then
    echo "[$(date '+%F %T')] Starting latest-apple PFT-B on GPU1."
    "${BONITO_PY}" -u "${ROOT}/scripts/09_train_bonito_partial_finetune.py" \
      --experiment "${PFT_EXPERIMENT}" --raw-manifest "${RAW_CACHE}/raw_chunk_manifest.csv" \
      --model-dir "${MODEL_DIR}" --initial-student-checkpoint "${FROZEN_DIR}/model.pth" \
      --output-dir "${RUN_ROOT}/pft_b" --mode pft_b --device cuda:1 \
      > "${ROOT}/logs/v2_pft_apple_b_gpu1.log" 2>&1 &
    pids+=("$!"); names+=("pft_b")
  fi
  status=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "ERROR: ${names[$index]} failed; inspect its log." >&2
      status=1
    fi
  done
  if [[ "${status}" != "0" ]]; then exit "${status}"; fi
fi
if [[ ! -s "${RUN_ROOT}/pft_a/summary.json" || ! -s "${RUN_ROOT}/pft_b/summary.json" ]]; then
  echo "ERROR: latest-apple PFT summaries are incomplete." >&2
  exit 2
fi

"${BONITO_PY}" "${ROOT}/scripts/10_compare_pft_results.py" \
  --experiment "${PFT_EXPERIMENT}" \
  --frozen-summary "${FROZEN_DIR}/summary.json" \
  --pft-a "${RUN_ROOT}/pft_a/summary.json" \
  --pft-b "${RUN_ROOT}/pft_b/summary.json" \
  --output-dir "${SUMMARY_ROOT}"
echo "[$(date '+%F %T')] v2 latest-apple PFT complete."
cat "${SUMMARY_ROOT}/decision_summary.json"
