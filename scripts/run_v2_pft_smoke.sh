#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715
EXTRACT_PY=${EXTRACT_PY:-/usr/bin/python3}
BONITO_PY=${BONITO_PY:-/mnt/zzbnew/rnamodel/dengjingye/tools/conda/envs/bonito090_py39/bin/python}
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

MANIFEST=${ROOT}/artifacts/manifests/v1_groupheldout_3000_seed42/group_split_manifest.csv
PFT_EXPERIMENT=${ROOT}/config/experiment_pft_small.json
FROZEN_EXPERIMENT=${ROOT}/config/experiment_apple_frozen_smoke.json
RAW_CACHE=${ROOT}/artifacts/cache/v2_pft_apple_smoke_raw_chunks
EMBED_CACHE=${ROOT}/artifacts/cache/v2_pft_apple_smoke_bonito768
MODEL_DIR=/mnt/zzbnew/rnamodel/dengjingye/bacteria/data/models
SOURCE_CCF=/mnt/zzbnew/rnamodel/dengjingye/bacteria/zymo_10_0616/extract_signal
SIGNAL_MODULE=/mnt/zzbnew/poregpt/shared/signal.py
RUN_ROOT=${ROOT}/artifacts/runs/v2_pft_apple_smoke
FROZEN_DIR=${RUN_ROOT}/apple_frozen_ce
SUMMARY_ROOT=${ROOT}/artifacts/summaries/v2_pft_apple_smoke

mkdir -p "${ROOT}/logs" "${RUN_ROOT}" "${SUMMARY_ROOT}"
echo "[$(date '+%F %T')] v2 latest-apple Bonito PFT smoke started."
"${EXTRACT_PY}" -c 'import numpy, pyccf5, scipy; print("extract preflight: ok")'
"${BONITO_PY}" -c 'import bonito, sklearn, torch; print(f"bonito preflight: torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}"); assert torch.cuda.is_available()'

"${EXTRACT_PY}" -u "${ROOT}/scripts/07_cache_groupheldout_raw_chunks.py" \
  --group-manifest "${MANIFEST}" \
  --output-dir "${RAW_CACHE}" \
  --source-ccf-root "${SOURCE_CCF}" \
  --signal-module-file "${SIGNAL_MODULE}" \
  --signal-strategy apple \
  --reads-per-split-per-species 2 \
  --max-workers 2

"${BONITO_PY}" -u "${ROOT}/scripts/08_cache_fresh_bonito_embeddings.py" \
  --raw-manifest "${RAW_CACHE}/raw_chunk_manifest.csv" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${EMBED_CACHE}" \
  --device cuda:0
"${BONITO_PY}" "${ROOT}/scripts/08_cache_fresh_bonito_embeddings.py" \
  --raw-manifest "${RAW_CACHE}/raw_chunk_manifest.csv" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${EMBED_CACHE}" \
  --merge-only

"${BONITO_PY}" -u "${ROOT}/scripts/04_train_small_signal_student.py" \
  --experiment "${FROZEN_EXPERIMENT}" \
  --group-manifest "${EMBED_CACHE}/bag_manifest.csv" \
  --output-dir "${FROZEN_DIR}" \
  --mode ce \
  --device cuda:0 \
  --num-workers 0 \
  > "${ROOT}/logs/v2_pft_apple_smoke_frozen.log" 2>&1

"${BONITO_PY}" -u "${ROOT}/scripts/09_train_bonito_partial_finetune.py" \
  --experiment "${PFT_EXPERIMENT}" \
  --raw-manifest "${RAW_CACHE}/raw_chunk_manifest.csv" \
  --model-dir "${MODEL_DIR}" \
  --initial-student-checkpoint "${FROZEN_DIR}/model.pth" \
  --output-dir "${RUN_ROOT}/pft_a" \
  --mode pft_a --device cuda:0 --epochs 2 --batch-size 2 --eval-batch-size 2 \
  --max-chunks 2 --chunk-microbatch 4 --num-workers 0 \
  > "${ROOT}/logs/v2_pft_apple_smoke_a.log" 2>&1

"${BONITO_PY}" -u "${ROOT}/scripts/09_train_bonito_partial_finetune.py" \
  --experiment "${PFT_EXPERIMENT}" \
  --raw-manifest "${RAW_CACHE}/raw_chunk_manifest.csv" \
  --model-dir "${MODEL_DIR}" \
  --initial-student-checkpoint "${FROZEN_DIR}/model.pth" \
  --output-dir "${RUN_ROOT}/pft_b" \
  --mode pft_b --device cuda:0 --epochs 2 --batch-size 2 --eval-batch-size 2 \
  --max-chunks 2 --chunk-microbatch 4 --num-workers 0 \
  > "${ROOT}/logs/v2_pft_apple_smoke_b.log" 2>&1

"${BONITO_PY}" "${ROOT}/scripts/10_compare_pft_results.py" \
  --experiment "${PFT_EXPERIMENT}" \
  --frozen-summary "${FROZEN_DIR}/summary.json" \
  --pft-a "${RUN_ROOT}/pft_a/summary.json" \
  --pft-b "${RUN_ROOT}/pft_b/summary.json" \
  --output-dir "${SUMMARY_ROOT}" \
  --smoke
echo "[$(date '+%F %T')] v2 latest-apple PFT smoke complete."
