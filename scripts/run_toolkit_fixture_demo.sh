#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MANIFEST="${MANIFEST:-${ROOT}/artifacts/benchmarks/zymo9_fixture_v1/raw_bundle/raw_benchmark_manifest.csv}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/output/toolkit_fixture_demo}"
export DATASET_ROLE="fixture-software-smoke-only"

echo "[fixture] This run validates software wiring only; its metrics are not scientific results."
bash "${ROOT}/scripts/run_toolkit_benchmark_eval.sh"

cat > "${OUTPUT_DIR}/FIXTURE_ONLY.txt" <<'EOF'
This output is an installation and end-to-end software smoke test.
It must not be used to claim model accuracy or compare scientific methods.
EOF

echo "[fixture] Complete: ${OUTPUT_DIR}"
