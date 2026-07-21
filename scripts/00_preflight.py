#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from squiggle_species.utils import read_json, save_json


REQUIRED_RESOURCES = (
    "legacy_split_manifest",
    "legacy_v10_bag_manifest",
    "bonito_chunk_cache",
    "sequence_kmer_cache",
    "signal_v10_metrics",
    "external_v28_summary",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight new0715 resources and Python environment.")
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true", help="Skip slow ML imports during data-only audit.")
    args = parser.parse_args()

    resources = read_json(args.resources)
    checks = {}
    failures = []
    for key, value in resources.items():
        path = Path(value)
        checks[key] = {"path": str(path), "exists": path.exists(), "is_dir": path.is_dir()}
        if key in REQUIRED_RESOURCES and not path.exists():
            failures.append(f"missing required resource: {key}={path}")

    cache_dir = Path(resources["bonito_chunk_cache"])
    meta_files = sorted(cache_dir.glob("*/*_chunk_cache_meta.json")) if cache_dir.exists() else []
    if not meta_files:
        failures.append(f"no chunk cache metadata found under {cache_dir}")

    ml_environment = {"status": "not_checked"}
    if not args.metadata_only:
        import numpy as np
        import sklearn
        import torch

        ml_environment = {
            "status": "checked",
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    report = {
        "status": "ok" if not failures else "failed",
        "python": sys.executable,
        "python_version": platform.python_version(),
        "ml_environment": ml_environment,
        "chunk_cache_meta_count": len(meta_files),
        "resource_checks": checks,
        "failures": failures,
    }
    save_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
