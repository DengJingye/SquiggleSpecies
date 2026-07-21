#!/usr/bin/env python3
"""Build a standalone, checksum-audited benchmark data handoff directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


TIERS = {
    "fixture": "zymo9_fixture_v1",
    "benchmark-mini": "zymo9_benchmark_mini_v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing destination has a different size: {destination}")
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tier_summaries = {}
    for public_name, source_name in TIERS.items():
        source = args.benchmark_root / source_name / "raw_bundle"
        source_summary = json.loads((source / "raw_benchmark_summary.json").read_text())
        destination = args.output_dir / public_name
        for relative in (Path("raw_benchmark_manifest.csv"), Path("preprocessing_profile.json")):
            materialize(source / relative, destination / relative, args.mode)
        for chunk in source_summary["chunk_files"]:
            relative = Path(chunk["path"])
            materialize(source / relative, destination / relative, args.mode)
        tier_summaries[public_name] = {
            "n_reads": source_summary["n_reads"],
            "n_chunks": source_summary["n_chunks"],
            "max_chunks_per_read": source_summary["max_chunks_per_read"],
            "preprocessing_profile_id": source_summary["preprocessing_profile_id"],
            "manifest": f"{public_name}/raw_benchmark_manifest.csv",
        }

    readme = """# Squiggle Species Benchmark Data v1

This directory is distributed separately from the source-code repository.

- `fixture`: 54 reads, software installation and end-to-end smoke only.
- `benchmark-mini`: 4500 reads, development regression, calibration, inference and figure generation.
- Both tiers use `legacy-stone-v1` and at most 16 uniformly sampled chunks per read.
- Neither tier replaces the 27000-read file-held-out benchmark used for formal scientific results.
- D6306 Zymo10 is an external known-9/OOD benchmark and is not included here.

The raw manifests use relative chunk paths. Verify all files with `sha256sum -c checksums.sha256`.
Redistribution remains subject to confirmation of the source-data and model licenses.
"""
    (args.output_dir / "README_DATA.md").write_text(readme)
    payload_bytes = sum(path.stat().st_size for path in args.output_dir.rglob("*") if path.is_file())
    handoff = {
        "status": "complete",
        "data_bundle": "SquiggleSpecies_Benchmark_Data_v1",
        "tiers": tier_summaries,
        "formal_result_warning": "benchmark-mini and fixture do not replace benchmark-full",
        "atlas_required": False,
        "external_mixture_included": False,
        "payload_bytes_before_checksums": payload_bytes,
        "payload_mib_before_checksums": payload_bytes / (1024**2),
    }
    (args.output_dir / "handoff_manifest.json").write_text(json.dumps(handoff, indent=2) + "\n")

    checksum_rows = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_rows.append(f"{sha256(path)}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "checksums.sha256").write_text("\n".join(checksum_rows) + "\n")
    print(json.dumps(handoff, indent=2), flush=True)


if __name__ == "__main__":
    main()
