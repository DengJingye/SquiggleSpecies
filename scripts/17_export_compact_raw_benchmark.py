#!/usr/bin/env python3
"""Export a portable standardized raw-chunk benchmark with bounded read bags."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


SPECIES = ("LB01", "LB06", "LB07", "LB08", "LB09", "LB12", "LB18", "LB11", "LB02")
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_path(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest.resolve().parent / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--source-raw-manifest", type=Path, required=True)
    parser.add_argument("--preprocessing-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_chunks < 1:
        raise ValueError("max-chunks must be positive")
    benchmark_rows = load_rows(args.benchmark_manifest)
    source_rows = load_rows(args.source_raw_manifest)
    source_by_read: dict[str, dict[str, str]] = {}
    for row in source_rows:
        if row["read_id"] in source_by_read:
            raise ValueError(f"Duplicate read_id in source raw manifest: {row['read_id']}")
        source_by_read[row["read_id"]] = row
    missing = [row["read_id"] for row in benchmark_rows if row["read_id"] not in source_by_read]
    if missing:
        raise ValueError(f"Benchmark has {len(missing)} reads absent from raw cache; examples={missing[:3]}")

    profile = json.loads(args.preprocessing_profile.read_text())
    source_hash = file_sha256(args.source_raw_manifest)
    benchmark_hash = file_sha256(args.benchmark_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(
        benchmark_rows,
        key=lambda row: (SPECIES.index(row["barcode"]), SPLIT_ORDER[row["split"]], int(row["split_order"])),
    )
    by_species: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ordered:
        by_species[row["barcode"]].append(row)

    output_rows: list[dict[str, object]] = []
    chunk_files = []
    for species in SPECIES:
        rows = by_species.get(species, [])
        if not rows:
            raise ValueError(f"Benchmark has no rows for {species}")
        selected_counts = [min(int(source_by_read[row["read_id"]]["raw_n_chunks"]), args.max_chunks) for row in rows]
        total_chunks = sum(selected_counts)
        first_source = source_by_read[rows[0]["read_id"]]
        first_array = np.load(resolve_path(first_source["raw_chunk_path"], args.source_raw_manifest), mmap_mode="r")
        if first_array.ndim != 2:
            raise ValueError(f"Expected 2D raw chunks, got {first_array.shape}")
        signal_length = int(first_array.shape[1])
        output_path = chunks_dir / f"{species}_raw_chunks.npy"
        temporary = output_path.with_suffix(".npy.tmp")
        output_array = open_memmap(temporary, mode="w+", dtype=np.float16, shape=(total_chunks, signal_length))
        source_arrays: dict[str, np.ndarray] = {}
        cursor = 0
        for row, selected_count in zip(rows, selected_counts):
            source = source_by_read[row["read_id"]]
            source_path = resolve_path(source["raw_chunk_path"], args.source_raw_manifest)
            path_key = str(source_path)
            if path_key not in source_arrays:
                source_arrays[path_key] = np.load(source_path, mmap_mode="r")
            array = source_arrays[path_key]
            start = int(source["raw_chunk_start"])
            n_chunks = int(source["raw_n_chunks"])
            indices = (
                np.arange(n_chunks, dtype=np.int64)
                if n_chunks <= args.max_chunks
                else np.linspace(0, n_chunks - 1, num=args.max_chunks, dtype=np.int64)
            )
            selected_chunks = np.asarray(array[start + indices], dtype=np.float16)
            if selected_chunks.shape != (selected_count, signal_length):
                raise ValueError(f"Unexpected chunk shape for {row['read_id']}: {selected_chunks.shape}")
            output_array[cursor : cursor + selected_count] = selected_chunks
            output_rows.append(
                {
                    "split": row["split"],
                    "split_order": row["split_order"],
                    "barcode": row["barcode"],
                    "label": row["label"],
                    "read_id": row["read_id"],
                    "raw_chunk_path": str(Path("chunks") / output_path.name),
                    "raw_chunk_start": cursor,
                    "raw_n_chunks": selected_count,
                    "source_n_chunks": n_chunks,
                    "ccf_file": row["ccf_file"],
                    "preprocessing_profile_id": profile["profile_id"],
                    "chunk_selection": "uniform_eval_v1",
                }
            )
            cursor += selected_count
        output_array.flush()
        del output_array
        os.replace(temporary, output_path)
        chunk_files.append(
            {
                "species": species,
                "path": str(output_path.relative_to(args.output_dir)),
                "shape": [total_chunks, signal_length],
                "dtype": "float16",
                "sha256": file_sha256(output_path),
            }
        )

    manifest_path = args.output_dir / "raw_benchmark_manifest.csv"
    fields = [
        "split", "split_order", "barcode", "label", "read_id", "raw_chunk_path", "raw_chunk_start",
        "raw_n_chunks", "source_n_chunks", "ccf_file", "preprocessing_profile_id", "chunk_selection",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    shutil.copyfile(args.preprocessing_profile, args.output_dir / "preprocessing_profile.json")
    summary = {
        "status": "complete",
        "benchmark_manifest": str(args.benchmark_manifest.resolve()),
        "benchmark_manifest_sha256": benchmark_hash,
        "source_raw_manifest": str(args.source_raw_manifest.resolve()),
        "source_raw_manifest_sha256": source_hash,
        "portable_raw_manifest": str(manifest_path.resolve()),
        "portable_raw_manifest_sha256": file_sha256(manifest_path),
        "preprocessing_profile_id": profile["profile_id"],
        "preprocessing_profile_sha256": file_sha256(args.preprocessing_profile),
        "max_chunks_per_read": args.max_chunks,
        "n_reads": len(output_rows),
        "n_chunks": sum(int(row["raw_n_chunks"]) for row in output_rows),
        "chunk_files": chunk_files,
    }
    (args.output_dir / "raw_benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
