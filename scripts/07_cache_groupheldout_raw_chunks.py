#!/usr/bin/env python3
"""Extract only the v1 group-held-out reads into resumable raw chunk memmaps."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyccf5 as slow5
from numpy.lib.format import open_memmap
from numpy.lib.stride_tricks import sliding_window_view


SPECIES = ["LB01", "LB06", "LB07", "LB08", "LB09", "LB12", "LB18", "LB11", "LB02"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-ccf-root", type=Path, required=True)
    parser.add_argument("--signal-module-file", type=Path, required=True)
    parser.add_argument("--signal-strategy", default="stone")
    parser.add_argument("--discard-first", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=6000)
    parser.add_argument("--overlap", type=int, default=3000)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--reads-per-split-per-species",
        type=int,
        default=0,
        help="Smoke-only cap; zero keeps the complete input manifest.",
    )
    return parser.parse_args()


def read_manifest(path: Path, cap: int) -> list[dict[str, str]]:
    selected = []
    counts: dict[tuple[str, str], int] = defaultdict(int)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["barcode"], row["split"])
            if cap > 0 and counts[key] >= cap:
                continue
            counts[key] += 1
            selected.append(row)
    return selected


def load_signal_function(path: Path):
    spec = importlib.util.spec_from_file_location("new0715_signal_standardization", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import signal module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.nanopore_process_signal


def restore_physical_signal(read: dict) -> np.ndarray:
    raw = np.asarray(read["signal"])
    if "lvdsmid" in read:
        return ((raw.astype(np.float32) - float(read["lvdsmid"])) * float(read["unit"])).astype(np.float32)
    signal = float(read["K"]) * float(read["scale"]) * (
        raw.astype(np.uint16).astype(np.float32) + float(read["offset"])
    ) + float(read["B"])
    return signal.astype(np.float32, copy=False)


def text(value) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def process_species(payload: dict) -> dict:
    barcode = payload["barcode"]
    rows = payload["rows"]
    output_dir = Path(payload["output_dir"])
    species_dir = output_dir / barcode
    species_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = int(payload["chunk_size"])
    stride = chunk_size - int(payload["overlap"])
    discard_first = int(payload["discard_first"])
    signal_strategy = payload["signal_strategy"]
    source_root = Path(payload["source_ccf_root"])
    signal_function = load_signal_function(Path(payload["signal_module_file"]))

    cursor = 0
    cache_rows = []
    for row in rows:
        n_chunks = int(row["n_chunks"])
        cache_row = dict(row)
        cache_row.update(
            {
                "raw_chunk_path": str(species_dir / f"{barcode}_raw_chunks.npy"),
                "raw_chunk_start": cursor,
                "raw_n_chunks": n_chunks,
                "source_ccf_file": str(source_root / barcode / Path(row["ccf_file"]).name),
            }
        )
        cache_rows.append(cache_row)
        cursor += n_chunks

    chunk_path = species_dir / f"{barcode}_raw_chunks.npy"
    manifest_path = species_dir / f"{barcode}_raw_chunk_manifest.csv"
    progress_path = species_dir / "progress.json"
    meta_path = species_dir / "cache_meta.json"
    if meta_path.exists() and manifest_path.exists() and chunk_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("status") == "complete" and int(meta.get("n_reads", -1)) == len(rows):
            return meta

    if chunk_path.exists():
        chunks = np.load(chunk_path, mmap_mode="r+")
        if chunks.shape != (cursor, chunk_size) or chunks.dtype != np.float16:
            raise ValueError(f"Existing raw cache shape/dtype mismatch: {chunk_path} {chunks.shape} {chunks.dtype}")
    else:
        chunks = open_memmap(chunk_path, mode="w+", dtype=np.float16, shape=(cursor, chunk_size))

    completed_files = set()
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        completed_files = set(progress.get("completed_files", []))

    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in cache_rows:
        by_file[row["source_ccf_file"]].append(row)

    for file_index, (source_file, file_rows) in enumerate(sorted(by_file.items()), 1):
        if source_file in completed_files:
            continue
        if not Path(source_file).exists():
            raise FileNotFoundError(source_file)
        reader = slow5.Open(source_file, "r", DEBUG=0)
        try:
            requested_ids = [row["read_id"] for row in file_rows]
            fetched = list(reader.get_read_list_multi(requested_ids, threads=1, batchsize=1, aux="all"))
            fetched_by_id = {text(read.get("read_id", "")): read for read in fetched}
            missing = sorted(set(requested_ids) - set(fetched_by_id))
            if missing:
                raise KeyError(f"Missing {len(missing)} reads in {source_file}: {missing[:3]}")
            for row in file_rows:
                read = fetched_by_id[row["read_id"]]
                signal = restore_physical_signal(read)
                normalized = np.asarray(signal_function(signal, signal_strategy), dtype=np.float32)
                if normalized.size < discard_first + chunk_size:
                    raise ValueError(f"Selected read became ineligible: {row['read_id']} length={normalized.size}")
                normalized = normalized[discard_first:]
                read_chunks = sliding_window_view(normalized, chunk_size)[::stride]
                expected = int(row["raw_n_chunks"])
                if len(read_chunks) != expected:
                    raise ValueError(
                        f"Chunk-count mismatch for {row['read_id']}: extracted={len(read_chunks)} expected={expected}"
                    )
                start = int(row["raw_chunk_start"])
                chunks[start : start + expected] = read_chunks.astype(np.float16, copy=False)
        finally:
            reader.close()
        chunks.flush()
        completed_files.add(source_file)
        atomic_json(
            progress_path,
            {
                "barcode": barcode,
                "completed_files": sorted(completed_files),
                "total_files": len(by_file),
                "n_reads": len(rows),
                "n_chunks": cursor,
            },
        )
        print(f"[{barcode}] {file_index}/{len(by_file)} files cached", flush=True)

    source_fields = list(rows[0])
    output_fields = source_fields + ["raw_chunk_path", "raw_chunk_start", "raw_n_chunks", "source_ccf_file"]
    write_csv(manifest_path, cache_rows, output_fields)
    meta = {
        "status": "complete",
        "barcode": barcode,
        "n_reads": len(rows),
        "n_chunks": cursor,
        "n_files": len(by_file),
        "dtype": "float16",
        "shape": [cursor, chunk_size],
        "chunk_path": str(chunk_path),
        "manifest_path": str(manifest_path),
        "signal_strategy": signal_strategy,
        "signal_module_file": payload["signal_module_file"],
        "discard_first": discard_first,
        "chunk_size": chunk_size,
        "overlap": int(payload["overlap"]),
    }
    atomic_json(meta_path, meta)
    return meta


def main() -> None:
    args = parse_args()
    if args.overlap >= args.chunk_size:
        raise ValueError("overlap must be smaller than chunk-size")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.group_manifest, args.reads_per_split_per_species)
    by_species: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_species[row["barcode"]].append(row)
    missing_species = sorted(set(SPECIES) - set(by_species))
    if missing_species:
        raise ValueError(f"Manifest is missing species: {missing_species}")

    payloads = [
        {
            "barcode": barcode,
            "rows": by_species[barcode],
            "output_dir": str(args.output_dir),
            "source_ccf_root": str(args.source_ccf_root),
            "signal_module_file": str(args.signal_module_file),
            "signal_strategy": args.signal_strategy,
            "discard_first": args.discard_first,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
        }
        for barcode in SPECIES
    ]
    metas = []
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(payloads))) as executor:
        futures = {executor.submit(process_species, payload): payload["barcode"] for payload in payloads}
        for future in as_completed(futures):
            barcode = futures[future]
            meta = future.result()
            metas.append(meta)
            print(f"[{barcode}] complete: reads={meta['n_reads']} chunks={meta['n_chunks']}", flush=True)

    merged_rows = []
    for barcode in SPECIES:
        path = args.output_dir / barcode / f"{barcode}_raw_chunk_manifest.csv"
        with path.open(newline="") as handle:
            merged_rows.extend(csv.DictReader(handle))
    merged_rows.sort(key=lambda row: (SPECIES.index(row["barcode"]), row["split"], int(row["split_order"])))
    fieldnames = list(merged_rows[0])
    merged_manifest = args.output_dir / "raw_chunk_manifest.csv"
    write_csv(merged_manifest, merged_rows, fieldnames)
    summary = {
        "status": "complete",
        "source_manifest": str(args.group_manifest.resolve()),
        "raw_chunk_manifest": str(merged_manifest),
        "reads_per_split_per_species_cap": args.reads_per_split_per_species,
        "n_reads": len(merged_rows),
        "n_chunks": sum(int(row["raw_n_chunks"]) for row in merged_rows),
        "species": metas,
    }
    atomic_json(args.output_dir / "cache_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
