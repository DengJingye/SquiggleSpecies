from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .preprocessing import process_signal, restore_physical_signal
from .utils import file_sha256, save_json


def discover_ccf5(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() != ".ccf5":
            raise ValueError(f"Expected a .ccf5 file: {source}")
        return [source.resolve()]
    if not source.is_dir():
        raise FileNotFoundError(f"CCF5 input does not exist: {source}")
    files = sorted(item.resolve() for item in source.rglob("*.ccf5"))
    if not files:
        raise FileNotFoundError(f"No .ccf5 files found under {source}")
    return files


def text_read_id(value) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def chunk_signal(
    signal: np.ndarray,
    *,
    discard_first: int,
    chunk_size: int,
    overlap: int,
) -> np.ndarray:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if signal.size < discard_first + chunk_size:
        return np.empty((0, chunk_size), dtype=np.float32)
    stride = chunk_size - overlap
    retained = np.asarray(signal[discard_first:], dtype=np.float32)
    return sliding_window_view(retained, chunk_size)[::stride]


def preprocess_read(
    record: dict,
    *,
    profile_id: str,
    discard_first: int,
    chunk_size: int,
    overlap: int,
) -> np.ndarray:
    physical = restore_physical_signal(record)
    normalized = process_signal(physical, profile_id)
    chunks = chunk_signal(
        normalized,
        discard_first=discard_first,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if chunks.size and not np.all(np.isfinite(chunks)):
        raise ValueError("Preprocessed chunks contain non-finite values")
    return chunks


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def cache_ccf5(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    profile_id: str,
    discard_first: int,
    chunk_size: int,
    overlap: int,
    reads_per_part: int = 1000,
    max_reads: int = 0,
) -> dict:
    try:
        import pyccf5 as slow5
    except ImportError as error:
        raise RuntimeError(
            "cache-ccf requires pyccf5; run it in the CCF5 extraction environment"
        ) from error
    if reads_per_part < 1:
        raise ValueError("reads_per_part must be positive")
    files = discover_ccf5(input_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "profile_id": profile_id,
        "discard_first": int(discard_first),
        "chunk_size": int(chunk_size),
        "overlap": int(overlap),
        "reads_per_part": int(reads_per_part),
        "max_reads": int(max_reads),
    }
    config_signature = file_sha256(_write_config(output_dir, config))
    all_rows: list[dict] = []
    file_summaries = []
    remaining = max_reads

    for file_index, ccf_path in enumerate(files):
        if max_reads > 0 and remaining <= 0:
            break
        file_dir = output_dir / "files" / f"{file_index:04d}_{ccf_path.stem}"
        summary_path = file_dir / "cache_meta.json"
        source_signature = {
            "path": str(ccf_path),
            "size": ccf_path.stat().st_size,
            "mtime_ns": ccf_path.stat().st_mtime_ns,
        }
        if summary_path.exists():
            existing = json.loads(summary_path.read_text())
            if (
                existing.get("status") == "complete"
                and existing.get("config_signature") == config_signature
                and existing.get("source") == source_signature
            ):
                with Path(existing["manifest"]).open(newline="") as handle:
                    all_rows.extend(csv.DictReader(handle))
                file_summaries.append(existing)
                if max_reads > 0:
                    remaining -= int(existing["eligible_reads"])
                continue
        if file_dir.exists():
            shutil.rmtree(file_dir)
        file_dir.mkdir(parents=True)

        rows: list[dict] = []
        part_chunks: list[np.ndarray] = []
        part_rows: list[dict] = []
        scanned = eligible = too_short = invalid = part_index = 0

        def flush_part() -> None:
            nonlocal part_index
            if not part_rows:
                return
            array = np.concatenate(part_chunks, axis=0).astype(np.float16, copy=False)
            part_path = file_dir / f"raw_chunks_part{part_index:05d}.npy"
            _atomic_save_array(part_path, array)
            cursor = 0
            for row, chunks in zip(part_rows, part_chunks):
                row["raw_chunk_path"] = str(part_path)
                row["raw_chunk_start"] = cursor
                row["raw_n_chunks"] = len(chunks)
                cursor += len(chunks)
                rows.append(row)
            part_chunks.clear()
            part_rows.clear()
            part_index += 1

        reader = slow5.Open(str(ccf_path), "r", DEBUG=0)
        try:
            for read_index, record in enumerate(reader.seq_reads(aux="all")):
                if max_reads > 0 and remaining <= 0:
                    break
                scanned += 1
                try:
                    chunks = preprocess_read(
                        record,
                        profile_id=profile_id,
                        discard_first=discard_first,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                except Exception:
                    invalid += 1
                    continue
                if not len(chunks):
                    too_short += 1
                    continue
                part_chunks.append(np.asarray(chunks, dtype=np.float32))
                part_rows.append(
                    {
                        "split": "test",
                        "split_order": eligible,
                        "barcode": "unlabeled",
                        "label": -1,
                        "read_id": text_read_id(record["read_id"]),
                        "ccf_file": str(ccf_path),
                        "source_read_index": read_index,
                        "preprocessing_profile_id": profile_id,
                    }
                )
                eligible += 1
                if max_reads > 0:
                    remaining -= 1
                if len(part_rows) >= reads_per_part:
                    flush_part()
            flush_part()
        finally:
            reader.close()

        fields = [
            "split",
            "split_order",
            "barcode",
            "label",
            "read_id",
            "ccf_file",
            "source_read_index",
            "preprocessing_profile_id",
            "raw_chunk_path",
            "raw_chunk_start",
            "raw_n_chunks",
        ]
        manifest_path = file_dir / "raw_chunk_manifest.csv"
        _atomic_write_csv(manifest_path, rows, fields)
        summary = {
            "status": "complete",
            "source": source_signature,
            "config_signature": config_signature,
            "manifest": str(manifest_path),
            "scanned_reads": scanned,
            "eligible_reads": eligible,
            "too_short_reads": too_short,
            "invalid_reads": invalid,
            "parts": part_index,
        }
        save_json(summary_path, summary)
        all_rows.extend(rows)
        file_summaries.append(summary)

    read_ids = [row["read_id"] for row in all_rows]
    if len(set(read_ids)) != len(read_ids):
        raise ValueError("Duplicate read_id values detected across CCF5 files")
    for order, row in enumerate(all_rows):
        row["split_order"] = order
    merged_manifest = output_dir / "raw_chunk_manifest.csv"
    if all_rows:
        _atomic_write_csv(merged_manifest, all_rows, list(all_rows[0]))
    summary = {
        "status": "complete",
        "mode": "persistent_raw_chunk_cache",
        "input": str(Path(input_path).resolve()),
        "output_dir": str(output_dir),
        "config": config,
        "config_signature": config_signature,
        "files": file_summaries,
        "total_reads": len(all_rows),
        "raw_chunk_manifest": str(merged_manifest),
    }
    save_json(output_dir / "cache_summary.json", summary)
    return summary


def _write_config(output_dir: Path, config: dict) -> Path:
    path = output_dir / "cache_config.json"
    save_json(path, config)
    return path
