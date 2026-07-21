#!/usr/bin/env python3
"""Build deterministic class-balanced benchmark tiers from a file-held-out split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


SPECIES = ("LB01", "LB06", "LB07", "LB08", "LB09", "LB12", "LB18", "LB11", "LB02")
SPLITS = ("train", "val", "test")


def digest(seed: int, *values: str) -> str:
    text = "|".join((str(seed), *values))
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def select_round_robin(rows: list[dict[str, str]], target: int, seed: int) -> list[dict[str, str]]:
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_file[row["ccf_file"]].append(row)
    groups = sorted(by_file, key=lambda value: digest(seed, "group", value))
    for group in groups:
        by_file[group].sort(key=lambda row: digest(seed, "read", row["read_id"]))
    selected: list[dict[str, str]] = []
    cursor = {group: 0 for group in groups}
    while len(selected) < target:
        advanced = False
        for group in groups:
            index = cursor[group]
            if index >= len(by_file[group]):
                continue
            selected.append(by_file[group][index])
            cursor[group] += 1
            advanced = True
            if len(selected) == target:
                break
        if not advanced:
            raise ValueError(f"Requested {target} rows but only {len(selected)} are available")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tier-name", default="benchmark-mini")
    parser.add_argument("--train-per-species", type=int, default=300)
    parser.add_argument("--val-per-species", type=int, default=100)
    parser.add_argument("--test-per-species", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.source_manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"split", "split_order", "barcode", "label", "read_id", "ccf_file"}
    missing = required - set(source_fields)
    if missing:
        raise ValueError(f"Source manifest is missing fields: {sorted(missing)}")

    targets = {
        "train": args.train_per_species,
        "val": args.val_per_species,
        "test": args.test_per_species,
    }
    selected: list[dict[str, str]] = []
    for species in SPECIES:
        for split in SPLITS:
            candidates = [row for row in rows if row["barcode"] == species and row["split"] == split]
            chosen = select_round_robin(candidates, targets[split], args.seed)
            for split_order, source_row in enumerate(chosen):
                row = dict(source_row)
                row["source_split_order"] = source_row["split_order"]
                row["split_order"] = str(split_order)
                selected.append(row)

    read_ids = [row["read_id"] for row in selected]
    if len(read_ids) != len(set(read_ids)):
        raise ValueError("Selected benchmark contains duplicate read_id values")
    groups = {
        split: {(row["barcode"], row["ccf_file"]) for row in selected if row["split"] == split}
        for split in SPLITS
    }
    group_overlap = (groups["train"] & groups["val"]) | (groups["train"] & groups["test"]) | (
        groups["val"] & groups["test"]
    )
    if group_overlap:
        raise ValueError(f"CCF-file leakage detected in selected benchmark: {len(group_overlap)} groups")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "benchmark_manifest.csv"
    fields = source_fields + ([] if "source_split_order" in source_fields else ["source_split_order"])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    counts: dict[str, int] = defaultdict(int)
    group_counts: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        key = f"{row['split']}:{row['barcode']}"
        counts[key] += 1
        group_counts[key].add(row["ccf_file"])
    summary = {
        "status": "complete",
        "benchmark_tier": args.tier_name,
        "seed": args.seed,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "benchmark_manifest": str(output.resolve()),
        "benchmark_manifest_sha256": file_sha256(output),
        "n_reads": len(selected),
        "species": list(SPECIES),
        "targets_per_species": targets,
        "counts": dict(sorted(counts.items())),
        "ccf_group_counts": {key: len(value) for key, value in sorted(group_counts.items())},
        "read_overlap_count": 0,
        "ccf_group_overlap_count": 0,
        "selection": "seeded_hash_with_ccf_file_round_robin_v1",
    }
    summary_path = args.output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
