#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from squiggle_species.constants import SPECIES, SPLITS
from squiggle_species.utils import read_json, save_json, write_csv


FIELDS = [
    "split",
    "split_order",
    "barcode",
    "label",
    "read_id",
    "part_id",
    "part_base",
    "part_read_index",
    "chunk_start",
    "n_chunks",
    "chunk_path",
    "ccf_file",
    "ccf_read_index",
]


def assign_groups(group_counts, targets, cap, seed):
    assignments = {}
    assignment_rows = []
    for species_index, barcode in enumerate(SPECIES):
        groups = [(ccf_file, count) for (group_barcode, ccf_file), count in group_counts.items() if group_barcode == barcode]
        rng = random.Random(seed + species_index * 1009)
        rng.shuffle(groups)
        cursor = 0
        for split in SPLITS:
            effective = 0
            while effective < targets[split]:
                if cursor >= len(groups):
                    raise ValueError(f"Not enough CCF groups for {barcode}/{split}; reached {effective}/{targets[split]}")
                ccf_file, count = groups[cursor]
                cursor += 1
                key = (barcode, ccf_file)
                assignments[key] = split
                usable = min(count, cap, targets[split] - effective)
                effective += usable
                assignment_rows.append(
                    {
                        "barcode": barcode,
                        "ccf_file": ccf_file,
                        "split": split,
                        "available_reads": count,
                        "selected_cap": usable,
                    }
                )
    return assignments, assignment_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CCF-file-isolated small Zymo9 split from the reusable v10 bag manifest.")
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resources = read_json(args.resources)
    experiment = read_json(args.experiment)
    source_manifest = Path(resources["legacy_v10_bag_manifest"])
    targets = {key: int(value) for key, value in experiment["target_reads_per_species"].items()}
    cap = int(experiment["max_reads_per_ccf_file"])
    seed = int(experiment["seed"])
    require_sequence = bool(experiment.get("require_sequence_cache_overlap", False))

    sequence_read_ids = None
    if require_sequence:
        sequence_read_ids = set()
        sequence_root = Path(resources["sequence_kmer_cache"])
        for source_split in ("train", "atlas", "val", "test"):
            sequence_manifest = sequence_root / "manifests" / f"{source_split}_sequence_manifest.csv"
            with sequence_manifest.open(newline="") as handle:
                sequence_read_ids.update(row["read_id"] for row in csv.DictReader(handle))

    group_counts = Counter()
    with source_manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if sequence_read_ids is not None and row["read_id"] not in sequence_read_ids:
                continue
            group_counts[(row["barcode"], row["ccf_file"])] += 1
    assignments, assignment_rows = assign_groups(group_counts, targets, cap, seed)
    write_csv(
        args.output_dir / "ccf_group_assignments.csv",
        assignment_rows,
        ["barcode", "ccf_file", "split", "available_reads", "selected_cap"],
    )

    selected_per_group = Counter()
    selected_per_species_split = Counter()
    split_order = Counter()
    seen_reads = set()
    duplicate_reads = 0
    output_manifest = args.output_dir / "group_split_manifest.csv"
    with source_manifest.open(newline="") as source, output_manifest.open("w", newline="") as target_handle:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target_handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in reader:
            if sequence_read_ids is not None and row["read_id"] not in sequence_read_ids:
                continue
            key = (row["barcode"], row["ccf_file"])
            split = assignments.get(key)
            if split is None:
                continue
            species_split = (row["barcode"], split)
            if selected_per_species_split[species_split] >= targets[split]:
                continue
            if selected_per_group[key] >= cap:
                continue
            read_id = row["read_id"]
            if read_id in seen_reads:
                duplicate_reads += 1
                continue
            seen_reads.add(read_id)
            selected_per_group[key] += 1
            selected_per_species_split[species_split] += 1
            row["split"] = split
            row["split_order"] = split_order[species_split]
            split_order[species_split] += 1
            writer.writerow({field: row[field] for field in FIELDS})

    missing = {
        f"{barcode}|{split}": targets[split] - selected_per_species_split[(barcode, split)]
        for barcode in SPECIES
        for split in SPLITS
        if selected_per_species_split[(barcode, split)] != targets[split]
    }
    split_groups = defaultdict(set)
    for row in assignment_rows:
        split_groups[row["split"]].add((row["barcode"], row["ccf_file"]))
    group_overlap = set()
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            group_overlap.update(split_groups[left] & split_groups[right])
    summary = {
        "status": "ok" if not missing and not group_overlap and duplicate_reads == 0 else "failed",
        "source_manifest": str(source_manifest),
        "output_manifest": str(output_manifest),
        "seed": seed,
        "max_reads_per_ccf_file": cap,
        "require_sequence_cache_overlap": require_sequence,
        "available_sequence_read_ids": None if sequence_read_ids is None else len(sequence_read_ids),
        "targets_per_species": targets,
        "selected_counts": {f"{barcode}|{split}": selected_per_species_split[(barcode, split)] for barcode in SPECIES for split in SPLITS},
        "selected_ccf_groups": {split: len(split_groups[split]) for split in SPLITS},
        "exact_read_duplicate_count": duplicate_reads,
        "ccf_group_overlap_count": len(group_overlap),
        "missing_counts": missing,
        "interpretation": "The same CCF file is never used by more than one split.",
    }
    save_json(args.output_dir / "group_split_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
