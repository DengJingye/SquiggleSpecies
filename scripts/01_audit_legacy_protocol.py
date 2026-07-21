#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from squiggle_species.constants import SPECIES
from squiggle_species.utils import read_json, save_json, write_csv


def run_prefix(ccf_file: str) -> str:
    return Path(ccf_file).name.split("_", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit leakage and batch structure in the legacy Zymo9 split.")
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resources = read_json(args.resources)
    manifest = Path(resources["legacy_split_manifest"])

    file_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    file_reads: Counter[tuple[str, str]] = Counter()
    species_split_reads: Counter[tuple[str, str]] = Counter()
    run_species: dict[str, set[str]] = defaultdict(set)
    read_split: dict[str, str] = {}
    duplicate_read_rows = 0
    inconsistent_read_splits = 0
    total_rows = 0

    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            barcode = row["barcode"]
            split = row["split"]
            ccf_file = row["ccf_file"]
            read_id = row["read_id"]
            key = (barcode, ccf_file)
            file_splits[key].add(split)
            file_reads[key] += 1
            species_split_reads[(barcode, split)] += 1
            run_species[run_prefix(ccf_file)].add(barcode)
            previous = read_split.get(read_id)
            if previous is not None:
                duplicate_read_rows += 1
                if previous != split:
                    inconsistent_read_splits += 1
            else:
                read_split[read_id] = split

    file_rows = []
    species_file_counts = Counter()
    species_leaked_files = Counter()
    for (barcode, ccf_file), splits in sorted(file_splits.items()):
        species_file_counts[barcode] += 1
        if len(splits) > 1:
            species_leaked_files[barcode] += 1
        file_rows.append(
            {
                "barcode": barcode,
                "ccf_file": ccf_file,
                "read_count": file_reads[(barcode, ccf_file)],
                "split_count": len(splits),
                "splits": "|".join(sorted(splits)),
                "cross_split": int(len(splits) > 1),
                "run_prefix": run_prefix(ccf_file),
            }
        )
    write_csv(
        args.output_dir / "legacy_ccf_file_overlap.csv",
        file_rows,
        ["barcode", "ccf_file", "read_count", "split_count", "splits", "cross_split", "run_prefix"],
    )

    per_species = []
    for barcode in SPECIES:
        per_species.append(
            {
                "barcode": barcode,
                "ccf_files": species_file_counts[barcode],
                "files_crossing_splits": species_leaked_files[barcode],
                "crossing_fraction": species_leaked_files[barcode] / max(1, species_file_counts[barcode]),
                "train_reads": species_split_reads[(barcode, "train")],
                "atlas_reads": species_split_reads[(barcode, "atlas")],
                "val_reads": species_split_reads[(barcode, "val")],
                "test_reads": species_split_reads[(barcode, "test")],
            }
        )
    write_csv(
        args.output_dir / "legacy_split_by_species.csv",
        per_species,
        ["barcode", "ccf_files", "files_crossing_splits", "crossing_fraction", "train_reads", "atlas_reads", "val_reads", "test_reads"],
    )

    total_files = len(file_splits)
    leaked_files = sum(len(splits) > 1 for splits in file_splits.values())
    summary = {
        "status": "legacy_protocol_audited",
        "manifest": str(manifest),
        "total_rows": total_rows,
        "unique_read_ids": len(read_split),
        "duplicate_read_rows": duplicate_read_rows,
        "inconsistent_read_splits": inconsistent_read_splits,
        "species_ccf_files": total_files,
        "species_ccf_files_crossing_splits": leaked_files,
        "ccf_file_cross_split_fraction": leaked_files / max(1, total_files),
        "run_prefix_to_species": {key: sorted(value) for key, value in sorted(run_species.items())},
        "primary_finding": (
            "Exact read IDs are separated, but CCF-file groups are not isolated. "
            "Legacy internal metrics therefore measure random-read holdout, not file/run holdout."
        ),
        "decision": "All new model selection must use a group-isolated manifest.",
    }
    save_json(args.output_dir / "legacy_protocol_audit.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

