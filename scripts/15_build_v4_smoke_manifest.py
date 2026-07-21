#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny cross-file raw-manifest smoke subset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-files-per-species", type=int, default=2)
    parser.add_argument("--train-reads-per-file", type=int, default=2)
    parser.add_argument("--eval-reads-per-species", type=int, default=2)
    args = parser.parse_args()

    with args.input.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    selected = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        by_species: dict[str, list[dict]] = defaultdict(list)
        for row in split_rows:
            by_species[row["barcode"]].append(row)
        for species in sorted(by_species):
            species_rows = by_species[species]
            if split == "train":
                by_file: dict[str, list[dict]] = defaultdict(list)
                for row in species_rows:
                    by_file[row["ccf_file"]].append(row)
                files = sorted(by_file)[: args.train_files_per_species]
                if len(files) < args.train_files_per_species:
                    raise ValueError(f"Not enough train CCF files for {species}")
                for ccf_file in files:
                    selected.extend(by_file[ccf_file][: args.train_reads_per_file])
            else:
                selected.extend(species_rows[: args.eval_reads_per_species])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    counts = defaultdict(int)
    groups = defaultdict(set)
    for row in selected:
        counts[(row["split"], row["barcode"])] += 1
        groups[(row["split"], row["barcode"])].add(row["ccf_file"])
    summary = {
        "status": "complete",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "n_rows": len(selected),
        "counts": {f"{split}:{species}": count for (split, species), count in sorted(counts.items())},
        "group_counts": {
            f"{split}:{species}": len(values) for (split, species), values in sorted(groups.items())
        },
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
