#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from squiggle_species.constants import SPECIES
from squiggle_species.utils import file_sha256, save_json, write_csv


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def read_distance_matrix(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        labels = [name.split()[0] for name in (reader.fieldnames or [])[1:]]
        matrix: dict[str, dict[str, float]] = {}
        for row in reader:
            source = row["label"].split()[0]
            matrix[source] = {
                target: float(row[original])
                for target, original in zip(labels, (reader.fieldnames or [])[1:])
            }
    return matrix


def greedy_diversity_order(
    species: tuple[str, ...],
    matrix: dict[str, dict[str, float]],
    anchors: tuple[str, str],
) -> list[str]:
    if anchors[0] == anchors[1] or not set(anchors).issubset(species):
        raise ValueError(f"Invalid anchor pair: {anchors}")
    selected = list(anchors)
    remaining = set(species) - set(selected)
    while remaining:
        candidate = max(
            remaining,
            key=lambda name: (min(matrix[name][chosen] for chosen in selected), name),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def subset_manifest(source: Path, destination: Path, selected: list[str]) -> dict:
    fields, rows = read_rows(source)
    label_map = {name: index for index, name in enumerate(selected)}
    kept = []
    for row in rows:
        if row["barcode"] not in label_map:
            continue
        copied = dict(row)
        copied["label"] = str(label_map[row["barcode"]])
        kept.append(copied)
    write_csv(destination, kept, fields)

    read_sets = {
        split: {row["read_id"] for row in kept if row["split"] == split}
        for split in ("train", "val", "test")
    }
    group_sets = {
        split: {(row["barcode"], row["ccf_file"]) for row in kept if row["split"] == split}
        for split in ("train", "val", "test")
    }
    read_overlap = (read_sets["train"] & read_sets["val"]) | (read_sets["train"] & read_sets["test"]) | (
        read_sets["val"] & read_sets["test"]
    )
    group_overlap = (group_sets["train"] & group_sets["val"]) | (
        group_sets["train"] & group_sets["test"]
    ) | (group_sets["val"] & group_sets["test"])
    counts = Counter((row["split"], row["barcode"]) for row in kept)
    expected = {(split, name) for split in ("train", "val", "test") for name in selected}
    missing = sorted(expected - set(counts))
    if read_overlap or group_overlap or missing:
        raise ValueError(
            f"Invalid subset manifest {destination}: read_overlap={len(read_overlap)}, "
            f"group_overlap={len(group_overlap)}, missing={missing}"
        )
    return {
        "path": str(destination.resolve()),
        "sha256": file_sha256(destination),
        "rows": len(kept),
        "split_counts": {
            split: sum(1 for row in kept if row["split"] == split)
            for split in ("train", "val", "test")
        },
        "per_species_split_counts": {
            f"{split}:{name}": counts[(split, name)]
            for split in ("train", "val", "test")
            for name in selected
        },
        "read_overlap_count": len(read_overlap),
        "ccf_group_overlap_count": len(group_overlap),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pre-registered nested 5-9 species manifests.")
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--bag-manifest", type=Path, required=True)
    parser.add_argument("--distance-matrix", type=Path, required=True)
    parser.add_argument("--frozen-experiment-template", type=Path, required=True)
    parser.add_argument("--pft-experiment-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-pair", nargs=2, default=("LB01", "LB12"))
    parser.add_argument("--min-classes", type=int, default=5)
    parser.add_argument("--max-classes", type=int, default=9)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not 2 <= args.min_classes <= args.max_classes <= len(SPECIES):
        raise ValueError("Class counts must be between 2 and 9")

    matrix = read_distance_matrix(args.distance_matrix)
    order = greedy_diversity_order(SPECIES, matrix, tuple(args.anchor_pair))
    frozen_template = json.loads(args.frozen_experiment_template.read_text())
    pft_template = json.loads(args.pft_experiment_template.read_text())
    subsets = []
    for class_count in range(args.min_classes, args.max_classes + 1):
        selected = order[:class_count]
        subset_dir = args.output_dir / f"k{class_count}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        raw_audit = subset_manifest(args.raw_manifest, subset_dir / "raw_chunk_manifest.csv", selected)
        bag_audit = subset_manifest(args.bag_manifest, subset_dir / "bag_manifest.csv", selected)
        protocol_note = {
            "class_count": class_count,
            "selection": "LB01/LB12 hard anchor plus greedy max-min reference-genome k-mer diversity",
            "selection_uses_validation_or_test": False,
        }
        for name, template in (("frozen_experiment", frozen_template), ("pft_experiment", pft_template)):
            experiment = json.loads(json.dumps(template))
            experiment["species"] = selected
            experiment["class_count_protocol"] = protocol_note
            (subset_dir / f"{name}.json").write_text(json.dumps(experiment, indent=2) + "\n")
        audit = {
            "class_count": class_count,
            "species": selected,
            "label_map": {name: index for index, name in enumerate(selected)},
            "raw_manifest": raw_audit,
            "bag_manifest": bag_audit,
        }
        save_json(subset_dir / "subset_audit.json", audit)
        subsets.append(audit)

    protocol = {
        "status": "complete",
        "protocol_name": "hard-anchor-plus-reference-diversity-v1",
        "selection_basis": (
            "Keep the known close reference pair LB01/LB12 in every task, then add species by greedy "
            "max-min reference-genome canonical 5-mer cosine distance. No validation/test predictions are used."
        ),
        "important_note": (
            "Large reference-genome distance usually makes classification easier. The hard anchor prevents the "
            "lower-class tasks from becoming an easy-only cherry-picked benchmark."
        ),
        "distance_matrix": str(args.distance_matrix.resolve()),
        "distance_matrix_sha256": file_sha256(args.distance_matrix),
        "frozen_experiment_template": str(args.frozen_experiment_template.resolve()),
        "pft_experiment_template": str(args.pft_experiment_template.resolve()),
        "nested_order": order,
        "nested_property": all(set(order[:k]).issubset(order[: k + 1]) for k in range(2, len(order))),
        "subsets": subsets,
    }
    save_json(args.output_dir / "class_count_protocol.json", protocol)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
