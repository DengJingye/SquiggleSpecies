#!/usr/bin/env python3
"""Create compact diagnostics for the v1 file-held-out comparison."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def coarse8(species: str) -> str:
    return "LB01_LB12_group" if species in {"LB01", "LB12"} else species


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    run_root = root / "artifacts/runs/v1_groupheldout_small"
    manifest_path = root / "artifacts/manifests/v1_groupheldout_3000_seed42/group_split_manifest.csv"
    output_dir = root / "artifacts/summaries/v1_groupheldout_small/diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {row["read_id"]: row for row in read_csv(manifest_path)}
    methods = {
        "signal_frozen_ce": run_root / "signal_ce/test_predictions.csv",
        "signal_crossmodal_kd": run_root / "signal_crossmodal_kd/test_predictions.csv",
    }

    coarse_rows: list[dict] = []
    confusion_rows: list[dict] = []
    file_rows: list[dict] = []
    method_summary: dict[str, dict] = {}

    for method, prediction_path in methods.items():
        predictions = read_csv(prediction_path)
        true9 = [row["true_species"] for row in predictions]
        pred9 = [row["pred_species"] for row in predictions]
        true8 = [coarse8(value) for value in true9]
        pred8 = [coarse8(value) for value in pred9]

        coarse_metrics = {
            "method": method,
            "n": len(predictions),
            "accuracy_9class": accuracy_score(true9, pred9),
            "macro_f1_9class": f1_score(true9, pred9, average="macro"),
            "accuracy_8group": accuracy_score(true8, pred8),
            "macro_f1_8group": f1_score(true8, pred8, average="macro"),
        }
        coarse_rows.append(coarse_metrics)

        pair_counts = Counter(
            (row["true_species"], row["pred_species"])
            for row in predictions
            if row["true_species"] != row["pred_species"]
        )
        total_errors = sum(pair_counts.values())
        for rank, ((true_species, pred_species), count) in enumerate(pair_counts.most_common(20), 1):
            confusion_rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "true_species": true_species,
                    "pred_species": pred_species,
                    "count": count,
                    "fraction_of_method_errors": count / total_errors if total_errors else 0.0,
                }
            )

        by_file: dict[str, list[dict[str, str]]] = {}
        for row in predictions:
            matched = manifest.get(row["read_id"])
            if matched is None:
                raise RuntimeError(f"Missing prediction read_id in manifest: {row['read_id']}")
            by_file.setdefault(matched["ccf_file"], []).append(row)

        file_accuracies = []
        for ccf_file, rows in sorted(by_file.items()):
            correct = sum(int(row["correct"]) for row in rows)
            accuracy = correct / len(rows)
            file_accuracies.append(accuracy)
            file_rows.append(
                {
                    "method": method,
                    "true_species": rows[0]["true_species"],
                    "ccf_file": ccf_file,
                    "n": len(rows),
                    "accuracy": accuracy,
                    "mean_confidence": sum(float(row["confidence"]) for row in rows) / len(rows),
                }
            )

        lb_pair_errors = sum(
            count
            for (true_species, pred_species), count in pair_counts.items()
            if {true_species, pred_species} == {"LB01", "LB12"}
        )
        method_summary[method] = {
            **coarse_metrics,
            "errors_9class": total_errors,
            "lb01_lb12_mutual_errors": lb_pair_errors,
            "lb01_lb12_fraction_of_errors": lb_pair_errors / total_errors if total_errors else 0.0,
            "test_ccf_files": len(file_accuracies),
            "min_ccf_file_accuracy": min(file_accuracies),
            "max_ccf_file_accuracy": max(file_accuracies),
            "mean_ccf_file_accuracy": sum(file_accuracies) / len(file_accuracies),
        }

    write_csv(output_dir / "coarse8_metrics.csv", coarse_rows, list(coarse_rows[0]))
    write_csv(
        output_dir / "top_confusions.csv",
        confusion_rows,
        ["method", "rank", "true_species", "pred_species", "count", "fraction_of_method_errors"],
    )
    write_csv(
        output_dir / "per_ccf_file_metrics.csv",
        file_rows,
        ["method", "true_species", "ccf_file", "n", "accuracy", "mean_confidence"],
    )

    summary = {
        "status": "complete",
        "protocol": "same-read file-held-out v1 test split",
        "methods": method_summary,
        "interpretation": {
            "atlas_primary_classifier": False,
            "kd_scaleup": False,
            "next_model_test": "small Bonito partial fine-tuning on the same group-held-out protocol",
        },
    }
    with (output_dir / "v1_diagnostics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
