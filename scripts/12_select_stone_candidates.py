#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from squiggle_species.constants import SPECIES
from squiggle_species.utils import save_json, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a stone ablation candidate by validation macro-F1 only.")
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-name", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for path in args.summaries:
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text())
        val = data.get("metrics", {}).get("val", {})
        if "macro_f1" not in val:
            raise ValueError(f"Missing validation macro-F1: {path}")
        candidates.append(
            {
                "summary": str(path.resolve()),
                "run_dir": str(path.resolve().parent),
                "checkpoint": str((path.resolve().parent / "model.pth")),
                "aggregation": data.get("aggregation", "transformer"),
                "objective": data.get("objective", data.get("mode", "ce")),
                "trainable_lstm_blocks": data.get("trainable_lstm_blocks", 0),
                "best_stage": data.get("best_stage", ""),
                "max_chunks": data.get("max_chunks"),
                "val_accuracy": val.get("accuracy"),
                "val_macro_f1": val["macro_f1"],
                "runtime_sec": data.get("runtime_sec"),
            }
        )
    candidates.sort(key=lambda row: (-float(row["val_macro_f1"]), str(row["run_dir"])))
    best = candidates[0]

    best_data = json.loads(Path(best["summary"]).read_text())
    matrix = best_data["metrics"]["val"]["confusion_matrix"]
    pair_rows = []
    for left in range(len(SPECIES)):
        for right in range(left + 1, len(SPECIES)):
            support = sum(matrix[left]) + sum(matrix[right])
            errors = matrix[left][right] + matrix[right][left]
            pair_rows.append(
                {
                    "left": SPECIES[left],
                    "right": SPECIES[right],
                    "symmetric_errors": int(errors),
                    "symmetric_error_rate": float(errors / support) if support else 0.0,
                }
            )
    pair_rows.sort(key=lambda row: (-row["symmetric_error_rate"], -row["symmetric_errors"], row["left"], row["right"]))
    hard_pairs = ",".join(f"{row['left']}:{row['right']}" for row in pair_rows[:3])
    selection = {
        "status": "complete",
        "selection_name": args.selection_name,
        "selection_basis": "validation macro-F1 only",
        "best": best,
        "hard_pairs_from_best_validation_confusion": hard_pairs,
        "candidates": candidates,
    }
    save_json(args.output_dir / "selection.json", selection)
    write_csv(args.output_dir / "candidates.csv", candidates, list(candidates[0]))
    write_csv(
        args.output_dir / "validation_confusable_pairs.csv",
        pair_rows,
        ["left", "right", "symmetric_errors", "symmetric_error_rate"],
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
