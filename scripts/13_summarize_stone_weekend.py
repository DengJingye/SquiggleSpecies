#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from squiggle_species.constants import SPECIES
from squiggle_species.utils import save_json, write_csv


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def selection_rows(path: Path, stage: str) -> list[dict]:
    data = read_json(path)
    return [{"stage": stage, **row} for row in data["candidates"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final v3 stone weekend ablation package.")
    parser.add_argument("--head-selection", type=Path, required=True)
    parser.add_argument("--depth-selection", type=Path, required=True)
    parser.add_argument("--objective-selection", type=Path, required=True)
    parser.add_argument("--final-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--softcopyright-threshold", type=float, default=0.82)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(selection_rows(args.head_selection, "frozen_head"))
    rows.extend(selection_rows(args.depth_selection, "pft_depth"))
    rows.extend(selection_rows(args.objective_selection, "pft_objective"))
    write_csv(args.output_dir / "all_validation_candidates.csv", rows, list(rows[0]))

    objective = read_json(args.objective_selection)
    final_eval = read_json(args.final_eval)
    val_metrics = final_eval["metrics"]["val"]
    test_metrics = final_eval["metrics"]["test"]
    best = objective["best"]
    summary = {
        "status": "complete",
        "selection_basis": "validation macro-F1 only; test evaluated once for selected PFT candidate",
        "selected": best,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "softcopyright_update_threshold": args.softcopyright_threshold,
        "softcopyright_update_recommended": bool(test_metrics["macro_f1"] >= args.softcopyright_threshold),
    }
    save_json(args.output_dir / "final_summary.json", summary)

    labels = []
    values = []
    colors = []
    palette = {"frozen_head": "#6A8CAF", "pft_depth": "#2E8B70", "pft_objective": "#C56B4A"}
    for row in rows:
        if row["stage"] == "frozen_head":
            label = f"Frozen {row['aggregation']}"
        elif row["stage"] == "pft_depth":
            label = f"PFT-{row['trainable_lstm_blocks']} CE"
        else:
            label = f"PFT-{row['trainable_lstm_blocks']} {row['objective']}"
        labels.append(label)
        values.append(float(row["val_macro_f1"]))
        colors.append(palette[row["stage"]])
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(10, max(5, len(rows) * 0.42)))
    ax.barh(np.asarray(labels)[order], np.asarray(values)[order], color=np.asarray(colors)[order])
    ax.set_xlabel("Validation macro-F1")
    ax.set_title("Stone Bonito ablation (validation-only selection)")
    ax.set_xlim(0, 1)
    for index, value in enumerate(np.asarray(values)[order]):
        ax.text(value + 0.008, index, f"{value:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "validation_ablation.png", dpi=220)
    plt.close(fig)

    matrix = np.asarray(test_metrics["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(SPECIES)), SPECIES, rotation=45, ha="right")
    ax.set_yticks(range(len(SPECIES)), SPECIES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference label")
    ax.set_title(f"Selected stone model test confusion (macro-F1={test_metrics['macro_f1']:.4f})")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.output_dir / "selected_test_confusion_matrix.png", dpi=220)
    plt.close(fig)

    recalls = [test_metrics["per_species"][species]["recall"] for species in SPECIES]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(SPECIES, recalls, color="#2E8B70")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Selected stone model: per-species test recall")
    ax.tick_params(axis="x", rotation=45)
    for index, value in enumerate(recalls):
        ax.text(index, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "selected_test_per_species_recall.png", dpi=220)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
