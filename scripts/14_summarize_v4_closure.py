#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from squiggle_species.group_robust import ccf_group_accuracy_summary
from squiggle_species.utils import read_json, save_json, write_csv


def prediction_group_metrics(predictions: Path, raw_manifest: Path) -> tuple[dict, list[dict]]:
    group_by_read = {}
    with raw_manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            group_by_read[row["read_id"]] = row["ccf_file"]
    read_ids, y_true, y_pred = [], [], []
    with predictions.open(newline="") as handle:
        for row in csv.DictReader(handle):
            read_ids.append(row["read_id"])
            y_true.append(int(row["true_label"]))
            y_pred.append(int(row["pred_label"]))
    return ccf_group_accuracy_summary(
        read_ids,
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        group_by_read,
    )


def run_metrics(name: str, seed: int, summary_path: Path) -> dict:
    data = read_json(summary_path)
    val = data["metrics"]["val"]
    test = data["metrics"].get("test", {})
    return {
        "name": name,
        "seed": seed,
        "val_accuracy": val["accuracy"],
        "val_macro_f1": val["macro_f1"],
        "test_accuracy": test.get("accuracy"),
        "test_macro_f1": test.get("macro_f1"),
        "runtime_sec": data.get("runtime_sec"),
    }


def save_reproducibility_plot(rows: list[dict], output: Path) -> None:
    valid = [row for row in rows if row["test_macro_f1"] is not None]
    x = np.arange(len(valid))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - 0.18, [row["val_macro_f1"] for row in valid], 0.36, label="Validation", color="#3A7CA5")
    ax.bar(x + 0.18, [row["test_macro_f1"] for row in valid], 0.36, label="Test", color="#4F8F68")
    ax.set_xticks(x, [str(row["seed"]) for row in valid])
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_group_plot(baseline_rows: list[dict], robust_rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.boxplot(
        [
            [row["accuracy"] for row in baseline_rows],
            [row["accuracy"] for row in robust_rows],
        ],
        showmeans=True,
    )
    ax.set_xticks([1, 2], ["v3 baseline", "v4 cross-file"])
    ax.set_ylabel("Validation CCF-file accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize bounded v4 reproducibility and cross-file robustness.")
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--repeat-summaries", type=Path, nargs="*", default=[])
    parser.add_argument("--robust-summary", type=Path, required=True)
    parser.add_argument("--robust-eval-summary", type=Path, default=None)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiment = read_json(args.experiment)
    baseline = read_json(args.baseline_summary)
    robust = read_json(args.robust_summary)
    baseline_val = baseline["metrics"]["val"]
    robust_val = robust["metrics"]["val"]
    baseline_group, baseline_group_rows = prediction_group_metrics(
        args.baseline_dir / "val_predictions.csv", args.raw_manifest
    )
    robust_group_path = args.robust_summary.parent / "val_ccf_group_summary.json"
    robust_group = read_json(robust_group_path)
    with (args.robust_summary.parent / "val_ccf_group_metrics.csv").open(newline="") as handle:
        robust_group_rows = [
            {
                **row,
                "label": int(row["label"]),
                "n_reads": int(row["n_reads"]),
                "accuracy": float(row["accuracy"]),
            }
            for row in csv.DictReader(handle)
        ]

    gates = experiment["gates"]
    val_gain = float(robust_val["macro_f1"] - baseline_val["macro_f1"])
    p10_gain = float(robust_group["p10_group_accuracy"] - baseline_group["p10_group_accuracy"])
    promoted = (
        val_gain >= float(gates["minimum_crossfile_val_macro_f1_gain"])
        and p10_gain >= float(gates["minimum_crossfile_p10_group_accuracy_gain"])
    )
    gate = {
        "status": "complete",
        "selection_split": "validation",
        "baseline_val_macro_f1": baseline_val["macro_f1"],
        "robust_val_macro_f1": robust_val["macro_f1"],
        "val_macro_f1_gain": val_gain,
        "minimum_val_macro_f1_gain": gates["minimum_crossfile_val_macro_f1_gain"],
        "baseline_val_p10_group_accuracy": baseline_group["p10_group_accuracy"],
        "robust_val_p10_group_accuracy": robust_group["p10_group_accuracy"],
        "p10_group_accuracy_gain": p10_gain,
        "minimum_p10_group_accuracy_gain": gates["minimum_crossfile_p10_group_accuracy_gain"],
        "promote_to_test": promoted,
        "test_policy": "evaluate once only when both validation gates pass",
    }
    save_json(args.output_dir / "gate_decision.json", gate)

    reproducibility = [run_metrics("v3_selected", 42, args.baseline_summary)]
    for summary_path in args.repeat_summaries:
        if summary_path.exists():
            data = read_json(summary_path)
            reproducibility.append(run_metrics("v3_repeat", int(data["seed"]), summary_path))
    write_csv(
        args.output_dir / "reproducibility_metrics.csv",
        reproducibility,
        ["name", "seed", "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1", "runtime_sec"],
    )
    test_values = [row["test_macro_f1"] for row in reproducibility if row["test_macro_f1"] is not None]
    reproducibility_summary = {
        "completed_seeds": [row["seed"] for row in reproducibility],
        "n_completed": len(reproducibility),
        "test_macro_f1_mean": float(np.mean(test_values)) if test_values else None,
        "test_macro_f1_std_population": float(np.std(test_values)) if test_values else None,
        "test_macro_f1_min": float(np.min(test_values)) if test_values else None,
        "test_macro_f1_max": float(np.max(test_values)) if test_values else None,
    }
    save_json(args.output_dir / "reproducibility_summary.json", reproducibility_summary)
    if test_values:
        save_reproducibility_plot(reproducibility, args.output_dir / "reproducibility_macro_f1.png")

    comparison_rows = []
    for method, rows in (("v3_baseline", baseline_group_rows), ("v4_crossfile", robust_group_rows)):
        for row in rows:
            comparison_rows.append({"method": method, **row})
    write_csv(
        args.output_dir / "validation_ccf_group_comparison.csv",
        comparison_rows,
        ["method", "ccf_file", "label", "n_reads", "accuracy"],
    )
    save_group_plot(baseline_group_rows, robust_group_rows, args.output_dir / "validation_ccf_group_accuracy.png")

    robust_test = None
    if args.robust_eval_summary is not None and args.robust_eval_summary.exists():
        robust_test = read_json(args.robust_eval_summary)["metrics"].get("test")
    final = {
        "status": "complete",
        "bounded_experiment": True,
        "baseline": {
            "val_macro_f1": baseline_val["macro_f1"],
            "test_macro_f1": baseline["metrics"]["test"]["macro_f1"],
            "val_ccf_groups": baseline_group,
        },
        "reproducibility": reproducibility_summary,
        "crossfile_groupdro": {
            "val_macro_f1": robust_val["macro_f1"],
            "val_ccf_groups": robust_group,
            "gate": gate,
            "test": robust_test,
        },
        "closure_decision": (
            "promote_crossfile_candidate_and_build_new_final_holdout"
            if promoted
            else "freeze_v3_and_stop_bonito_closed_set_optimization"
        ),
    }
    save_json(args.output_dir / "v4_final_summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
