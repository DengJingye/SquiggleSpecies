#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from squiggle_species.utils import read_json, save_json, write_csv


def main():
    parser = argparse.ArgumentParser(description="Compare group-held-out CE and KD small experiments.")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--ce-summary", type=Path, required=True)
    parser.add_argument("--kd-summary", type=Path, required=True)
    parser.add_argument("--teacher-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment = read_json(args.experiment)
    ce = read_json(args.ce_summary)
    kd = read_json(args.kd_summary)
    teacher = read_json(args.teacher_metrics)
    rows = [
        {
            "method": "sequence_teacher_group_heldout",
            "val_accuracy": teacher["metrics"]["val"]["accuracy"],
            "val_macro_f1": teacher["metrics"]["val"]["macro_f1"],
            "test_accuracy": teacher["metrics"]["test"]["accuracy"],
            "test_macro_f1": teacher["metrics"]["test"]["macro_f1"],
        },
        {
            "method": "signal_frozen_ce_group_heldout",
            "val_accuracy": ce["metrics"]["val"]["accuracy"],
            "val_macro_f1": ce["metrics"]["val"]["macro_f1"],
            "test_accuracy": ce["metrics"]["test"]["accuracy"],
            "test_macro_f1": ce["metrics"]["test"]["macro_f1"],
        },
        {
            "method": "signal_frozen_crossmodal_kd_group_heldout",
            "val_accuracy": kd["metrics"]["val"]["accuracy"],
            "val_macro_f1": kd["metrics"]["val"]["macro_f1"],
            "test_accuracy": kd["metrics"]["test"]["accuracy"],
            "test_macro_f1": kd["metrics"]["test"]["macro_f1"],
        },
    ]
    write_csv(args.output_dir / "small_groupheldout_comparison.csv", rows, list(rows[0].keys()))
    gain = rows[2]["val_macro_f1"] - rows[1]["val_macro_f1"]
    gates = experiment["gates"]
    pass_score = rows[2]["val_macro_f1"] >= float(gates["minimum_val_macro_f1_for_scaleup"])
    pass_gain = gain >= float(gates["minimum_val_gain_for_complex_method"])
    decision = "scale_kd" if pass_score and pass_gain else "do_not_scale_kd"
    summary = {
        "status": "complete",
        "selection_basis": "validation macro-F1 only",
        "kd_val_gain_vs_ce": gain,
        "minimum_val_macro_f1_for_scaleup": gates["minimum_val_macro_f1_for_scaleup"],
        "minimum_val_gain_for_complex_method": gates["minimum_val_gain_for_complex_method"],
        "decision": decision,
        "next_step": (
            "Run a larger group-held-out KD experiment and prepare small raw chunks for partial Bonito fine-tuning."
            if decision == "scale_kd"
            else "Do not add more losses. Audit domain/batch coverage and obtain independent per-species runs or test raw-signal reference matching."
        ),
        "methods": rows,
    }
    save_json(args.output_dir / "decision_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

