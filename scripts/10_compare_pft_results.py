#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from squiggle_species.utils import read_json, save_json, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--frozen-summary", type=Path, required=True)
    parser.add_argument("--pft-a", type=Path, required=True)
    parser.add_argument("--pft-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    experiment = read_json(args.experiment)
    gates = experiment["gates"]
    frozen = read_json(args.frozen_summary)
    baseline = float(frozen["metrics"]["val"]["macro_f1"])
    summaries = [read_json(args.pft_a), read_json(args.pft_b)]
    rows = []
    for summary in summaries:
        val_f1 = float(summary["metrics"]["val"]["macro_f1"])
        rows.append(
            {
                "method": summary["mode"],
                "trainable_lstm_blocks": summary["trainable_lstm_blocks"],
                "val_accuracy": summary["metrics"]["val"]["accuracy"],
                "val_macro_f1": val_f1,
                "test_accuracy": summary["metrics"]["test"]["accuracy"],
                "test_macro_f1": summary["metrics"]["test"]["macro_f1"],
                "val_gain_vs_frozen_ce": val_f1 - baseline,
                "runtime_sec": summary["runtime_sec"],
            }
        )
    best = max(rows, key=lambda row: row["val_macro_f1"])
    scale = (
        best["val_macro_f1"] >= float(gates["minimum_val_macro_f1_for_scaleup"])
        and best["val_gain_vs_frozen_ce"] >= float(gates["minimum_val_gain_for_scaleup"])
    )
    result = {
        "status": "complete",
        "selection_basis": "validation macro-F1 only",
        "frozen_ce_val_macro_f1": baseline,
        "frozen_ce_test_macro_f1": frozen["metrics"]["test"]["macro_f1"],
        "best": best,
        "decision": "smoke_only_no_scientific_decision" if args.smoke else ("scale_best_pft" if scale else "do_not_scale_pft"),
        "next_step": (
            "Smoke only: verify program completion; do not interpret metrics."
            if args.smoke
            else (
                "Increase reads per species only after preserving the same group-held-out protocol."
                if scale
                else "Stop frozen/partial Bonito head tuning; prioritize independent runs or raw-signal reference matching."
            )
        ),
        "methods": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pft_candidates.csv", rows, list(rows[0]))
    save_json(args.output_dir / "decision_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
