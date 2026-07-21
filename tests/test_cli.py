from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "squiggle_species", *args],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class SquiggleSpeciesCliTest(unittest.TestCase):
    def test_help(self) -> None:
        result = run_cli("--help")
        self.assertIn("inventory-ccf", result.stdout)
        self.assertIn("calibrate", result.stdout)
        self.assertIn("predict-raw-cache", result.stdout)

    def test_inventory_and_manifest_audit(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="squiggle_inventory_"))
        signals = root / "signals" / "LB01"
        signals.mkdir(parents=True)
        (signals / "example.ccf5").write_bytes(b"ccf5")
        inventory = root / "inventory.json"
        run_cli("inventory-ccf", str(root / "signals"), "-o", str(inventory))
        self.assertEqual(json.loads(inventory.read_text())["total_files"], 1)

        manifest = root / "manifest.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["read_id", "split", "barcode", "label", "ccf_file"])
            writer.writeheader()
            writer.writerows(
                [
                    {"read_id": "r1", "split": "train", "barcode": "LB01", "label": 0, "ccf_file": "a.ccf5"},
                    {"read_id": "r2", "split": "val", "barcode": "LB01", "label": 0, "ccf_file": "b.ccf5"},
                    {"read_id": "r3", "split": "test", "barcode": "LB01", "label": 0, "ccf_file": "c.ccf5"},
                ]
            )
        audit = root / "audit.json"
        run_cli("validate-manifest", str(manifest), "-o", str(audit))
        self.assertEqual(json.loads(audit.read_text())["status"], "pass")

    def test_dynamic_calibration_and_report(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="squiggle_calibration_"))
        predictions = root / "val_predictions.csv"
        rows = []
        for index in range(90):
            truth = index % 9
            correct = index % 5 != 0
            rows.append(
                {
                    "read_id": f"read_{index}",
                    "true_label": truth,
                    "predicted_label": truth if correct else (truth + 1) % 9,
                    "confidence": 0.9 if correct else 0.2,
                }
            )
        with predictions.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        calibration = root / "calibration"
        run_cli("calibrate", "--predictions", str(predictions), "--output-dir", str(calibration), "--min-coverage", "0.5")
        payload = json.loads((calibration / "calibration.json").read_text())
        self.assertEqual(payload["selection_split"], "validation")
        self.assertEqual(payload["selection_status"], "target_met")
        self.assertTrue(payload["threshold_enabled"])
        self.assertGreater(payload["threshold"], 0.2)
        self.assertGreaterEqual(payload["selected_metrics"]["accepted_accuracy"], 0.9)
        self.assertGreaterEqual(payload["selected_metrics"]["min_per_class_coverage"], 0.5)

        report = root / "report"
        run_cli(
            "report",
            "--predictions",
            str(predictions),
            "--threshold-json",
            str(calibration / "calibration.json"),
            "--output-dir",
            str(report),
        )
        summary = json.loads((report / "report_summary.json").read_text())
        self.assertGreater(summary["accepted_rate"], 0.5)
        self.assertTrue(summary["threshold_enabled"])
        self.assertIn("full_metrics", summary)
        self.assertIn("per_species_acceptance", summary)
        self.assertTrue((report / "species_abundance.csv").exists())
        self.assertTrue((report / "confusion_matrix.png").exists())

    def test_manifest_allows_same_filename_in_different_species(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="squiggle_manifest_group_"))
        manifest = root / "manifest.csv"
        rows = [
            {"read_id": "r1", "split": "train", "barcode": "LB01", "label": 0, "ccf_file": "LB01/shared.ccf5"},
            {"read_id": "r2", "split": "test", "barcode": "LB02", "label": 8, "ccf_file": "LB02/shared.ccf5"},
        ]
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        audit = root / "audit.json"
        run_cli("validate-manifest", str(manifest), "-o", str(audit))
        payload = json.loads(audit.read_text())
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(payload["ccf_file_split_leakage"], 0)

    def test_infeasible_calibration_disables_abstention(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="squiggle_infeasible_calibration_"))
        predictions = root / "val_predictions.csv"
        rows = []
        for index in range(90):
            truth = index % 9
            correct = index % 5 != 0
            rows.append(
                {
                    "read_id": f"read_{index}",
                    "true_label": truth,
                    "predicted_label": truth if correct else (truth + 1) % 9,
                    "confidence": 0.8,
                }
            )
        with predictions.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        calibration = root / "calibration"
        run_cli(
            "calibrate",
            "--predictions",
            str(predictions),
            "--output-dir",
            str(calibration),
            "--target-accuracy",
            "0.95",
        )
        payload = json.loads((calibration / "calibration.json").read_text())
        self.assertEqual(payload["selection_status"], "no_feasible_selective_operating_point")
        self.assertFalse(payload["threshold_enabled"])
        self.assertEqual(payload["threshold"], 0.0)


if __name__ == "__main__":
    unittest.main()
