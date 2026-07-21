from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from squiggle_species.bonito_pft import BonitoPartialStudent, RawChunkBagDataset, read_raw_manifest
from squiggle_species.inference import predict_raw_bags
from squiggle_species.models import SignalStudent
from squiggle_species.preprocessing import legacy_stone_v1, process_signal


ROOT = Path(__file__).resolve().parents[1]
SPECIES = ("LB01", "LB06", "LB07", "LB08", "LB09", "LB12", "LB18", "LB11", "LB02")


class BenchmarkPackagingTest(unittest.TestCase):
    def test_legacy_stone_has_exact_unclamped_mad_behavior(self) -> None:
        signal = np.asarray([0.0, 0.0, 1.0, 100.0], dtype=np.float32)
        expected = ((signal - np.median(signal)) / max(1.4826 * np.median(np.abs(signal - np.median(signal))), 1.0)).astype(
            np.float32
        )
        actual = legacy_stone_v1(signal)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        self.assertGreater(float(actual.max()), 6.0)
        np.testing.assert_array_equal(actual, process_signal(signal, "legacy-stone-v1"))

    def test_fixed_manifest_and_portable_raw_bundle(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="squiggle_benchmark_"))
        source_manifest = root / "source_manifest.csv"
        raw_manifest = root / "source_raw_manifest.csv"
        source_fields = ["split", "split_order", "barcode", "label", "read_id", "ccf_file"]
        source_rows = []
        raw_rows = []
        raw_starts = {species: 0 for species in SPECIES}
        arrays = {species: np.arange(24 * 4 * 8, dtype=np.float16).reshape(24 * 4, 8) for species in SPECIES}
        for label, species in enumerate(SPECIES):
            np.save(root / f"{species}.npy", arrays[species])
            split_order = {"train": 0, "val": 0, "test": 0}
            for split_index, split in enumerate(("train", "val", "test")):
                for read_index in range(8):
                    read_id = f"{species}_{split}_{read_index}"
                    ccf_file = f"{species}_{split}_file{read_index % 2}.ccf5"
                    row = {
                        "split": split,
                        "split_order": split_order[split],
                        "barcode": species,
                        "label": label,
                        "read_id": read_id,
                        "ccf_file": ccf_file,
                    }
                    split_order[split] += 1
                    source_rows.append(row)
                    raw_rows.append(
                        {
                            **row,
                            "raw_chunk_path": f"{species}.npy",
                            "raw_chunk_start": raw_starts[species],
                            "raw_n_chunks": 4,
                        }
                    )
                    raw_starts[species] += 4
        with source_manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=source_fields)
            writer.writeheader()
            writer.writerows(source_rows)
        with raw_manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
            writer.writeheader()
            writer.writerows(raw_rows)

        fixed = root / "fixed"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/16_build_fixed_benchmark_manifest.py"),
                "--source-manifest",
                str(source_manifest),
                "--output-dir",
                str(fixed),
                "--tier-name",
                "fixture",
                "--train-per-species",
                "2",
                "--val-per-species",
                "2",
                "--test-per-species",
                "2",
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        summary = json.loads((fixed / "benchmark_summary.json").read_text())
        self.assertEqual(summary["n_reads"], 54)
        self.assertEqual(summary["read_overlap_count"], 0)
        self.assertEqual(summary["ccf_group_overlap_count"], 0)

        bundle = root / "bundle"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/17_export_compact_raw_benchmark.py"),
                "--benchmark-manifest",
                str(fixed / "benchmark_manifest.csv"),
                "--source-raw-manifest",
                str(raw_manifest),
                "--preprocessing-profile",
                str(ROOT / "config/preprocessing_legacy_stone_v1.json"),
                "--output-dir",
                str(bundle),
                "--max-chunks",
                "3",
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        bundle_summary = json.loads((bundle / "raw_benchmark_summary.json").read_text())
        self.assertEqual(bundle_summary["n_reads"], 54)
        self.assertEqual(bundle_summary["n_chunks"], 162)
        records = read_raw_manifest(bundle / "raw_benchmark_manifest.csv")
        self.assertEqual({split: len(values) for split, values in records.items()}, {"train": 18, "val": 18, "test": 18})
        dataset = RawChunkBagDataset(records["test"], max_chunks=16, training=False, seed=0)
        chunks, label, read_id = dataset[0]
        self.assertEqual(chunks.shape, (3, 8))
        self.assertIsInstance(label, int)
        self.assertTrue(read_id)

    def test_raw_cache_predictor_loads_pft_checkpoint(self) -> None:
        class DummyEncode(nn.Module):
            def forward(self, hidden: torch.Tensor) -> torch.Tensor:
                values = hidden.mean(dim=2).transpose(0, 1).unsqueeze(-1)
                return values.repeat(1, 1, 768)

        class DummyBonito(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(*[nn.Identity() for _ in range(8)], DummyEncode())

        root = Path(tempfile.mkdtemp(prefix="squiggle_raw_predict_"))
        chunks = np.arange(4 * 16, dtype=np.float16).reshape(4, 16)
        np.save(root / "chunks.npy", chunks)
        manifest = root / "raw_manifest.csv"
        rows = [
            {
                "split": "test",
                "split_order": index,
                "barcode": "LB01",
                "label": 0,
                "read_id": f"read_{index}",
                "raw_chunk_path": "chunks.npy",
                "raw_chunk_start": index * 2,
                "raw_n_chunks": 2,
                "ccf_file": "test.ccf5",
                "preprocessing_profile_id": "legacy-stone-v1",
            }
            for index in range(2)
        ]
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        experiment = {
            "species": list(SPECIES),
            "model": {
                "hidden_dim": 8,
                "projection_dim": 8,
                "attention_dim": 4,
                "transformer_layers": 1,
                "transformer_heads": 2,
                "transformer_ff_dim": 16,
                "max_chunks": 2,
                "dropout": 0.0,
                "aggregation": "mean",
            },
            "training": {"eval_batch_size": 2, "chunk_microbatch": 4},
        }
        student = SignalStudent(
            input_dim=768,
            hidden_dim=8,
            projection_dim=8,
            attention_dim=4,
            num_classes=9,
            dropout=0.0,
            transformer_layers=1,
            transformer_heads=2,
            transformer_ff_dim=16,
            aggregation="mean",
        )
        model = BonitoPartialStudent(DummyBonito(), student, trainable_lstm_blocks=1, chunk_microbatch=4)
        checkpoint = root / "model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "experiment": experiment,
                "trainable_lstm_blocks": 1,
                "species": SPECIES,
            },
            checkpoint,
        )
        fake_bonito = types.ModuleType("bonito")
        fake_util = types.ModuleType("bonito.util")
        fake_util.load_model = lambda *_args, **_kwargs: DummyBonito()
        previous_bonito = sys.modules.get("bonito")
        previous_util = sys.modules.get("bonito.util")
        sys.modules["bonito"] = fake_bonito
        sys.modules["bonito.util"] = fake_util
        try:
            output = root / "predictions.csv"
            summary = predict_raw_bags(
                manifest,
                checkpoint,
                root,
                output,
                device="cpu",
                expected_preprocessing_profile="legacy-stone-v1",
            )
            with self.assertRaisesRegex(ValueError, "profile mismatch"):
                predict_raw_bags(
                    manifest,
                    checkpoint,
                    root,
                    root / "wrong.csv",
                    device="cpu",
                    expected_preprocessing_profile="apple-sclamp-v1",
                )
        finally:
            if previous_bonito is None:
                sys.modules.pop("bonito", None)
            else:
                sys.modules["bonito"] = previous_bonito
            if previous_util is None:
                sys.modules.pop("bonito.util", None)
            else:
                sys.modules["bonito.util"] = previous_util
        self.assertEqual(summary["reads"], 2)
        self.assertEqual(summary["preprocessing_profile_id"], "legacy-stone-v1")
        with output.open(newline="") as handle:
            predictions = list(csv.DictReader(handle))
        self.assertEqual(len(predictions), 2)
        self.assertIn("prob_LB01", predictions[0])


if __name__ == "__main__":
    unittest.main()
