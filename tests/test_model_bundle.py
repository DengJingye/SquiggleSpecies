from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from squiggle_species.model_bundle import load_model_bundle, verify_bonito_weights
from squiggle_species.utils import file_sha256


class ModelBundleTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> Path:
        checkpoint = root / "model.pth"
        checkpoint.write_bytes(b"checkpoint")
        model_dir = root / "bonito"
        model_dir.mkdir()
        weights = model_dir / "weights_0.tar"
        weights.write_bytes(b"bonito")
        bundle = {
            "schema_version": "1.0",
            "model_family": "bonito_pft",
            "class_names": ["A", "B"],
            "checkpoint": {
                "path": "model.pth",
                "sha256": file_sha256(checkpoint),
            },
            "backbone": {
                "weights": {
                    "filename": "weights_0.tar",
                    "sha256": file_sha256(weights),
                }
            },
            "preprocessing": {"profile_id": "legacy-stone-v1"},
            "chunking": {
                "discard_first": 5000,
                "chunk_size": 6000,
                "overlap": 3000,
                "max_chunks": 16,
            },
            "calibration": {
                "selected_on": "validation",
                "threshold_enabled": True,
                "threshold": 0.7,
            },
        }
        path = root / "model_bundle.json"
        path.write_text(json.dumps(bundle))
        return path

    def test_bundle_and_bonito_hashes_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = load_model_bundle(self.make_bundle(root))
            audit = verify_bonito_weights(bundle, root / "bonito")
            self.assertEqual(audit["expected_sha256"], audit["observed_sha256"])
            self.assertEqual(bundle["class_names"], ["A", "B"])

    def test_checkpoint_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_bundle(root)
            (root / "model.pth").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Checkpoint SHA256 mismatch"):
                load_model_bundle(path)

    def test_non_validation_calibration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_bundle(root)
            payload = json.loads(path.read_text())
            payload["calibration"]["selected_on"] = "test"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "selected_on"):
                load_model_bundle(path)


if __name__ == "__main__":
    unittest.main()
