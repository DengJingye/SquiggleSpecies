from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from squiggle_species.ccf import chunk_signal, discover_ccf5, preprocess_read


class CcfPipelineContractTest(unittest.TestCase):
    def test_discovery_accepts_file_and_recursive_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "nested"
            nested.mkdir()
            first = root / "a.ccf5"
            second = nested / "b.ccf5"
            first.touch()
            second.touch()
            self.assertEqual(discover_ccf5(first), [first.resolve()])
            self.assertEqual(discover_ccf5(root), [first.resolve(), second.resolve()])

    def test_chunking_uses_declared_stride(self) -> None:
        signal = np.arange(17000, dtype=np.float32)
        chunks = chunk_signal(
            signal,
            discard_first=5000,
            chunk_size=6000,
            overlap=3000,
        )
        self.assertEqual(chunks.shape, (3, 6000))
        self.assertEqual(float(chunks[0, 0]), 5000.0)
        self.assertEqual(float(chunks[1, 0]), 8000.0)

    def test_physical_restoration_precedes_stone_normalization(self) -> None:
        record = {
            "signal": np.arange(12000, dtype=np.int16),
            "lvdsmid": 10.0,
            "unit": 0.5,
            "read_id": "read-1",
        }
        chunks = preprocess_read(
            record,
            profile_id="legacy-stone-v1",
            discard_first=5000,
            chunk_size=6000,
            overlap=3000,
        )
        self.assertEqual(chunks.shape, (1, 6000))
        self.assertTrue(np.isfinite(chunks).all())


if __name__ == "__main__":
    unittest.main()
