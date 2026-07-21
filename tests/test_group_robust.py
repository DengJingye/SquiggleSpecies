from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np
import torch

from squiggle_species.group_robust import (
    CrossFilePairBatchSampler,
    ccf_group_accuracy_summary,
    cross_file_supcon_loss,
    group_dro_loss,
)


@dataclass
class Record:
    label: int
    ccf_file: str


class GroupRobustTest(unittest.TestCase):
    def test_sampler_guarantees_cross_file_pairs(self):
        records = [Record(label, f"class{label}_file{file_index}") for label in range(3) for file_index in range(3) for _ in range(2)]
        sampler = CrossFilePairBatchSampler(records, batch_size=4, seed=42, batches_per_epoch=12)
        for batch in sampler:
            self.assertEqual(len(batch), 4)
            for start in (0, 2):
                left, right = (records[index] for index in batch[start : start + 2])
                self.assertEqual(left.label, right.label)
                self.assertNotEqual(left.ccf_file, right.ccf_file)

    def test_losses_are_finite_and_group_weights_update(self):
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]])
        labels = torch.tensor([0, 0, 0, 1])
        groups = torch.tensor([0, 1, 2, 3])
        contrast = cross_file_supcon_loss(features, labels, groups, temperature=0.1)
        self.assertTrue(torch.isfinite(contrast))
        weights = torch.full((4,), 0.25)
        robust, group_losses = group_dro_loss(torch.tensor([0.1, 0.2, 1.5, 0.3]), groups, weights, 0.1)
        self.assertTrue(torch.isfinite(robust))
        self.assertEqual(group_losses.shape[0], 4)
        self.assertGreater(float(weights[2]), 0.25)

    def test_group_accuracy_summary(self):
        summary, rows = ccf_group_accuracy_summary(
            ["a", "b", "c", "d"],
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 1, 1, 1]),
            {"a": "f0", "b": "f0", "c": "f1", "d": "f1"},
        )
        self.assertEqual(summary["n_groups"], 2)
        self.assertEqual(summary["worst_group_accuracy"], 0.5)
        self.assertEqual(summary["best_group_accuracy"], 1.0)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
