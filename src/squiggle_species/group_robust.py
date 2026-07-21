from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


class CrossFilePairBatchSampler(Sampler[list[int]]):
    """Build class-balanced batches with same-class reads from different files."""

    def __init__(self, records: Sequence, batch_size: int, seed: int, batches_per_epoch: int | None = None):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Cross-file pair batches require an even batch_size >= 2")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.batches_per_epoch = int(batches_per_epoch or max(1, len(records) // batch_size))
        grouped: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, record in enumerate(records):
            grouped[int(record.label)][str(record.ccf_file)].append(index)
        invalid = {label: len(files) for label, files in grouped.items() if len(files) < 2}
        if invalid:
            raise ValueError(f"Each class needs at least two CCF files for cross-file pairing: {invalid}")
        self.grouped = {label: dict(files) for label, files in grouped.items()}
        self.labels = sorted(self.grouped)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003)
        pairs_per_batch = self.batch_size // 2
        label_stream = np.resize(rng.permutation(self.labels), self.batches_per_epoch * pairs_per_batch)
        for batch_index in range(self.batches_per_epoch):
            batch: list[int] = []
            start = batch_index * pairs_per_batch
            for label in label_stream[start : start + pairs_per_batch]:
                files = self.grouped[int(label)]
                selected_files = rng.choice(sorted(files), size=2, replace=False)
                for ccf_file in selected_files:
                    indices = files[str(ccf_file)]
                    batch.append(int(indices[int(rng.integers(0, len(indices)))]))
            yield batch


def cross_file_supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Supervised contrastive loss whose positives must cross CCF files."""

    features = F.normalize(features, dim=1)
    logits = features @ features.T / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    non_self = ~torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive = labels[:, None].eq(labels[None, :]) & group_ids[:, None].ne(group_ids[None, :]) & non_self
    denominator = (torch.exp(logits) * non_self).sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_prob = logits - denominator.log()
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        raise ValueError("Cross-file batch contains no valid same-class/different-file positives")
    return -((log_prob * positive).sum(dim=1)[valid] / positive_count[valid]).mean()


def group_dro_loss(
    per_sample_loss: torch.Tensor,
    group_ids: torch.Tensor,
    group_weights: torch.Tensor,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update GroupDRO weights and return the active-group weighted loss."""

    unique_groups, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
    group_losses = per_sample_loss.new_zeros(unique_groups.shape[0])
    counts = per_sample_loss.new_zeros(unique_groups.shape[0])
    group_losses.scatter_add_(0, inverse, per_sample_loss)
    counts.scatter_add_(0, inverse, torch.ones_like(per_sample_loss))
    group_losses = group_losses / counts.clamp_min(1.0)
    with torch.no_grad():
        update = torch.exp(float(step_size) * group_losses.detach()).clamp(max=1e4)
        group_weights[unique_groups] *= update
        group_weights /= group_weights.sum().clamp_min(1e-12)
    active_weights = group_weights[unique_groups]
    active_weights = active_weights / active_weights.sum().clamp_min(1e-12)
    return (active_weights * group_losses).sum(), group_losses.detach()


def ccf_group_accuracy_summary(
    read_ids: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_by_read: dict[str, str],
) -> tuple[dict, list[dict]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    labels: dict[str, int] = {}
    for index, (read_id, true_label) in enumerate(zip(read_ids, y_true)):
        group = group_by_read[str(read_id)]
        grouped[group].append(index)
        labels.setdefault(group, int(true_label))
        if labels[group] != int(true_label):
            raise ValueError(f"CCF group contains multiple labels: {group}")
    rows = []
    for group in sorted(grouped):
        indices = np.asarray(grouped[group], dtype=np.int64)
        accuracy = float(np.mean(y_true[indices] == y_pred[indices]))
        rows.append(
            {
                "ccf_file": group,
                "label": labels[group],
                "n_reads": int(indices.size),
                "accuracy": accuracy,
            }
        )
    accuracies = np.asarray([row["accuracy"] for row in rows], dtype=np.float64)
    summary = {
        "n_groups": len(rows),
        "worst_group_accuracy": float(accuracies.min()),
        "p10_group_accuracy": float(np.quantile(accuracies, 0.10)),
        "median_group_accuracy": float(np.median(accuracies)),
        "mean_group_accuracy": float(accuracies.mean()),
        "best_group_accuracy": float(accuracies.max()),
    }
    return summary, rows
