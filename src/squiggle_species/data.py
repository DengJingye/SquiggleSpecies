from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BagRecord:
    split: str
    split_order: int
    barcode: str
    label: int
    read_id: str
    part_id: int
    part_base: str
    part_read_index: int
    chunk_start: int
    n_chunks: int
    chunk_path: str
    ccf_file: str
    ccf_read_index: int


def read_bag_manifest(path: str | Path) -> dict[str, list[BagRecord]]:
    by_split: dict[str, list[BagRecord]] = {"train": [], "val": [], "test": []}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in by_split:
                continue
            by_split[split].append(
                BagRecord(
                    split=split,
                    split_order=int(row["split_order"]),
                    barcode=row["barcode"],
                    label=int(row["label"]),
                    read_id=row["read_id"],
                    part_id=int(row["part_id"]),
                    part_base=row["part_base"],
                    part_read_index=int(row["part_read_index"]),
                    chunk_start=int(row["chunk_start"]),
                    n_chunks=int(row["n_chunks"]),
                    chunk_path=row["chunk_path"],
                    ccf_file=row["ccf_file"],
                    ccf_read_index=int(row["ccf_read_index"]),
                )
            )
    for records in by_split.values():
        records.sort(key=lambda record: (record.label, record.split_order))
    return by_split


class ChunkBagDataset(Dataset):
    def __init__(self, records: list[BagRecord], max_chunks: int, training: bool, seed: int):
        self.records = records
        self.max_chunks = int(max_chunks)
        self.training = bool(training)
        self.rng = np.random.default_rng(seed)
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _array(self, path: str) -> np.ndarray:
        if path not in self._arrays:
            self._arrays[path] = np.load(path, mmap_mode="r")
        return self._arrays[path]

    def _indices(self, n_chunks: int) -> np.ndarray:
        if n_chunks <= self.max_chunks:
            return np.arange(n_chunks, dtype=np.int64)
        if self.training:
            return np.sort(self.rng.choice(n_chunks, size=self.max_chunks, replace=False)).astype(np.int64)
        return np.linspace(0, n_chunks - 1, num=self.max_chunks, dtype=np.int64)

    def __getitem__(self, index: int):
        record = self.records[index]
        local = self._indices(record.n_chunks)
        array = self._array(record.chunk_path)
        chunks = np.asarray(array[record.chunk_start + local], dtype=np.float32)
        return chunks, record.label, record.read_id


def mil_collate(batch):
    max_chunks = max(item[0].shape[0] for item in batch)
    feature_dim = batch[0][0].shape[1]
    x = np.zeros((len(batch), max_chunks, feature_dim), dtype=np.float32)
    mask = np.zeros((len(batch), max_chunks), dtype=bool)
    labels = np.empty((len(batch),), dtype=np.int64)
    read_ids = []
    for row, (chunks, label, read_id) in enumerate(batch):
        n_chunks = chunks.shape[0]
        x[row, :n_chunks] = chunks
        mask[row, :n_chunks] = True
        labels[row] = label
        read_ids.append(read_id)
    return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(labels), read_ids

