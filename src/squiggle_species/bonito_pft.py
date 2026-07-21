from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .models import SignalStudent


@dataclass(frozen=True)
class RawBagRecord:
    split: str
    split_order: int
    barcode: str
    label: int
    read_id: str
    raw_chunk_path: str
    raw_chunk_start: int
    raw_n_chunks: int
    legacy_chunk_path: str
    legacy_chunk_start: int
    ccf_file: str
    preprocessing_profile_id: str


def read_raw_manifest(path: str | Path) -> dict[str, list[RawBagRecord]]:
    by_split: dict[str, list[RawBagRecord]] = {"train": [], "val": [], "test": []}
    manifest_path = Path(path).resolve()
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in by_split:
                continue
            by_split[split].append(
                RawBagRecord(
                    split=split,
                    split_order=int(row["split_order"]),
                    barcode=row["barcode"],
                    label=int(row["label"]),
                    read_id=row["read_id"],
                    raw_chunk_path=str(
                        Path(row["raw_chunk_path"])
                        if Path(row["raw_chunk_path"]).is_absolute()
                        else manifest_path.parent / row["raw_chunk_path"]
                    ),
                    raw_chunk_start=int(row["raw_chunk_start"]),
                    raw_n_chunks=int(row["raw_n_chunks"]),
                    legacy_chunk_path=row.get("chunk_path", ""),
                    legacy_chunk_start=int(row.get("chunk_start", 0) or 0),
                    ccf_file=row.get("ccf_file", ""),
                    preprocessing_profile_id=row.get("preprocessing_profile_id", ""),
                )
            )
    for records in by_split.values():
        records.sort(key=lambda record: (record.label, record.split_order))
    return by_split


class RawChunkBagDataset(Dataset):
    def __init__(self, records: list[RawBagRecord], max_chunks: int, training: bool, seed: int):
        self.records = records
        self.max_chunks = int(max_chunks)
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _array(self, path: str) -> np.ndarray:
        if path not in self._arrays:
            self._arrays[path] = np.load(path, mmap_mode="r")
        return self._arrays[path]

    def _indices(self, index: int, n_chunks: int) -> np.ndarray:
        if n_chunks <= self.max_chunks:
            return np.arange(n_chunks, dtype=np.int64)
        if self.training:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            return np.sort(rng.choice(n_chunks, size=self.max_chunks, replace=False)).astype(np.int64)
        return np.linspace(0, n_chunks - 1, num=self.max_chunks, dtype=np.int64)

    def __getitem__(self, index: int):
        record = self.records[index]
        local = self._indices(index, record.raw_n_chunks)
        array = self._array(record.raw_chunk_path)
        chunks = np.asarray(array[record.raw_chunk_start + local], dtype=np.float32)
        return chunks, record.label, record.read_id


def raw_mil_collate(batch):
    max_chunks = max(item[0].shape[0] for item in batch)
    signal_length = batch[0][0].shape[1]
    x = np.zeros((len(batch), max_chunks, signal_length), dtype=np.float32)
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


class BonitoPartialStudent(nn.Module):
    def __init__(
        self,
        bonito_model: nn.Module,
        student: SignalStudent,
        trainable_lstm_blocks: int,
        chunk_microbatch: int,
    ):
        super().__init__()
        children = list(bonito_model.encoder.children())
        if len(children) < 9:
            raise ValueError(f"Expected at least 9 Bonito encoder children, got {len(children)}")
        if trainable_lstm_blocks not in {1, 2, 3, 4, 5}:
            raise ValueError("trainable_lstm_blocks must be between 1 and 5")
        self.encoder_layers = nn.ModuleList(children[:9])
        self.train_start = 9 - trainable_lstm_blocks
        self.trainable_lstm_blocks = trainable_lstm_blocks
        self.chunk_microbatch = int(chunk_microbatch)
        self.student = student
        for parameter in self.encoder_layers.parameters():
            parameter.requires_grad = False
        for layer in self.encoder_layers[self.train_start :]:
            # Bonito loads CUDA inference weights as FP16. Keep trainable master
            # weights in FP32 so AMP/GradScaler can unscale them safely.
            layer.float()
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        for layer in self.encoder_layers[: self.train_start]:
            layer.eval()
        return self

    def trainable_backbone_parameters(self):
        for layer in self.encoder_layers[self.train_start :]:
            yield from layer.parameters()

    def _encode_microbatch(self, raw_chunks: torch.Tensor) -> torch.Tensor:
        hidden = raw_chunks.unsqueeze(1)
        with torch.no_grad():
            for layer in self.encoder_layers[: self.train_start]:
                hidden = layer(hidden)
        hidden = hidden.detach()
        for layer in self.encoder_layers[self.train_start :]:
            hidden = layer(hidden)
        if hidden.ndim != 3 or hidden.shape[-1] != 768:
            raise ValueError(f"Unexpected Bonito prefix output shape: {tuple(hidden.shape)}")
        if hidden.shape[1] == raw_chunks.shape[0]:
            return hidden.mean(dim=0)
        if hidden.shape[0] == raw_chunks.shape[0]:
            return hidden.mean(dim=1)
        raise ValueError(f"Cannot locate batch dimension in Bonito output: {tuple(hidden.shape)}")

    @staticmethod
    def _mean_time(hidden: torch.Tensor, expected_batch: int) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != 768:
            raise ValueError(f"Unexpected Bonito prefix output shape: {tuple(hidden.shape)}")
        if hidden.shape[1] == expected_batch:
            return hidden.mean(dim=0)
        if hidden.shape[0] == expected_batch:
            return hidden.mean(dim=1)
        raise ValueError(f"Cannot locate batch dimension in Bonito output: {tuple(hidden.shape)}")

    def encode_chunks(self, raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, chunks, _ = raw.shape
        flat_mask = mask.reshape(-1)
        valid_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
        flat_raw = raw.reshape(batch * chunks, -1).index_select(0, valid_indices)
        encoded_parts = []
        for start in range(0, flat_raw.shape[0], self.chunk_microbatch):
            encoded_parts.append(self._encode_microbatch(flat_raw[start : start + self.chunk_microbatch]))
        encoded = torch.cat(encoded_parts, dim=0)
        dense = encoded.new_zeros((batch * chunks, encoded.shape[1]))
        dense = dense.index_copy(0, valid_indices, encoded)
        return dense.reshape(batch, chunks, encoded.shape[1])

    def encode_chunks_manifold_mix(
        self,
        raw: torch.Tensor,
        mask: torch.Tensor,
        permutation: torch.Tensor,
        lam: torch.Tensor,
        mix_layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix read-aligned Bonito hidden states before a trainable LSTM layer.

        This mirrors DNABERT-S MI-Mix more closely than mixing final read
        embeddings. Padded chunk positions are removed with the intersection
        mask, analogous to intersecting attention masks in DNABERT-S.
        """
        if mix_layer < self.train_start or mix_layer >= len(self.encoder_layers):
            raise ValueError(
                f"mix_layer must be in [{self.train_start}, {len(self.encoder_layers) - 1}], got {mix_layer}"
            )
        batch, chunks, _ = raw.shape
        flat_raw = raw.reshape(batch * chunks, -1)
        prefix_parts = []
        for start in range(0, flat_raw.shape[0], self.chunk_microbatch):
            hidden = flat_raw[start : start + self.chunk_microbatch].unsqueeze(1)
            with torch.no_grad():
                for layer in self.encoder_layers[: self.train_start]:
                    hidden = layer(hidden)
            hidden = hidden.detach()
            for layer in self.encoder_layers[self.train_start : mix_layer]:
                hidden = layer(hidden)
            prefix_parts.append(hidden)
        hidden = torch.cat(prefix_parts, dim=1)
        if hidden.ndim != 3 or hidden.shape[1] != batch * chunks:
            raise ValueError(f"Unexpected pre-mix Bonito shape: {tuple(hidden.shape)}")
        hidden = hidden.reshape(hidden.shape[0], batch, chunks, hidden.shape[-1])
        lam_hidden = lam.to(hidden.dtype).reshape(1, batch, 1, 1)
        hidden = lam_hidden * hidden + (1.0 - lam_hidden) * hidden[:, permutation]
        mixed_mask = mask & mask[permutation]
        hidden = hidden.reshape(hidden.shape[0], batch * chunks, hidden.shape[-1])

        encoded_parts = []
        for start in range(0, batch * chunks, self.chunk_microbatch):
            part = hidden[:, start : start + self.chunk_microbatch]
            for layer in self.encoder_layers[mix_layer:]:
                part = layer(part)
            encoded_parts.append(self._mean_time(part, part.shape[1]))
        encoded = torch.cat(encoded_parts, dim=0).reshape(batch, chunks, -1)
        return encoded, mixed_mask

    def forward(self, raw: torch.Tensor, mask: torch.Tensor):
        chunk_embeddings = self.encode_chunks(raw, mask)
        logits, embedding, attention = self.student(chunk_embeddings.float(), mask)
        return logits, embedding, attention

    def forward_manifold_mix(
        self,
        raw: torch.Tensor,
        mask: torch.Tensor,
        permutation: torch.Tensor,
        lam: torch.Tensor,
        mix_layer: int,
    ):
        chunk_embeddings, mixed_mask = self.encode_chunks_manifold_mix(raw, mask, permutation, lam, mix_layer)
        logits, embedding, attention = self.student(chunk_embeddings.float(), mixed_mask)
        return logits, embedding, attention, mixed_mask


def augment_signal(raw: torch.Tensor, mask: torch.Tensor, config: dict) -> torch.Tensor:
    output = raw.clone()
    batch, chunks, length = output.shape
    amplitude = float(config.get("amplitude_jitter", 0.0))
    offset = float(config.get("offset_jitter", 0.0))
    noise_std = float(config.get("gaussian_noise_std", 0.0))
    if amplitude > 0:
        scale = 1.0 + (torch.rand((batch, chunks, 1), device=output.device) * 2.0 - 1.0) * amplitude
        output = output * scale
    if offset > 0:
        shift = (torch.rand((batch, chunks, 1), device=output.device) * 2.0 - 1.0) * offset
        output = output + shift
    if noise_std > 0:
        output = output + torch.randn_like(output) * noise_std
    mask_points = min(int(config.get("time_mask_points", 0)), length)
    mask_probability = float(config.get("time_mask_probability", 0.0))
    if mask_points > 0 and mask_probability > 0:
        apply_mask = (torch.rand((batch, chunks), device=output.device) < mask_probability) & mask
        for row, col in torch.nonzero(apply_mask, as_tuple=False):
            start = int(torch.randint(0, max(1, length - mask_points + 1), (1,), device=output.device).item())
            output[row, col, start : start + mask_points] = 0.0
    return output * mask.unsqueeze(-1)


def symmetric_consistency(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    target = F.softmax(teacher_logits.detach(), dim=1)
    return F.kl_div(F.log_softmax(student_logits, dim=1), target, reduction="batchmean")
