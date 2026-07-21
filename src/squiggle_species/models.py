from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalStudent(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 384,
        projection_dim: int = 256,
        attention_dim: int = 128,
        num_classes: int = 9,
        dropout: float = 0.15,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 512,
        max_position: int = 512,
        aggregation: str = "transformer",
    ):
        super().__init__()
        if aggregation not in {"mean", "attention", "transformer"}:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        self.aggregation = aggregation
        self.max_position = max_position
        self.chunk_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.position_embedding = nn.Embedding(max_position, projection_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.attention_v = nn.Linear(projection_dim, attention_dim)
        self.attention_u = nn.Linear(projection_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        batch, chunks, dim = x.shape
        h = self.chunk_encoder(x.reshape(batch * chunks, dim)).reshape(batch, chunks, -1)
        if self.aggregation == "transformer":
            positions = torch.arange(chunks, device=x.device).clamp(max=self.max_position - 1)
            h = h + self.position_embedding(positions).unsqueeze(0)
            h = self.transformer(h, src_key_padding_mask=~mask)
        if self.aggregation == "mean":
            attention = mask.to(h.dtype)
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            gated = torch.tanh(self.attention_v(h)) * torch.sigmoid(self.attention_u(h))
            scores = self.attention_w(gated).squeeze(-1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores, dim=1)
        bag = torch.sum(attention.unsqueeze(-1) * h, dim=1)
        embedding = F.normalize(self.dropout(bag), p=2, dim=1)
        return self.classifier(embedding), embedding, attention


class KmerTeacher(nn.Module):
    """Architecture-compatible loader for the existing sequence k-mer teacher."""

    def __init__(
        self,
        input_dim: int = 2080,
        hidden_dim: int = 1024,
        hidden_dim2: int = 512,
        projection_dim: int = 256,
        num_classes: int = 9,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim2, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(self, x: torch.Tensor):
        hidden = self.backbone(x)
        embedding = F.normalize(self.projector(hidden), p=2, dim=1)
        return self.classifier(embedding), embedding


def cross_modal_distillation_loss(
    student_logits: torch.Tensor,
    student_embedding: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_embedding: torch.Tensor,
    temperature: float,
    label_smoothing: float,
    distill_weight: float,
    alignment_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ce = F.cross_entropy(student_logits, labels, label_smoothing=label_smoothing)
    kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature**2)
    if alignment_weight > 0:
        if student_embedding.shape[1] != teacher_embedding.shape[1]:
            raise ValueError(
                f"Embedding alignment requires equal dimensions, got student={student_embedding.shape[1]} "
                f"teacher={teacher_embedding.shape[1]}"
            )
        alignment = (1.0 - F.cosine_similarity(student_embedding, teacher_embedding, dim=1)).mean()
    else:
        alignment = student_embedding.sum() * 0.0
    total = ce + distill_weight * kd + alignment_weight * alignment
    return total, {"ce": ce, "kd": kd, "alignment": alignment}
