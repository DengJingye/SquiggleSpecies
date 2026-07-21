from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from .base import BackboneSpec, SignalBackboneAdapter
from .registry import register_backbone


class BonitoPrefixAdapter(SignalBackboneAdapter):
    """Adapter for the validated Bonito encoder prefix ``children[0:9]``."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        prefix_layers: int = 9,
        recurrent_start: int = 4,
        feature_dim: int = 768,
        chunk_microbatch: int = 64,
        preprocessing_profiles: tuple[str, ...] = ("legacy-stone-v1", "apple-sclamp-v1"),
    ):
        super().__init__()
        from bonito.util import load_model

        loaded = load_model(str(model_dir), device=device)
        children = list(loaded.encoder.children())
        if len(children) < prefix_layers:
            raise ValueError(f"Expected at least {prefix_layers} Bonito encoder layers, got {len(children)}")
        if not 0 <= recurrent_start < prefix_layers:
            raise ValueError("recurrent_start must identify a layer inside the selected prefix")
        self.layers = nn.ModuleList(children[:prefix_layers])
        self.feature_dim = int(feature_dim)
        self.chunk_microbatch = int(chunk_microbatch)
        self.recurrent_start = int(recurrent_start)
        unit_names = tuple(f"encoder.layer{index}" for index in range(recurrent_start, prefix_layers))
        self._spec = BackboneSpec(
            backbone_id=f"bonito-prefix-0-{prefix_layers}",
            feature_dim=self.feature_dim,
            preprocessing_profiles=tuple(preprocessing_profiles),
            trainable_units=unit_names,
            description="Bonito encoder prefix with adapter-specific recurrent suffix units.",
        )
        del loaded

    @property
    def spec(self) -> BackboneSpec:
        return self._spec

    def adaptation_units(self):
        return OrderedDict(
            (f"encoder.layer{index}", self.layers[index])
            for index in range(self.recurrent_start, len(self.layers))
        )

    def _encode_part(self, raw_chunks: torch.Tensor) -> torch.Tensor:
        hidden = raw_chunks.unsqueeze(1)
        trainable_indices = [
            index
            for index, layer in enumerate(self.layers)
            if any(parameter.requires_grad for parameter in layer.parameters())
        ]
        first_trainable = min(trainable_indices) if trainable_indices else len(self.layers)
        with torch.no_grad():
            for layer in self.layers[:first_trainable]:
                hidden = layer(hidden)
        hidden = hidden.detach()
        for layer in self.layers[first_trainable:]:
            hidden = layer(hidden)
        if hidden.ndim != 3 or hidden.shape[-1] != self.feature_dim:
            raise ValueError(f"Unexpected Bonito output shape: {tuple(hidden.shape)}")
        expected_batch = raw_chunks.shape[0]
        if hidden.shape[1] == expected_batch:
            return hidden.mean(dim=0)
        if hidden.shape[0] == expected_batch:
            return hidden.mean(dim=1)
        raise ValueError(f"Cannot locate batch dimension in Bonito output: {tuple(hidden.shape)}")

    def encode_chunks(self, raw_chunks: torch.Tensor) -> torch.Tensor:
        parts = [
            self._encode_part(raw_chunks[start : start + self.chunk_microbatch])
            for start in range(0, raw_chunks.shape[0], self.chunk_microbatch)
        ]
        if not parts:
            return raw_chunks.new_empty((0, self.feature_dim))
        return torch.cat(parts, dim=0)


@register_backbone("bonito-prefix-0-9")
def create_bonito_prefix_adapter(**kwargs) -> BonitoPrefixAdapter:
    return BonitoPrefixAdapter(**kwargs)
