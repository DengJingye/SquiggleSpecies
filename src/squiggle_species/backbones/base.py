from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BackboneSpec:
    """Machine-readable contract declared by each signal backbone adapter."""

    backbone_id: str
    feature_dim: int
    preprocessing_profiles: tuple[str, ...]
    trainable_units: tuple[str, ...]
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AdaptationSummary:
    mode: str
    selected_units: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int

    def to_dict(self) -> dict:
        return asdict(self)


class SignalBackboneAdapter(nn.Module, ABC):
    """Common interface for Bonito, local foundation models, and future models.

    An adapter owns only the chunk encoder. Read aggregation and classification
    remain separate toolkit modules. Each adapter defines its own ordered
    trainable units, so ``unfreeze_last_n=3`` is meaningful for that backbone
    rather than a global assumption about model depth.
    """

    @property
    @abstractmethod
    def spec(self) -> BackboneSpec:
        raise NotImplementedError

    @abstractmethod
    def adaptation_units(self) -> Mapping[str, nn.Module]:
        """Return ordered, non-overlapping modules eligible for fine-tuning."""

    @abstractmethod
    def encode_chunks(self, raw_chunks: torch.Tensor) -> torch.Tensor:
        """Encode ``[n_chunks, signal_length]`` into ``[n_chunks, feature_dim]``."""

    def validate_preprocessing(self, profile_id: str) -> None:
        if profile_id not in self.spec.preprocessing_profiles:
            allowed = ", ".join(self.spec.preprocessing_profiles)
            raise ValueError(
                f"Backbone {self.spec.backbone_id!r} is incompatible with preprocessing "
                f"profile {profile_id!r}; allowed profiles: {allowed}"
            )

    def configure_adaptation(
        self,
        *,
        mode: str,
        unfreeze_last_n: int | None = None,
        trainable_units: Sequence[str] | None = None,
    ) -> AdaptationSummary:
        """Freeze the backbone or select adapter-specific trainable units."""

        if mode not in {"frozen", "partial_finetune"}:
            raise ValueError(f"Unsupported adaptation mode: {mode}")
        if unfreeze_last_n is not None and trainable_units is not None:
            raise ValueError("Use either unfreeze_last_n or trainable_units, not both")

        units = self.adaptation_units()
        declared = tuple(units)
        if declared != self.spec.trainable_units:
            raise ValueError(
                f"Adapter unit order {declared!r} does not match spec {self.spec.trainable_units!r}"
            )
        for parameter in self.parameters():
            parameter.requires_grad = False

        selected: tuple[str, ...] = ()
        if mode == "partial_finetune":
            if trainable_units is not None:
                selected = tuple(trainable_units)
            elif unfreeze_last_n is not None:
                if unfreeze_last_n < 1 or unfreeze_last_n > len(declared):
                    raise ValueError(
                        f"unfreeze_last_n must be between 1 and {len(declared)} for "
                        f"{self.spec.backbone_id}, got {unfreeze_last_n}"
                    )
                selected = declared[-unfreeze_last_n:]
            else:
                raise ValueError("partial_finetune requires unfreeze_last_n or trainable_units")

            unknown = sorted(set(selected) - set(declared))
            if unknown:
                raise ValueError(f"Unknown trainable units for {self.spec.backbone_id}: {unknown}")
            for name in selected:
                for parameter in units[name].parameters():
                    parameter.requires_grad = True

        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return AdaptationSummary(
            mode=mode,
            selected_units=selected,
            trainable_parameters=trainable,
            total_parameters=total,
        )
