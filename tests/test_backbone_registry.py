from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from squiggle_species.backbones import (
    BackboneSpec,
    SignalBackboneAdapter,
    create_backbone,
    register_backbone,
    registered_backbones,
)


class DummyBackbone(SignalBackboneAdapter):
    def __init__(self, feature_dim: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(6, 6), nn.Linear(6, feature_dim)])
        self._spec = BackboneSpec(
            backbone_id="dummy-signal-v1",
            feature_dim=feature_dim,
            preprocessing_profiles=("dummy-profile-v1",),
            trainable_units=("block0", "block1"),
        )

    @property
    def spec(self) -> BackboneSpec:
        return self._spec

    def adaptation_units(self):
        return {"block0": self.blocks[0], "block1": self.blocks[1]}

    def encode_chunks(self, raw_chunks: torch.Tensor) -> torch.Tensor:
        return self.blocks[1](torch.relu(self.blocks[0](raw_chunks)))


@register_backbone("test-dummy")
def make_dummy(feature_dim: int = 4) -> DummyBackbone:
    return DummyBackbone(feature_dim=feature_dim)


class BackboneRegistryTest(unittest.TestCase):
    def test_builtin_bonito_adapter_is_registered_lazily(self) -> None:
        self.assertIn("bonito-prefix-0-9", registered_backbones())

    def test_registered_factory_and_adapter_specific_unfreeze(self) -> None:
        backbone = create_backbone({"adapter": "test-dummy", "kwargs": {"feature_dim": 3}})
        backbone.validate_preprocessing("dummy-profile-v1")
        summary = backbone.configure_adaptation(mode="partial_finetune", unfreeze_last_n=1)
        self.assertEqual(backbone.spec.feature_dim, 3)
        self.assertEqual(summary.selected_units, ("block1",))
        self.assertFalse(any(parameter.requires_grad for parameter in backbone.blocks[0].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in backbone.blocks[1].parameters()))

    def test_preprocessing_mismatch_is_rejected(self) -> None:
        backbone = DummyBackbone()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            backbone.validate_preprocessing("legacy-stone-v1")

    def test_named_units_are_supported(self) -> None:
        backbone = DummyBackbone()
        summary = backbone.configure_adaptation(mode="partial_finetune", trainable_units=["block0"])
        self.assertEqual(summary.selected_units, ("block0",))
        self.assertTrue(all(parameter.requires_grad for parameter in backbone.blocks[0].parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in backbone.blocks[1].parameters()))


if __name__ == "__main__":
    unittest.main()
