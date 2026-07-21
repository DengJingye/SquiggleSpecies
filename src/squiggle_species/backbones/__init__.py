"""Extensible raw-signal backbone contracts and registry."""

from .base import AdaptationSummary, BackboneSpec, SignalBackboneAdapter
from .registry import create_backbone, load_factory, register_backbone, registered_backbones
from .bonito import BonitoPrefixAdapter

__all__ = [
    "AdaptationSummary",
    "BackboneSpec",
    "SignalBackboneAdapter",
    "BonitoPrefixAdapter",
    "create_backbone",
    "load_factory",
    "register_backbone",
    "registered_backbones",
]
