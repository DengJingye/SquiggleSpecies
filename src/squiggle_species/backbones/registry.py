from __future__ import annotations

from importlib import import_module
from typing import Callable

from .base import SignalBackboneAdapter


BackboneFactory = Callable[..., SignalBackboneAdapter]
_REGISTRY: dict[str, BackboneFactory] = {}


def register_backbone(name: str, factory: BackboneFactory | None = None):
    """Register a factory directly or use as ``@register_backbone(name)``."""

    def decorator(candidate: BackboneFactory) -> BackboneFactory:
        if name in _REGISTRY and _REGISTRY[name] is not candidate:
            raise ValueError(f"Backbone {name!r} is already registered")
        _REGISTRY[name] = candidate
        return candidate

    return decorator(factory) if factory is not None else decorator


def registered_backbones() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def load_factory(locator: str) -> BackboneFactory:
    """Load an external factory from ``package.module:factory_name``."""

    if ":" not in locator:
        raise ValueError("External backbone adapter must use the form 'module:factory'")
    module_name, object_name = locator.split(":", 1)
    factory = getattr(import_module(module_name), object_name)
    if not callable(factory):
        raise TypeError(f"Backbone factory {locator!r} is not callable")
    return factory


def create_backbone(config: dict) -> SignalBackboneAdapter:
    """Build a registered or dynamically imported backbone adapter."""

    locator = str(config.get("adapter", "")).strip()
    if not locator:
        raise ValueError("Backbone config must declare 'adapter'")
    factory = _REGISTRY.get(locator) or load_factory(locator)
    instance = factory(**dict(config.get("kwargs", {})))
    if not isinstance(instance, SignalBackboneAdapter):
        raise TypeError(
            f"Factory {locator!r} returned {type(instance).__name__}; expected SignalBackboneAdapter"
        )
    return instance
