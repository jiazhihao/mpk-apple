"""Model packages. One directory per architecture, registered by its HF ``architectures[0]`` name; the only place
model names appear (design D16). Importing this package imports every model package so the registry is complete."""

from .registry import MODELS, register_model, resolve_model
from . import qwen3_5  # noqa: F401  (registration)

__all__ = ["MODELS", "register_model", "resolve_model"]
