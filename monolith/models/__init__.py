"""Model packages. One directory per architecture, registered by its HF ``architectures[0]`` name; the only place
model names appear (design D16)."""

from .registry import MODELS, register_model, resolve_model

__all__ = ["MODELS", "register_model", "resolve_model"]
