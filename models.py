"""Extracted study functions; use the validated public API where available."""
from __future__ import annotations
from typing import Any

def require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required only for actual model fitting") from exc
    return torch, nn


def make_binary_predictor(input_dim: int, width: int = 64):
    """Identical architecture for A, B, and C."""
    _, nn = require_torch()
    return nn.Sequential(
        nn.Linear(input_dim, width),
        nn.ReLU(),
        nn.Linear(width, 1),
    )


def make_gold_classifier(input_dim: int, width: int = 64):
    """Identical architecture for D and E except for their training loss."""
    _, nn = require_torch()
    return nn.Sequential(
        nn.Linear(input_dim, width),
        nn.ReLU(),
        nn.Linear(width, 3),
    )

