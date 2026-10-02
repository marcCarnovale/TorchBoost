"""Portable checkpoint metadata without importing arbitrary pickle globals."""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import torch


def portable_state(value):
    """Recursively reduce state to tensors and restricted-unpickler primitives.

    In particular, NumPy scalar values can arise from physical-state arithmetic
    and must not force callers to disable ``weights_only=True`` on reload.
    Unsupported application objects fail before a checkpoint is written.
    """
    if isinstance(value, np.generic):
        return portable_state(value.item())
    if isinstance(value, np.ndarray):
        return portable_state(value.tolist())
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Mapping):
        return {portable_state(k): portable_state(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(portable_state(v) for v in value)
    if isinstance(value, list):
        return [portable_state(v) for v in value]
    raise TypeError(f"unsupported checkpoint state type {type(value).__name__}")
