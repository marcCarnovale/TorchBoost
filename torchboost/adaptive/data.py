"""Training-only preprocessing, label contracts, and aligned sample weights."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from torch import Tensor


@dataclass
class DataSplit:
    x: Tensor
    y: Tensor
    weight: Tensor

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for tensor in (self.x, self.y, self.weight):
            array = tensor.detach().cpu().numpy()
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()


def check_x(x) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError("X must be a nonempty 2D numeric array")
    return value


def sample_weights(weight, n: int) -> np.ndarray:
    value = np.ones(n, dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    if value.shape != (n,) or not np.isfinite(value).all() or np.any(value < 0) or not np.any(value > 0):
        raise ValueError("sample weights must be a finite nonnegative vector with positive sum")
    # Losses depend only on relative weights. Rescale without overflow before
    # statistics/float32 conversion, preserving a positive mean-one vector.
    value = value / value.max()
    return value / value.mean()


class Preprocessor:
    def __init__(self):
        self.mean = self.scale = None
        self.target_mean = self.target_scale = None
        self.classes: np.ndarray | None = None
        self.task = ""
        self.output_dim = 0

    def fit(self, x, y, *, classification: bool, weights: np.ndarray) -> None:
        x = check_x(x)
        finite = np.isfinite(x)
        counts = (finite * weights[:, None]).sum(0)
        self.mean = (np.where(finite, x, 0.) * weights[:, None]).sum(0) / np.maximum(counts, 1e-12)
        difference = np.where(finite, x - self.mean, 0.)
        variance = (difference**2 * weights[:, None]).sum(0) / np.maximum(counts, 1e-12)
        self.scale = np.where(variance > 1e-12, np.sqrt(variance), 1.)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("feature statistics overflowed float64; rescale input units")
        y = np.asarray(y)
        if len(y) != len(x):
            raise ValueError("X and y length mismatch")
        if classification:
            if y.ndim != 1:
                raise ValueError("classification requires one categorical target per example")
            if y.dtype.kind in "fc" and not np.isfinite(y).all():
                raise ValueError("classification labels must be finite")
            self.classes = np.unique(y)
            if len(self.classes) < 2:
                raise ValueError("classification requires at least two classes")
            self.task = "binary" if len(self.classes) == 2 else "multiclass"
            self.output_dim = 1 if self.task == "binary" else len(self.classes)
        else:
            y = np.asarray(y, dtype=np.float64)
            if y.ndim == 1:
                y = y[:, None]
            if y.ndim != 2 or y.shape[1] < 1 or not np.isfinite(y).all():
                raise ValueError("regression y must be finite with shape [N] or [N, outputs]")
            self.task, self.output_dim = "regression", y.shape[1]
            self.target_mean = np.average(y, axis=0, weights=weights)
            variance = np.average((y - self.target_mean)**2, axis=0, weights=weights)
            self.target_scale = np.where(variance > 1e-12, np.sqrt(variance), 1.)
            if not np.isfinite(self.target_mean).all() or not np.isfinite(self.target_scale).all():
                raise ValueError("target statistics overflowed float64; rescale target units")

    def transform_x(self, x) -> Tensor:
        if self.mean is None:
            raise ValueError("preprocessor is not fitted")
        x = check_x(x)
        if x.shape[1] != len(self.mean):
            raise ValueError("feature count mismatch")
        result = (np.where(np.isfinite(x), x, self.mean) - self.mean) / self.scale
        # No arbitrary clipping: overflow is an explicit data/numerics error.
        if not np.isfinite(result).all() or np.max(np.abs(result)) > np.finfo(np.float32).max:
            raise ValueError("transformed features are outside finite float32 range")
        return torch.from_numpy(np.ascontiguousarray(result, dtype=np.float32))

    def transform_y(self, y) -> Tensor:
        value = np.asarray(y)
        if self.classes is not None:
            if value.ndim != 1:
                raise ValueError("classification labels must be 1D")
            mapping = {v: i for i, v in enumerate(self.classes.tolist())}
            try:
                encoded = np.asarray([mapping[v] for v in value.tolist()], dtype=np.int64)
            except (KeyError, TypeError) as error:
                raise ValueError("target contains a class absent from training") from error
            return torch.from_numpy(encoded)
        value = np.asarray(value, dtype=np.float64)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2 or value.shape[1] != self.output_dim or not np.isfinite(value).all():
            raise ValueError("invalid regression targets")
        return torch.from_numpy(np.ascontiguousarray((value - self.target_mean) / self.target_scale, dtype=np.float32))

    def split(self, x, y, weight=None) -> DataSplit:
        tx, ty = self.transform_x(x), self.transform_y(y)
        if len(tx) != len(ty):
            raise ValueError("X and y length mismatch")
        return DataSplit(tx, ty, torch.from_numpy(sample_weights(weight, len(tx)).astype(np.float32)))

    def inverse_target(self, prediction: np.ndarray) -> np.ndarray:
        return prediction * self.target_scale + self.target_mean

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "target_mean": None if self.target_mean is None else self.target_mean.tolist(),
                "target_scale": None if self.target_scale is None else self.target_scale.tolist(),
                "classes": None if self.classes is None else self.classes.tolist(),
                "task": self.task, "output_dim": self.output_dim}

    def load_state_dict(self, value: dict) -> None:
        for name in ("mean", "scale", "target_mean", "target_scale", "classes"):
            setattr(self, name, None if value[name] is None else np.asarray(value[name]))
        self.task, self.output_dim = value["task"], value["output_dim"]
