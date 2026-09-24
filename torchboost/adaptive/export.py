"""Versioned NumPy-only inference for ragged forest exports, soft or hard."""
from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .data import Preprocessor
    from .forest import AdaptiveForest


class _NumpyPreprocessor:
    """Standalone inference reads only the already-fitted normalization state."""
    def __init__(self, state):
        for name in ("mean", "scale", "target_mean", "target_scale", "classes"):
            setattr(self, name, None if state[name] is None else np.asarray(state[name]))
        self.task, self.output_dim = state["task"], state["output_dim"]
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("invalid exported normalization")

    def transform_x(self, X):
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 2 or min(x.shape) < 1 or x.shape[1] != len(self.mean):
            raise ValueError("invalid feature matrix for export")
        value = (np.where(np.isfinite(x), x, self.mean) - self.mean) / self.scale
        if not np.isfinite(value).all() or np.max(np.abs(value)) > np.finfo(np.float32).max:
            raise ValueError("nonfinite exported preprocessing result")
        return value.astype(np.float32).astype(np.float64)

    def inverse_target(self, y):
        return y * self.target_scale + self.target_mean


def export_model(model: AdaptiveForest, preprocessor: Preprocessor, *, hard: bool = True) -> dict:
    trees = []
    for tree in model.trees:
        nodes = {}
        for node in tree.nodes.values():
            nodes[node.node_id] = {"value": node.value.detach().cpu().tolist(),
                                   "children": list(node.children_ids), "active": node.active,
                                   "structural": None if node.structural is None else float(node.structural.detach()),
                                   "temperature": float(node.temperature),
                                   "weight": None if node.routing_weight is None else node.routing_weight.detach().cpu().tolist(),
                                   "bias": None if node.routing_bias is None else node.routing_bias.detach().cpu().tolist()}
        trees.append({"root": tree.root_id, "nodes": nodes, "feature_mask": tree.feature_mask.cpu().tolist()})
    return {"format": "torchboost.adaptive.numpy", "version": 1, "hard": hard,
            "preprocessor": preprocessor.state_dict(), "trees": trees,
            "bias": model.bias.detach().cpu().tolist(), "aggregation": model.config.aggregation,
            "attention_weight": model.attention_weight.detach().cpu().tolist(),
            "attention_bias": model.attention_bias.detach().cpu().tolist(),
            "residual": (model.residual_logits.detach().sigmoid().cpu().tolist()
                         if model.config.residual_weights else [[1.] for _ in model.trees]),
            "shrinkage": model.config.shrinkage}


def _softmax(value: np.ndarray, axis: int) -> np.ndarray:
    value = value - value.max(axis=axis, keepdims=True)
    exponent = np.exp(value)
    return exponent / exponent.sum(axis=axis, keepdims=True)


class ExportedForest:
    def __init__(self, specification: dict | str | Path):
        if isinstance(specification, (str, Path)):
            specification = json.loads(Path(specification).read_text())
        if specification.get("format") != "torchboost.adaptive.numpy" or specification.get("version") != 1:
            raise ValueError("unsupported export format")
        self.specification = specification
        self.preprocessor = _NumpyPreprocessor(specification["preprocessor"])
        for tree in specification["trees"]:
            visited = set()
            def visit(key: str) -> None:
                if key in visited or key not in tree["nodes"]:
                    raise ValueError("export contains a cycle, shared child, or missing node")
                visited.add(key)
                node = tree["nodes"][key]
                if node["children"] and len(node["weight"]) != len(node["children"]):
                    raise ValueError("routing arity mismatch")
                if node["temperature"] <= 0:
                    raise ValueError("invalid temperature")
                for child in node["children"]:
                    visit(child)
            visit(tree["root"])
            if visited != set(tree["nodes"]):
                raise ValueError("export contains unreachable allocated nodes")

    @classmethod
    def load(cls, path: str | Path):
        return cls(path)

    def decision_function(self, X) -> np.ndarray:
        spec = self.specification
        x = self.preprocessor.transform_x(X)
        outputs, active = [], []
        for tree in spec["trees"]:
            tx = x * np.asarray(tree["feature_mask"])
            active.append(tree["nodes"][tree["root"]]["active"])
            def visit(key: str) -> np.ndarray:
                node = tree["nodes"][key]
                if not node["active"]:
                    return np.zeros((len(x), self.preprocessor.output_dim))
                result = np.broadcast_to(np.asarray(node["value"]), (len(x), self.preprocessor.output_dim)).copy()
                if node["children"]:
                    score = (tx @ np.asarray(node["weight"]).T + np.asarray(node["bias"])) / node["temperature"]
                    p = np.eye(len(node["children"]))[score.argmax(1)] if spec["hard"] else _softmax(score, 1)
                    gate = 1. if node["structural"] is None else np.clip(node["structural"], 0., 1.)
                    if spec["hard"] and node["structural"] is not None:
                        gate = float(node["structural"] >= .5)
                    children = np.stack([visit(child) for child in node["children"]], 1)
                    result += gate * (p[:, :, None] * children).sum(1)
                return result
            outputs.append(visit(tree["root"]))
        outputs = np.stack(outputs, 1)
        active = np.asarray(active, dtype=bool)
        if not active.any():
            raise ValueError("all exported trees are inactive")
        if spec["aggregation"] == "attention":
            scores = np.einsum("nd,thd->nth", x, np.asarray(spec["attention_weight"])) + np.asarray(spec["attention_bias"])
            scores[:, ~active] = -np.inf
            weights = _softmax(scores, 1)
        else:
            weights = np.broadcast_to(active[None, :, None], (len(x), len(active), 1)).astype(float)
            if spec["aggregation"] == "mean":
                weights /= active.sum()
        result = np.asarray(spec["bias"]) + (weights * np.asarray(spec["residual"]) * spec["shrinkage"] * outputs).sum(1)
        return result

    def predict_proba(self, X) -> np.ndarray:
        logits = self.decision_function(X)
        if self.preprocessor.task == "binary":
            positive = 1. / (1 + np.exp(-np.clip(logits, -700, 700)))
            return np.concatenate((1 - positive, positive), 1)
        if self.preprocessor.task == "multiclass":
            return _softmax(logits, 1)
        raise ValueError("regression exports do not have class probabilities")

    def predict(self, X) -> np.ndarray:
        if self.preprocessor.classes is not None:
            return self.preprocessor.classes[self.predict_proba(X).argmax(1)]
        value = self.preprocessor.inverse_target(self.decision_function(X))
        return value[:, 0] if self.preprocessor.output_dim == 1 else value
