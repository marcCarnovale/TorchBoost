"""Stagewise logistic Newton boosting, distinct from the legacy joint ensemble."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import Tensor, nn

from .control import CapacitorController
from .metrics import PerformanceTracker, SplitMetricsCollector
from .objectives import BinaryLogisticObjective
from .trees import BinarySoftTree


class _AdditiveModel(nn.Module):
    def __init__(self, base_score: float):
        super().__init__()
        self.register_buffer("base_score", torch.tensor(float(base_score)))
        self.register_buffer("rates", torch.zeros(0))
        self.trees = nn.ModuleList()

    def forward(self, x: Tensor, *, hard: bool = False) -> Tensor:
        score = self.base_score.expand(len(x)).clone()
        for rate, tree in zip(self.rates, self.trees):
            score = score + rate * tree(x, hard=hard)
        return score


class StagewiseBinaryClassifier(ClassifierMixin, BaseEstimator):
    """One differentiable soft tree per logistic boosting stage.

    Accepted stages are immutable; only the new candidate is optimized. `lr`
    scales each accepted tree; `optimizer_lr` controls its inner Adam updates.
    Training uses analytic logistic gradients/Hessians and a quadratic surrogate.
    A training-only backtracking line search checks the exact logistic loss before
    accepting a stage. `curvature='first_order'` uses unit curvature as an explicit
    residual-fitting ablation, not a purported Hessian.

    Optional CART initialization is a disclosed hybrid warm start. `init='random'`
    tests the fully gradient-initialized alternative. This is a small-data reference
    baseline, not the planned dynamically sparse engine. The dense leaf solve is
    O(2**(3*depth)) compute and O(2**(2*depth)) workspace in the worst case.

    X may contain NaNs: weighted means and scales are fitted on training data only.
    Entirely missing features become zero after transformation. Infinities fail.
    Data and cached ensemble scores remain on CPU; only minibatches enter the GPU.

    `eval_set` chooses the best accepted-stage prefix, including the intercept-only
    model. `control_set` drives the optional capacitor experiment; users must supply
    an independent development split. Neither is a final test set. Controller
    changes apply only to the new candidate, never to already accepted stages.

    Checkpoints support inference and inspection, not mid-optimizer training resume.
    """

    def __init__(self, n_estimators: int = 40, max_depth: int = 3, lr: float = 0.15,
                 epochs_per_stage: int = 20, optimizer_lr: float = 0.02,
                 batch_size: int = 512, leaf_l2: float = 1e-4, gate_l2: float = 1e-4,
                 temperature: float = 1.0, final_temperature: float = 0.5,
                 init: str = "cart", min_samples_leaf: int = 5,
                 curvature: str = "newton", hessian_floor: float = 1e-6,
                 patience: int = 10, random_state: int = 0, device: str = "cpu",
                 controller: dict | None = None, collect_metrics: bool = True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.epochs_per_stage = epochs_per_stage
        self.optimizer_lr = optimizer_lr
        self.batch_size = batch_size
        self.leaf_l2, self.gate_l2 = leaf_l2, gate_l2
        self.temperature, self.final_temperature = temperature, final_temperature
        self.init, self.min_samples_leaf = init, min_samples_leaf
        self.curvature, self.hessian_floor = curvature, hessian_floor
        self.patience, self.random_state, self.device = patience, random_state, device
        self.controller, self.collect_metrics = controller, collect_metrics

    def _validate_config(self) -> None:
        for name in ("n_estimators", "epochs_per_stage", "batch_size", "min_samples_leaf", "patience"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.max_depth, int) or not 1 <= self.max_depth <= 8:
            raise ValueError("max_depth must be in [1, 8] for the dense Newton solver")
        for name in ("lr", "optimizer_lr", "leaf_l2", "temperature", "final_temperature", "hessian_floor"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.gate_l2) or self.gate_l2 < 0:
            raise ValueError("gate_l2 must be finite and nonnegative")
        if self.init not in ("cart", "random") or self.curvature not in ("newton", "first_order"):
            raise ValueError("invalid initialization or curvature mode")
        if self.controller is not None and not isinstance(self.controller, dict):
            raise ValueError("controller must be a configuration dictionary or None")

    @staticmethod
    def _array(x) -> np.ndarray:
        if isinstance(x, Tensor):
            x = x.detach().cpu().numpy()
        result = np.asarray(x, dtype=np.float64)
        if result.ndim != 2 or min(result.shape) < 1 or np.isinf(result).any():
            raise ValueError("X must be a nonempty 2D array without infinities")
        return result

    @staticmethod
    def _weights(weight, n: int) -> Tensor:
        if weight is None:
            return torch.ones(n)
        w = torch.as_tensor(weight, dtype=torch.float64).detach().cpu().reshape(-1)
        if len(w) != n or not torch.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("sample weights must be finite, nonnegative, length n, with positive sum")
        return (w / w.mean()).float()

    def _transform(self, x) -> Tensor:
        array = self._array(x)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")
        array = np.where(np.isnan(array), self.center_, array)
        result = (array-self.center_)/self.scale_
        if not np.isfinite(result).all() or np.max(np.abs(result)) > np.finfo(np.float32).max:
            raise ValueError("preprocessing produced nonfinite or unrepresentable values")
        return torch.from_numpy(result.astype(np.float32))

    def _labels(self, y, n: int) -> Tensor:
        if isinstance(y, Tensor):
            y = y.detach().cpu().numpy()
        labels = np.asarray(y).reshape(-1)
        if len(labels) != n or not np.isin(labels, self.classes_).all():
            raise ValueError("labels must match the fitted binary classes and sample count")
        return torch.from_numpy((labels == self.classes_[1]).astype(np.float32))

    def _eval_data(self, data):
        if data is None:
            return None
        if len(data) not in (2, 3):
            raise ValueError("evaluation data must be (X, y) or (X, y, weights)")
        x = self._transform(data[0])
        return x, self._labels(data[1], len(x)), self._weights(data[2] if len(data) == 3 else None, len(x))

    def _predict_tensor(self, module: nn.Module, x: Tensor, *, hard: bool = False) -> Tensor:
        with torch.no_grad():
            return torch.cat([module(batch.to(self.device), hard=hard).cpu()
                              for batch in x.split(self.batch_size)])

    @staticmethod
    def _loss(score: Tensor, target: Tensor, weight: Tensor) -> float:
        loss = BinaryLogisticObjective.loss(score.double(), target.double())
        return float((loss*weight.double()).sum()/weight.double().sum())

    def fit(self, X, y, sample_weight=None, *, eval_set=None, control_set=None):
        """Fit from scratch; evaluation/control splits are explicit keyword-only roles."""
        self._validate_config()
        xraw = self._array(X)
        labels = y.detach().cpu().numpy() if isinstance(y, Tensor) else np.asarray(y)
        labels = labels.reshape(-1)
        if len(labels) != len(xraw):
            raise ValueError("X and y lengths differ")
        self.classes_ = np.unique(labels)
        if len(self.classes_) != 2:
            raise ValueError("StagewiseBinaryClassifier requires exactly two classes")
        self.n_features_in_ = xraw.shape[1]
        weight = self._weights(sample_weight, len(xraw))
        wnp = weight.numpy().astype(np.float64)[:, None]
        valid = ~np.isnan(xraw)
        denom = (wnp*valid).sum(0)
        self.center_ = np.divide((np.where(valid, xraw, 0)*wnp).sum(0), denom,
                                 out=np.zeros(self.n_features_in_), where=denom > 0)
        imputed = np.where(valid, xraw, self.center_)
        self.scale_ = np.sqrt(((imputed-self.center_)**2*wnp).sum(0)/wnp.sum())
        self.scale_ = np.where(self.scale_ > 1e-12, self.scale_, 1.0)
        x, target = self._transform(xraw), self._labels(labels, len(xraw))
        base = float(BinaryLogisticObjective.initial_score(target.double(), weight.double()))
        self.model_ = _AdditiveModel(base).to(self.device)
        self.tracker_ = PerformanceTracker()
        self.history_, self.control_history_ = [], []
        evaluation, control = self._eval_data(eval_set), self._eval_data(control_set)
        if self.controller is not None and control is None:
            raise ValueError("a separate control_set is required when enabling the capacitor")
        score = torch.full_like(target, base)
        eval_score = torch.full_like(evaluation[1], base) if evaluation else None
        control_score = torch.full_like(control[1], base) if control else None
        best_loss = self._loss(eval_score, evaluation[1], evaluation[2]) if evaluation else math.inf
        best_count, stale = 0, 0
        generator = torch.Generator().manual_seed(self.random_state)

        for stage in range(self.n_estimators):
            gradient, exact_hessian = BinaryLogisticObjective.derivatives(score, target)
            hessian = exact_hessian.clamp_min(self.hessian_floor) if self.curvature == "newton" else torch.ones_like(score)
            tree = BinarySoftTree(self.n_features_in_, self.max_depth, self.temperature,
                                  self.random_state+stage).to(self.device)
            if self.init == "cart":
                tree.initialize_cart(x, -gradient/hessian, weight*hessian,
                                     min_samples_leaf=self.min_samples_leaf, seed=self.random_state+stage)
            tree.solve_leaves(x, gradient, hessian, weight, self.leaf_l2, self.batch_size)
            optimizer = torch.optim.Adam(tree.parameters(), lr=self.optimizer_lr)
            collector = SplitMetricsCollector(tree.num_nodes, stage, self.device) if self.collect_metrics else None
            physical = None
            if self.controller is not None:
                config = {"initial_temperature": self.temperature, **self.controller}
                physical = CapacitorController(tree.num_nodes, **config).to(self.device)
                physical.observe_validation(self._loss(control_score, control[1], control[2]))
                tree.controller = physical

            for epoch in range(1, self.epochs_per_stage+1):
                if physical is None:
                    fraction = (epoch-1)/max(1, self.epochs_per_stage-1)
                    temp = self.final_temperature + 0.5*(self.temperature-self.final_temperature)*(1+math.cos(math.pi*fraction))
                    tree.set_temperature(temp)
                else:
                    tree.set_temperature(physical.temperature)
                if collector:
                    collector.reset_epoch()
                tree.train()
                order = torch.randperm(len(x), generator=generator)
                for indices in order.split(self.batch_size):
                    xb, yb, wb, gb, hb = [a[indices].to(self.device) for a in (x, target, weight, gradient, hessian)]
                    optimizer.zero_grad(set_to_none=True)
                    trace = tree.routing(xb)
                    output = trace.leaves @ tree.leaf_values
                    surrogate = (wb*(gb*output+0.5*hb*output.square())).mean()
                    regularization = 0.5*self.leaf_l2*tree.leaf_values.square().sum()
                    regularization += 0.5*self.gate_l2*tree.weights.square().mean()
                    loss = surrogate+regularization
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite stage objective")
                    loss.backward()
                    if collector:
                        collector.observe_batch(trace, yb, wb)
                        before_w, before_b = tree.weights.detach().clone(), tree.biases.detach().clone()
                        grad_w, grad_b = tree.weights.grad.clone(), tree.biases.grad.clone()
                    torch.nn.utils.clip_grad_norm_(tree.parameters(), 2.0, error_if_nonfinite=True)
                    optimizer.step()
                    if collector:
                        collector.observe_update(before_w, before_b, tree.weights, tree.biases, grad_w, grad_b)
                if collector:
                    self.tracker_.record(collector.finish_epoch(epoch))
                tree.eval()
                if physical is not None:
                    current = control_score+self.lr*self._predict_tensor(tree, control[0])
                    control_loss = self._loss(current, control[1], control[2])
                    physical.observe_validation(control_loss)
                    # Uniform resistances are the declared minimal-controller baseline.
                    state = physical.advance(torch.ones_like(physical.temperature))
                    self.control_history_.append({"stage": stage, "epoch": epoch,
                        "control_loss": control_loss, "charge": float(state["charge"]),
                        "injected_charge": float(physical.last_injected_charge),
                        "source_energy": float(physical.last_source_energy),
                        "heating": float(state["heat"].sum()), "cooling": float(state["cooling"].sum()),
                        "temperature_mean": float(state["temperature"].mean())})

            tree.solve_leaves(x, gradient, hessian, weight, self.leaf_l2, self.batch_size)
            direction = self._predict_tensor(tree, x)
            before_loss = self._loss(score, target, weight)
            rate = self.lr
            after_loss = math.inf
            for _ in range(16):
                after_loss = self._loss(score+rate*direction, target, weight)
                if math.isfinite(after_loss) and after_loss < before_loss-1e-12:
                    break
                rate *= 0.5
            else:
                self.history_.append({"stage": stage, "accepted": False, "train_loss": before_loss})
                break
            tree.requires_grad_(False)
            self.model_.trees.append(tree)
            self.model_.rates = torch.cat((self.model_.rates, self.model_.rates.new_tensor([rate])))
            score = score+rate*direction
            record = {"stage": stage, "accepted": True, "train_loss": after_loss,
                      "step_size": rate, "curvature_floor_fraction": float((exact_hessian < self.hessian_floor).float().mean())}
            if control:
                control_score = control_score+rate*self._predict_tensor(tree, control[0])
            if evaluation:
                eval_score = eval_score+rate*self._predict_tensor(tree, evaluation[0])
                val_loss = self._loss(eval_score, evaluation[1], evaluation[2])
                record["validation_loss"] = val_loss
                if val_loss < best_loss:
                    best_loss, best_count, stale = val_loss, len(self.model_.trees), 0
                else:
                    stale += 1
            self.history_.append(record)
            if evaluation and stale >= self.patience:
                break
        if evaluation:
            self.model_.trees = nn.ModuleList(list(self.model_.trees)[:best_count])
            self.model_.rates = self.model_.rates[:best_count].clone()
            self.best_validation_loss_ = best_loss
        self.n_estimators_ = len(self.model_.trees)
        self.model_.eval()
        return self

    def decision_function(self, X, *, hard: bool = False) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self._predict_tensor(self.model_, self._transform(X), hard=hard).numpy()

    def predict_proba(self, X, *, hard: bool = False) -> np.ndarray:
        score = torch.from_numpy(self.decision_function(X, hard=hard)).double()
        p = score.sigmoid().numpy()
        return np.column_stack((1-p, p))

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]

    def export_json(self, path: str | Path) -> None:
        """Write a versioned hard-inference model; measure soft/hard fidelity separately."""
        import json
        check_is_fitted(self, "model_")
        payload = {"schema": "torchboost.hard-binary.v1", "classes": self.classes_.tolist(),
                   "center": self.center_.tolist(), "scale": self.scale_.tolist(),
                   "base_score": float(self.model_.base_score),
                   "rates": self.model_.rates.cpu().tolist(),
                   "trees": [tree.to_dict() for tree in self.model_.trees]}
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")

    def save(self, path: str | Path) -> None:
        """Save an inference/inspection checkpoint; no unpickled estimator object."""
        check_is_fitted(self, "model_")
        payload = {"schema": 1, "params": copy.deepcopy(self.get_params(deep=False)),
                   "classes": self.classes_.tolist(), "center": self.center_.tolist(),
                   "scale": self.scale_.tolist(), "count": self.n_estimators_,
                   "model": {k: v.detach().cpu() for k, v in self.model_.state_dict().items()},
                   "history": self.history_, "control_history": self.control_history_,
                   "tracker": self.tracker_.state_dict(),
                   "best_validation_loss": getattr(self, "best_validation_loss_", None)}
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> StagewiseBinaryClassifier:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != 1:
            raise ValueError("unsupported checkpoint schema")
        params = {**payload["params"], "device": device}
        model = cls(**params)
        model.classes_ = np.asarray(payload["classes"])
        model.center_, model.scale_ = np.asarray(payload["center"]), np.asarray(payload["scale"])
        model.n_features_in_ = len(model.center_)
        model.n_estimators_ = payload["count"]
        model.model_ = _AdditiveModel(0)
        model.model_.rates = torch.zeros(model.n_estimators_)
        for i in range(model.n_estimators_):
            tree = BinarySoftTree(model.n_features_in_, model.max_depth)
            if model.controller is not None:
                tree.controller = CapacitorController(tree.num_nodes, **{"initial_temperature": model.temperature, **model.controller})
            model.model_.trees.append(tree)
        model.model_.load_state_dict(payload["model"])
        model.model_.to(device).requires_grad_(False).eval()
        model.history_, model.control_history_ = payload["history"], payload["control_history"]
        model.tracker_ = PerformanceTracker()
        model.tracker_.load_state_dict(payload["tracker"])
        if payload["best_validation_loss"] is not None:
            model.best_validation_loss_ = payload["best_validation_loss"]
        return model
