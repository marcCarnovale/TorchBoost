"""A task-general Newton reference using the same multiway routing kernel.

This is a new comparator, not a copy or replacement of the repository's original
StagewiseBinaryClassifier. It deliberately has no adaptive physical/plastic
control: earlier accepted stages are immutable, and all task Hessians are real.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import Tensor, nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .config import ForestConfig, StructureConfig
from .data import Preprocessor, sample_weights
from .forest import RaggedTree
from .objectives import Objective


@dataclass
class StagewiseConfig:
    n_estimators: int = 16
    depth: int = 2
    arity: int = 2
    inner_steps: int = 8
    learning_rate: float = .3
    optimizer_lr: float = .03
    leaf_l2: float = 1e-3
    gate_l2: float = 1e-4
    initial_temperature: float = 1.
    final_temperature: float = .5
    curvature: str = "newton"
    random_state: int = 0

    def __post_init__(self):
        for name in ("n_estimators", "depth", "arity", "inner_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.arity < 2 or self.depth > 5 or self.arity**self.depth > 128:
            raise ValueError("dense conditional Newton reference requires 2<=arity and <=128 leaves")
        for name in ("learning_rate", "optimizer_lr", "leaf_l2", "initial_temperature", "final_temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.gate_l2) or self.gate_l2 < 0:
            raise ValueError("gate_l2 must be finite and nonnegative")
        if self.curvature not in ("newton", "first_order"):
            raise ValueError("curvature must be newton or first_order")


def loss_derivatives(score: Tensor, target: Tensor, task: str) -> tuple[Tensor, Tensor]:
    """Per-example score gradient and FULL output Hessian, not diagonal fiction."""
    outputs = score.shape[1]
    if task == "binary":
        p = score.sigmoid()
        return p - target.reshape(-1, 1), (p * (1 - p))[:, :, None]
    if task == "multiclass":
        p = score.softmax(1)
        g = p - torch.nn.functional.one_hot(target.long(), outputs)
        return g, torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
    if task == "regression":
        g = 2 * (score - target.reshape_as(score)) / outputs
        h = (2 / outputs) * torch.eye(outputs, device=score.device, dtype=score.dtype)
        return g, h[None].expand(len(score), -1, -1)
    raise ValueError("unknown task")


def leaf_routes(tree: RaggedTree, x: Tensor) -> tuple[Tensor, list]:
    _, trace = tree(x, trace=True)
    leaves = [n for n in tree.nodes.values() if n.is_leaf]
    return torch.stack([trace[n.node_id].reach for n in leaves], 1), leaves


@torch.no_grad()
def solve_soft_leaves(tree: RaggedTree, x: Tensor, g: Tensor, h: Tensor,
                      weights: Tensor, l2: float) -> float:
    """Solve coupled leaf AND class Newton system in float64.

    Multiclass off-diagonal Hessian terms are retained. Ridge regularization
    fixes the softmax translation degeneracy. Returns the residual infinity norm.
    This reference intentionally materializes a dense (leaves*outputs)^2 matrix.
    """
    routes, leaves = leaf_routes(tree, x)
    r, gd, hd, w = routes.double(), g.double(), h.double(), weights.double()
    w = w / w.sum()
    outputs = g.shape[1]
    size = len(leaves) * outputs
    if size > 2048:
        raise ValueError("conditional Newton reference is limited to 2048 leaf-output unknowns")
    matrix = torch.einsum("ia,ib,icd,i->acbd", r, r, hd, w).reshape(size, size)
    matrix += l2 * torch.eye(size, dtype=torch.float64, device=matrix.device)
    rhs = -torch.einsum("ia,ic,i->ac", r, gd, w).reshape(size)
    values = torch.linalg.solve(matrix, rhs)
    if not torch.isfinite(values).all():
        raise FloatingPointError("nonfinite coupled leaf solution")
    for leaf, value in zip(leaves, values.reshape(len(leaves), outputs)):
        leaf.value.copy_(value.to(leaf.value))
    return float((matrix @ values - rhs).abs().max())


class _AdditiveStages(nn.Module):
    def __init__(self, intercept: Tensor):
        super().__init__()
        self.register_buffer("intercept", intercept.clone())
        self.register_buffer("rates", torch.zeros(0))
        self.trees = nn.ModuleList()

    def forward(self, x: Tensor, *, hard=False) -> Tensor:
        result = self.intercept.expand(len(x), -1).clone()
        for rate, tree in zip(self.rates, self.trees):
            result = result + rate * tree(x, hard=hard)[0]
        return result


class _StagewiseEstimator(BaseEstimator):
    classification = False

    def __init__(self, config: StagewiseConfig | None = None):
        self.config = config

    def fit(self, X, y, sample_weight=None, *, eval_set=None):
        cfg = self.config or StagewiseConfig()
        cfg.__post_init__()
        self.config_ = cfg
        self.preprocessor_ = Preprocessor()
        weights = sample_weights(sample_weight, len(X))
        self.preprocessor_.fit(X, y, classification=self.classification, weights=weights)
        self.n_features_in_, self.n_outputs_ = np.asarray(X).shape[1], self.preprocessor_.output_dim
        if self.classification:
            self.classes_ = self.preprocessor_.classes
        self.objective_ = Objective(self.preprocessor_.task, self.n_outputs_)
        data = self.preprocessor_.split(X, y, weights)
        selection = data if eval_set is None else self.preprocessor_.split(*eval_set)
        if self.classification:
            counts = torch.bincount(data.y, weights=data.weight, minlength=len(self.classes_))
            p = (counts / counts.sum()).clamp_min(1e-7)
            intercept = (p[1]/p[0]).log().reshape(1) if len(p) == 2 else p.log()
        else:
            intercept = (data.y * data.weight[:, None]).sum(0) / data.weight.sum()
        self.model_ = _AdditiveStages(intercept)
        generator = torch.Generator().manual_seed(cfg.random_state)
        architecture = ForestConfig(n_trees=1, structure=StructureConfig(
            dynamic=False, max_depth=cfg.depth, initial_depth=cfg.depth, arity=cfg.arity,
            max_nodes=sum(cfg.arity**d for d in range(cfg.depth+1)),
            structural_gate=False, complexity=0., gate_bimodality=0., allocation_regularization=0.))
        self.history_ = []
        best_count = 0
        with torch.no_grad():
            best_score = float(self.objective_.weighted_loss(self.model_(selection.x), selection.y, selection.weight))
        self.initial_loss_ = float(self.objective_.weighted_loss(self.model_(data.x), data.y, data.weight))
        for stage in range(cfg.n_estimators):
            with torch.no_grad():
                score = self.model_(data.x)
                g, h = loss_derivatives(score, data.y, self.objective_.task)
                if cfg.curvature == "first_order":
                    h = torch.eye(self.n_outputs_)[None].expand(len(data.x), -1, -1)
            tree = RaggedTree(stage, self.n_features_in_, self.n_outputs_, architecture, generator)
            router_parameters = []
            for node in tree.nodes.values():
                node.allocation_logit.requires_grad_(False)
                if not node.is_leaf:
                    node.value.requires_grad_(False)  # internal bypass is exactly zero
                    router_parameters.extend((node.routing_weight, node.routing_bias))
                node.set_temperature(cfg.initial_temperature)
            tree.depth_logits.requires_grad_(False)
            residual = solve_soft_leaves(tree, data.x, g, h, data.weight, cfg.leaf_l2)
            optimizer = torch.optim.Adam([p for p in tree.parameters() if p.requires_grad], lr=cfg.optimizer_lr)
            for step in range(cfg.inner_steps):
                temperature = cfg.final_temperature + .5 * (cfg.initial_temperature-cfg.final_temperature) * (1+math.cos(math.pi*step/max(1,cfg.inner_steps-1)))
                for node in tree.nodes.values():
                    node.set_temperature(temperature)
                optimizer.zero_grad(set_to_none=True)
                prediction = tree(data.x)[0]
                linear = (g * prediction).sum(1)
                quadratic = .5 * torch.einsum("ic,icd,id->i", prediction, h, prediction)
                loss = ((linear+quadratic)*data.weight).sum()/data.weight.sum()
                loss += .5 * cfg.leaf_l2 * sum(n.value.square().sum() for n in tree.nodes.values() if n.is_leaf)
                loss += .5 * cfg.gate_l2 * sum(p.square().mean() for p in router_parameters)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite stagewise surrogate")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tree.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
            residual = max(residual, solve_soft_leaves(tree, data.x, g, h, data.weight, cfg.leaf_l2))
            tree.eval()
            with torch.no_grad():
                direction = tree(data.x)[0]
                before = float(self.objective_.weighted_loss(score, data.y, data.weight))
                rate = cfg.learning_rate
                for _ in range(20):
                    after = float(self.objective_.weighted_loss(score+rate*direction, data.y, data.weight))
                    if math.isfinite(after) and after < before - 1e-10:
                        break
                    rate *= .5
                else:
                    self.history_.append({"stage":stage,"accepted":False,"train_loss":before,"solve_residual":residual})
                    break
                tree.requires_grad_(False)
                self.model_.trees.append(tree)
                self.model_.rates = torch.cat((self.model_.rates, torch.tensor([rate])))
                validation = float(self.objective_.weighted_loss(self.model_(selection.x),selection.y,selection.weight))
                self.history_.append({"stage":stage,"accepted":True,"train_loss":after,
                                      "selection_loss":validation,"step_size":rate,"solve_residual":residual})
                if validation < best_score:
                    best_score, best_count = validation, len(self.model_.trees)
        self.model_.trees = nn.ModuleList(list(self.model_.trees)[:best_count])
        self.model_.rates = self.model_.rates[:best_count].clone()
        self.model_.eval()
        self.n_estimators_, self.best_selection_loss_ = best_count, best_score
        return self

    @torch.no_grad()
    def decision_function(self, X, *, hard=False):
        check_is_fitted(self, "model_")
        return self.model_(self.preprocessor_.transform_x(X), hard=hard).numpy()

    def predict(self, X, *, hard=False):
        score = self.decision_function(X, hard=hard)
        if self.classification:
            return self.classes_[self.predict_proba(X, hard=hard).argmax(1)]
        value = self.preprocessor_.inverse_target(score)
        return value[:, 0] if value.shape[1] == 1 else value


class AdaptiveStagewiseClassifier(ClassifierMixin, _StagewiseEstimator):
    classification = True

    def predict_proba(self, X, *, hard=False):
        return self.objective_.response(torch.from_numpy(self.decision_function(X, hard=hard))).numpy()


class AdaptiveStagewiseRegressor(RegressorMixin, _StagewiseEstimator):
    pass
