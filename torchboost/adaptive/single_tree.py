"""Fixed-topology specialization of the native adaptive forest, with exactly one tree.

This is an optimization/diagnostic path, not an implementation of dynamic growth,
plasticity or physical state transitions. Packing is checked against the native
model. Every fitted model can be materialized back into that native format.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from .config import ForestConfig, PhysicsConfig, StructureConfig
from .data import Preprocessor, sample_weights
from .forest import AdaptiveForest
from .objectives import Objective


@dataclass
class SingleTreeConfig:
    depth: int = 4
    arity: int = 2
    epochs: int = 1024
    learning_rate: float = .01
    final_learning_rate_ratio: float = .1
    lr_schedule: str = 'cosine'
    weight_decay: float = 1e-4
    logit_penalty: float = 0.
    route_balance: float = 0.
    temperature: float = 1.
    final_temperature: float | None = None
    readout: str = 'residual'  # residual: original native values at every node
    initializer: str = 'random'
    cart_strength: float = 4.
    cart_min_leaf: int = 5
    batch_size: int = 256
    refit_every: int = 0
    refit_iterations: int = 25
    gradient_clip: float = 10.
    random_state: int = 0
    evaluate_every: int = 4
    max_nodes: int = 8191
    routing_diagnostics_every: int = 0
    schedule_epochs: int | None = None

    def __post_init__(self):
        for name in ('depth', 'refit_every', 'routing_diagnostics_every'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('arity', 'epochs', 'batch_size', 'refit_iterations', 'evaluate_every', 'max_nodes', 'cart_min_leaf'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.schedule_epochs is not None and (not isinstance(self.schedule_epochs, int) or not 1 <= self.schedule_epochs <= self.epochs):
            raise ValueError('schedule_epochs must be a positive integer no larger than epochs')
        if self.arity < 2:
            raise ValueError('arity must be at least two')
        if (self.arity ** (self.depth + 1) - 1) // (self.arity - 1) > self.max_nodes:
            raise ValueError('requested complete tree exceeds max_nodes')
        for name in ('learning_rate', 'temperature', 'gradient_clip', 'cart_strength'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('weight_decay', 'logit_penalty', 'route_balance'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 < self.final_learning_rate_ratio <= 1:
            raise ValueError('the learning-rate tail must be positive and <= the initial rate')
        if self.final_temperature is not None and (not math.isfinite(self.final_temperature) or self.final_temperature <= 0):
            raise ValueError('final_temperature must be finite and positive')
        if self.lr_schedule not in ('constant', 'cosine'):
            raise ValueError('lr_schedule must be constant or cosine')
        if self.readout not in ('residual', 'leaf'):
            raise ValueError('readout must be residual or leaf')
        if self.initializer not in ('random', 'cart', 'balanced'):
            raise ValueError('initializer must be random, cart, or balanced')
        if self.initializer == 'balanced' and self.arity != 2:
            raise ValueError('balanced flow initialization currently supports binary trees')
        if self.initializer == 'cart' and (self.arity != 2 or self.readout != 'leaf'):
            raise ValueError('CART initialization requires a binary leaf readout')

    def forest_config(self):
        return ForestConfig(n_trees=1, aggregation='mean', residual_weights=False,
            execution='forest_packed', epochs=self.epochs, learning_rate=self.learning_rate,
            batch_size=self.batch_size, weight_decay=self.weight_decay,
            collect_metrics=False, compact_history=True, random_state=self.random_state,
            structure=StructureConfig(initial_depth=0, max_depth=self.depth, arity=self.arity,
                max_nodes=self.max_nodes, max_parameters=20_000_000, structural_gate=False,
                complexity=0., gate_bimodality=0., allocation_regularization=0.),
            physics=PhysicsConfig(initial_temperature=self.temperature,
                ambient_temperature=min(.1, self.temperature), max_temperature=max(10., self.temperature)))


class PackedSingleTree(nn.Module):
    """Pack a complete, active native tree; remove only redundant ensemble controls.

    Residual readout retains every node value. With gates identically one, a leaf
    readout is functionally equivalent: each leaf stores the sum of its ancestor
    residual values. Packing does not change routing or convert oblique to axis splits.
    """
    def __init__(self, native: AdaptiveForest, *, readout: str = 'residual'):
        super().__init__()
        if len(native.trees) != 1:
            raise ValueError('packing requires exactly one tree, not one boosting round')
        if readout not in ('residual', 'leaf'):
            raise ValueError('invalid readout')
        if native.config.feature_dropout or native.config.structure.dynamic or native.config.physics.mode != 'none' or native.config.plasticity.mode != 'none' or native.config.online.enabled:
            raise ValueError('fixed packing requires dropout-free, nonadaptive training; keep adaptive policies in the native engine')
        tree = native.trees[0]
        self.native_config = deepcopy(native.config)
        self.native_config.n_trees = 1
        self.native_config.aggregation = 'mean'
        self.native_config.residual_weights = False
        self.native_config.structure.structural_gate = False
        self.native_config.feature_dropout = self.native_config.tree_dropout = 0.
        self.input_dim, self.output_dim = native.input_dim, native.output_dim
        self.depth = max(n.depth for n in tree.nodes.values())
        self.arity = native.config.structure.arity
        self.readout = readout
        nodes = [tree.get(tree.root_id)]
        for node in nodes:
            nodes.extend(tree.get(key) for key in node.children_ids)
        if len(nodes) != (self.arity ** (self.depth + 1) - 1) // (self.arity - 1):
            raise ValueError('fixed packing requires a complete tree')
        if any(not n.active or n.frozen or n.locked for n in nodes):
            raise ValueError('fixed packing requires active, unlocked, trainable nodes')
        if any(n.structural is not None for n in nodes):
            raise ValueError('fixed packing requires structural gates to be disabled, not silently discarded')
        if any((n.is_leaf and n.depth != self.depth) or (not n.is_leaf and len(n.children_ids) != self.arity) for n in nodes):
            raise ValueError('packing requires uniform arity and a complete fixed depth')
        if not torch.all(tree.feature_mask == 1):
            raise ValueError('feature-mask constraints must remain in the native engine')
        self.node_ids = [n.node_id for n in nodes]
        self.n_nodes = len(nodes)
        self.n_leaves = self.arity ** self.depth
        self.n_internal = self.n_nodes - self.n_leaves
        internal = nodes[:self.n_internal]
        # Include the scalar residual multiplier in the linear readout, exactly.
        multiplier = (native.residual_logits.detach().sigmoid()[0, 0]
                      if native.config.residual_weights else 1.) * native.config.shrinkage
        self.native_config.shrinkage = 1.
        weights = torch.stack([n.routing_weight.detach() for n in internal]) if internal else native.bias.new_empty((0, self.arity, self.input_dim))
        biases = torch.stack([n.routing_bias.detach() for n in internal]) if internal else native.bias.new_empty((0, self.arity))
        temperatures = torch.stack([n.temperature.detach() for n in internal]) if internal else native.bias.new_empty(0)
        self.routing_weight = nn.Parameter(weights.clone())
        self.routing_bias = nn.Parameter(biases.clone())
        self.register_buffer('temperatures', temperatures.clone())
        values = torch.stack([n.value.detach() for n in nodes]) * multiplier
        if readout == 'leaf':
            accumulated = values.clone()
            for i in range(1, self.n_nodes):
                accumulated[i] += accumulated[(i - 1) // self.arity]
            values = accumulated[self.n_internal:]
        self.values = nn.Parameter(values.clone())
        self.bias = nn.Parameter(native.bias.detach().clone())

    def routing(self, x, *, hard=False):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError('feature shape mismatch')
        if self.n_internal:
            logits = (x @ self.routing_weight.flatten(0, 1).T).reshape(len(x), self.n_internal, self.arity)
            logits = (logits + self.routing_bias) / self.temperatures[None, :, None]
            p = logits.softmax(-1)
            if hard:
                p = torch.nn.functional.one_hot(p.argmax(-1), self.arity).to(x.dtype)
        else:
            p = x.new_empty((len(x), 0, self.arity))
        mass = x.new_ones((len(x), 1))
        levels = [mass]
        start = 0
        for level in range(self.depth):
            width = self.arity ** level
            mass = (mass[..., None] * p[:, start:start + width]).flatten(1)
            levels.append(mass)
            start += width
        return torch.cat(levels, dim=1), p

    def basis(self, x, *, hard=False):
        mass, _ = self.routing(x, hard=hard)
        return mass if self.readout == 'residual' else mass[:, self.n_internal:]

    def forward(self, x, *, hard=False):
        return self.basis(x, hard=hard) @ self.values + self.bias

    @torch.no_grad()
    def to_native(self):
        """Materialize ordinary native nodes. No packed-only inference is necessary."""
        cfg = deepcopy(self.native_config)
        cfg.structure.dynamic = False
        cfg.structure.max_depth = self.depth
        cfg.structure.initial_depth = 0
        cfg.structure.max_nodes = max(cfg.structure.max_nodes, self.n_nodes)
        native = AdaptiveForest(self.input_dim, self.output_dim, cfg)
        native.to(device=self.bias.device, dtype=self.bias.dtype)
        tree = native.trees[0]
        nodes = [tree.get(tree.root_id)]
        for n in nodes:
            nodes.extend(tree.get(c) for c in n.children_ids)
        native.bias.copy_(self.bias)
        for i, n in enumerate(nodes):
            n.value.zero_()
            if self.readout == 'residual':
                n.value.copy_(self.values[i])
            elif i >= self.n_internal:
                n.value.copy_(self.values[i - self.n_internal])
            if i < self.n_internal:
                n.routing_weight.copy_(self.routing_weight[i])
                n.routing_bias.copy_(self.routing_bias[i])
                n.temperature.copy_(self.temperatures[i])
        return native.eval()


def _center_logits(logits, task):
    return logits - logits.mean(1, keepdim=True) if task == 'multiclass' else logits


def _fit_objective(model, x, y, w, objective, config, *, include_balance=True):
    use_balance = include_balance and config.route_balance and model.n_internal
    if use_balance:
        mass, prob = model.routing(x)
        basis = mass if model.readout == 'residual' else mass[:, model.n_internal:]
        logits = basis @ model.values + model.bias
    else:
        logits = model(x)
    primary = objective.weighted_loss(logits, y, w)
    penalty = config.logit_penalty * (_center_logits(logits, objective.task).square().mean(1) * w).sum() / w.sum()
    if use_balance:
        reach = mass[:, :model.n_internal] * w[:, None]
        usage = (reach[..., None] * prob).sum(0) / reach.sum(0).clamp_min(1e-12)[:, None]
        # Uniform-use KL; weighted by node occupancy. Does NOT force each example's route to be uncertain.
        kl = -usage.clamp_min(1e-8).log().mean(1) - math.log(model.arity)
        penalty = penalty + config.route_balance * (kl * reach.sum(0) / w.sum()).sum() / max(model.depth, 1)
    return primary + penalty


@torch.no_grad()
def cart_initialize(model, train, objective, config):
    """Disclosed supervised axis-aligned warm start, subsequently trained obliquely."""
    if model.readout != 'leaf' or model.arity != 2:
        raise ValueError('CART initialization requires binary leaf readout')
    x, y, w = train.x.numpy(), train.y.numpy(), train.weight.numpy()
    if model.depth == 0:
        return
    cls = DecisionTreeRegressor if objective.task == 'regression' else DecisionTreeClassifier
    cart = cls(max_depth=model.depth, min_samples_leaf=config.cart_min_leaf, random_state=config.random_state)
    cart.fit(x, y, sample_weight=w)
    model.bias.zero_()
    t = cart.tree_
    def fill(i, k):
        if i >= model.n_internal:
            value = t.value[k].reshape(-1)
            if objective.task != 'regression':
                # Tree values in supported sklearn are proportions; reconstruct weighted counts.
                counts = value / max(value.sum(), 1e-12) * t.weighted_n_node_samples[k]
                p = (counts + .5) / (counts.sum() + .5 * len(counts))
                value = np.array([math.log(p[1] / p[0])]) if objective.task == 'binary' else np.log(p) - np.log(p).mean()
            model.values[i - model.n_internal].copy_(torch.as_tensor(value, dtype=model.values.dtype))
            return
        feature = t.feature[k]
        if feature >= 0:
            model.routing_weight[i].zero_()
            model.routing_weight[i, 0, feature] = -config.cart_strength / 2
            model.routing_weight[i, 1, feature] = config.cart_strength / 2
            model.routing_bias[i, 0] = config.cart_strength * t.threshold[k] / 2
            model.routing_bias[i, 1] = -config.cart_strength * t.threshold[k] / 2
            fill(2 * i + 1, t.children_left[k]); fill(2 * i + 2, t.children_right[k])
        else:
            fill(2 * i + 1, k); fill(2 * i + 2, k)
    fill(0, 0)


@torch.no_grad()
def balanced_initialize(model, train):
    """Label-free conditional flow balancing at initialization only.

    Keep each oblique direction; fit its scalar offset so training-weighted soft
    branch occupancy is 1/2 conditional on reaching its parent. This does not
    force balanced traffic after training or claim that balanced splits are best.
    A common logit shift is unidentifiable, so use opposite half-offsets.
    """
    if model.arity != 2:
        raise ValueError('balanced initialization currently supports binary routing')
    x=train.x.to(dtype=torch.float64)
    weight=train.weight.to(dtype=torch.float64)
    reach=[torch.ones(len(x),dtype=x.dtype)]
    for i in range(model.n_internal):
        difference=(x @ (model.routing_weight[i,0]-model.routing_weight[i,1]).double()) / model.temperatures[i].double()
        conditional=weight*reach[i]
        conditional=conditional/conditional.sum().clamp_min(1e-30)
        lo=-60.-float(difference.max());hi=60.-float(difference.min())
        for _ in range(52):
            mid=(lo+hi)/2
            occupancy=float((conditional*torch.sigmoid(difference+mid)).sum())
            if occupancy>.5:hi=mid
            else:lo=mid
        shift=(lo+hi)/2
        model.routing_bias[i,0]=float(model.temperatures[i])*shift/2
        model.routing_bias[i,1]=-float(model.temperatures[i])*shift/2
        probability=torch.sigmoid(difference+shift)
        reach.append(reach[i]*probability);reach.append(reach[i]*(1-probability))


def refit_readout(model, train, objective, config):
    """Fit the convex conditional readout on TRAINING data; line-search and rollback.

    No claim of reaching the exact global minimizer at a finite iteration budget.
    Routing is held fixed. The ridge term is on logits, not a depth-dependent
    arbitrary redundant parameter basis. Caller resets obsolete Adam readout state.
    """
    design = model.basis(train.x).detach().to(torch.float64)
    value = model.values.detach().to(torch.float64).clone().requires_grad_(True)
    bias = model.bias.detach().to(torch.float64).clone().requires_grad_(True)
    w = train.weight.to(torch.float64)
    def loss():
        logits = design @ value + bias
        return objective.weighted_loss(logits, train.y, w) + config.logit_penalty * (_center_logits(logits, objective.task).square().mean(1) * w).sum() / w.sum()
    before = float(loss().detach())
    opt = torch.optim.LBFGS([value, bias], lr=1., max_iter=config.refit_iterations,
        history_size=10, tolerance_grad=1e-8, tolerance_change=1e-10, line_search_fn='strong_wolfe')
    calls = 0
    def closure():
        nonlocal calls
        calls += 1
        opt.zero_grad(); result = loss(); result.backward(); return result
    opt.step(closure)
    after = float(loss().detach())
    accepted = math.isfinite(after) and after <= before + 1e-10
    original_values, original_bias = model.values.detach().clone(), model.bias.detach().clone()
    with torch.no_grad():
        actual_before = float(_fit_objective(model, train.x, train.y, train.weight, objective, config, include_balance=False))
        if accepted:
            model.values.copy_(value.to(model.values)); model.bias.copy_(bias.to(model.bias))
        actual_after = float(_fit_objective(model, train.x, train.y, train.weight, objective, config, include_balance=False))
        if not math.isfinite(actual_after) or actual_after > actual_before + 1e-7 * max(1., actual_before):
            model.values.copy_(original_values); model.bias.copy_(original_bias); accepted = False
    return {'before': actual_before, 'after': actual_after, 'accepted': accepted, 'closure_calls': calls,
            'float64_before': before, 'float64_after': after}


@torch.no_grad()
def routing_telemetry(model, train):
    """Training-only, nonmutating flow statistics; entropy counts are not ranks."""
    mass, probability = model.routing(train.x)
    weight = train.weight / train.weight.sum()
    occupancy = (mass * weight[:, None]).sum(0)
    leaf_usage = occupancy[model.n_internal:]
    hard_leaf = model.routing(train.x, hard=True)[0][:, model.n_internal:].argmax(1)
    levels = []
    start = 0
    for depth in range(model.depth):
        width = model.arity ** depth
        p = probability[:, start:start+width]
        reach = mass[:, start:start+width] * weight[:, None]
        entropy = -(p.clamp_min(1e-30).log()*p).sum(-1)
        saturated = p.max(-1).values > .99
        g = model.routing_weight.grad
        levels.append(dict(depth=depth,
            conditional_entropy=float((reach*entropy).sum()/reach.sum().clamp_min(1e-30)),
            saturated_mass=float((reach*saturated).sum()/reach.sum().clamp_min(1e-30)),
            last_batch_routing_gradient_norm=float(g[start:start+width].norm()) if g is not None else 0.))
        start += width
    return dict(effective_leaves=float(torch.exp(-(leaf_usage*leaf_usage.clamp_min(1e-30).log()).sum())),
        hard_visited_leaves=int(hard_leaf[train.weight>0].unique().numel()),
        soft_leaves_below_one_example=int((leaf_usage*len(train.x)<1.).sum()),
        levels=levels)


def model_state_digest(model):
    """Hash numerical model state at a schedule boundary for exact replay checks."""
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class _SingleTreeEstimator(BaseEstimator):
    classification = False
    def __init__(self, config=None):
        self.config = config

    def fit(self, X, y, sample_weight=None, *, eval_set=None):
        self.config_ = deepcopy(self.config or SingleTreeConfig())
        c = self.config_
        w = sample_weights(sample_weight, len(X))
        self.preprocessor_ = Preprocessor()
        self.preprocessor_.fit(X, y, classification=self.classification, weights=w)
        train = self.preprocessor_.split(X, y, w)
        if eval_set is None:
            valid = train
        elif len(eval_set) in (2, 3):
            valid = self.preprocessor_.split(*eval_set)
        else:
            raise ValueError('eval_set must be (X,y) or (X,y,weight)')
        self.n_features_in_ = train.x.shape[1]
        self.objective_ = Objective(self.preprocessor_.task, self.preprocessor_.output_dim)
        if self.classification:
            self.classes_ = self.preprocessor_.classes.copy()
        self.generator_ = torch.Generator().manual_seed(c.random_state)
        native = AdaptiveForest(self.n_features_in_, self.objective_.output_dim, c.forest_config(), generator=self.generator_)
        if self.classification:
            counts = torch.bincount(train.y, weights=train.weight, minlength=len(self.classes_))
            p = (counts / counts.sum()).clamp_min(1e-7)
            with torch.no_grad():
                native.bias.copy_(torch.log(p[1] / p[0]).reshape(1) if self.objective_.task == 'binary' else p.log())
        model = PackedSingleTree(native, readout=c.readout)
        if c.initializer == 'cart':
            cart_initialize(model, train, self.objective_, c)
        elif c.initializer == 'balanced':
            balanced_initialize(model, train)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
        self.history_, self.refits_ = [], []
        self.best_score_, self.best_epoch_ = math.inf, -1
        self.optimizer_steps_ = self.examples_seen_ = 0
        def evaluate(epoch, rate, grad):
            with torch.no_grad():
                tr, va = model(train.x), model(valid.x)
                tl = float(self.objective_.weighted_loss(tr, train.y, train.weight))
                vl = float(self.objective_.weighted_loss(va, valid.y, valid.weight))
                if not math.isfinite(tl + vl):
                    raise FloatingPointError('nonfinite evaluation loss')
                trp, vap = self.objective_.response(tr), self.objective_.response(va)
                terr = float((trp.argmax(1) != train.y).float().mean()) if self.classification else None
                verr = float((vap.argmax(1) != valid.y).float().mean()) if self.classification else None
                row = dict(epoch=epoch, train_loss=tl, validation_loss=vl, train_error=terr,
                    validation_error=verr, learning_rate=rate, gradient_norm=grad,
                    temperature=float(model.temperatures.mean()) if model.n_internal else c.temperature,
                    optimizer_steps=self.optimizer_steps_, examples_seen=self.examples_seen_)
                if c.routing_diagnostics_every and (epoch == 0 or epoch == c.epochs or epoch % c.routing_diagnostics_every == 0):
                    row['routing'] = routing_telemetry(model, train)
                if c.schedule_epochs is not None and epoch == c.schedule_epochs:
                    row['schedule_boundary_state_sha256'] = model_state_digest(model)
                self.history_.append(row)
                if vl < self.best_score_:
                    self.best_score_, self.best_epoch_ = vl, epoch
                    self.best_state_ = deepcopy(model.state_dict())
        evaluate(0, c.learning_rate, 0.)
        started = time.perf_counter()
        for epoch in range(1, c.epochs + 1):
            progress = min(1., (epoch - 1) / max((c.schedule_epochs or c.epochs) - 1, 1))
            f = 1. if c.lr_schedule == 'constant' else c.final_learning_rate_ratio + (1 - c.final_learning_rate_ratio) * .5 * (1 + math.cos(math.pi * progress))
            rate = c.learning_rate * f
            for group in optimizer.param_groups:
                group['lr'] = rate
            if c.final_temperature is not None:
                model.temperatures.fill_(c.temperature * (c.final_temperature / c.temperature) ** progress)
            order = torch.randperm(len(train.x), generator=self.generator_)
            gradient = 0.
            for start in range(0, len(order), c.batch_size):
                idx = order[start:start + c.batch_size]
                if float(train.weight[idx].sum()) == 0:
                    continue
                optimizer.zero_grad()
                loss = _fit_objective(model, train.x[idx], train.y[idx], train.weight[idx], self.objective_, c)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                loss.backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), c.gradient_clip, error_if_nonfinite=True))
                optimizer.step()
                self.optimizer_steps_ += 1; self.examples_seen_ += len(idx)
            if c.refit_every and epoch % c.refit_every == 0:
                record = refit_readout(model, train, self.objective_, c)
                self.refits_.append({'epoch': epoch, **record})
                if record['accepted']:
                    optimizer.state.pop(model.values, None); optimizer.state.pop(model.bias, None)
            if epoch == 1 or epoch % c.evaluate_every == 0 or epoch == c.epochs or epoch == c.schedule_epochs:
                evaluate(epoch, rate, gradient)
        self.fit_seconds_ = time.perf_counter() - started
        self.last_state_ = deepcopy(model.state_dict())
        self.optimizer_state_ = deepcopy(optimizer.state_dict())
        self.model_ = model
        self.model_.load_state_dict(self.best_state_)
        self.model_.eval()
        self.data_fingerprints_ = {'train': train.fingerprint(), 'validation': valid.fingerprint()}
        return self

    @torch.no_grad()
    def _raw_predict(self, X, *, hard=False):
        check_is_fitted(self, 'model_')
        x = self.preprocessor_.transform_x(X)
        return torch.cat([self.model_(chunk, hard=hard) for chunk in x.split(1024)]).numpy()

    def predict(self, X, *, hard=False):
        logits = self._raw_predict(X, hard=hard)
        if self.classification:
            prob = self.objective_.response(torch.from_numpy(logits)).numpy()
            return self.classes_[prob.argmax(1)]
        result = self.preprocessor_.inverse_target(logits)
        return result[:, 0] if result.shape[1] == 1 else result

    def decision_function(self, X, *, hard=False):
        logits = self._raw_predict(X, hard=hard)
        return logits[:, 0] if logits.shape[1] == 1 else logits

    def native_model(self):
        check_is_fitted(self, 'model_')
        return self.model_.to_native()

    def save(self, path):
        check_is_fitted(self, 'model_')
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(format='torchboost.single_tree', version=1, config=asdict(self.config_),
            classification=self.classification, preprocessor=self.preprocessor_.state_dict(),
            model=self.best_state_, last=self.last_state_, history=self.history_, refits=self.refits_,
            best_epoch=self.best_epoch_, best_score=self.best_score_, fingerprints=self.data_fingerprints_,
            optimizer_steps=self.optimizer_steps_, examples_seen=self.examples_seen_)
        tmp = path.with_suffix(path.suffix + '.tmp')
        torch.save(state, tmp); tmp.replace(path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, weights_only=True, map_location='cpu')
        if state.get('format') != 'torchboost.single_tree' or state.get('version') != 1:
            raise ValueError('unsupported single-tree checkpoint')
        if state['classification'] != cls.classification:
            raise ValueError('checkpoint task does not match estimator')
        instance = cls(SingleTreeConfig(**state['config']))
        instance.config_ = deepcopy(instance.config)
        instance.preprocessor_ = Preprocessor(); instance.preprocessor_.load_state_dict(state['preprocessor'])
        instance.n_features_in_ = len(instance.preprocessor_.mean)
        instance.objective_ = Objective(instance.preprocessor_.task, instance.preprocessor_.output_dim)
        if instance.classification:
            instance.classes_ = instance.preprocessor_.classes.copy()
        native = AdaptiveForest(instance.n_features_in_, instance.objective_.output_dim, instance.config_.forest_config())
        instance.model_ = PackedSingleTree(native, readout=instance.config_.readout)
        instance.model_.load_state_dict(state['model']); instance.model_.eval()
        for attr, key in [('best_state_', 'model'), ('last_state_', 'last'), ('history_', 'history'),
                          ('refits_', 'refits'), ('best_epoch_', 'best_epoch'), ('best_score_', 'best_score'),
                          ('data_fingerprints_', 'fingerprints'), ('optimizer_steps_', 'optimizer_steps'),
                          ('examples_seen_', 'examples_seen')]:
            setattr(instance, attr, state[key])
        return instance


class SingleTreeClassifier(ClassifierMixin, _SingleTreeEstimator):
    classification = True
    def predict_proba(self, X, *, hard=False):
        raw = self._raw_predict(X, hard=hard)
        return self.objective_.response(torch.from_numpy(raw)).numpy()


class SingleTreeRegressor(RegressorMixin, _SingleTreeEstimator):
    classification = False
