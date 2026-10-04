"""Forests assembled from independently tuned input/tree specialists.

This bridge is not stagewise boosting. Members can have different feature maps,
depths and temperatures. It preserves the native adaptive forest as a separate
API rather than pretending these heterogeneous input maps have one shared gate
matrix. Optional joint refinement is explicitly named and independently tested.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import log_loss, mean_squared_error
from .autotune import MappedTree


class SpecialistForest:
    def __init__(self, members, *, weights=None):
        if not members or not all(isinstance(m, MappedTree) and
                                  hasattr(m, 'tree_') and hasattr(m, 'encoder_')
                                  for m in members):
            raise ValueError('members must be fitted MappedTree instances')
        self.members = list(members)
        self.classification = members[0].classification
        if any(m.classification != self.classification for m in members):
            raise ValueError('member tasks differ')
        self.n_trees_ = len(members)
        if self.classification:
            self.classes_ = members[0].classes_.copy()
            if any(not np.array_equal(m.classes_, self.classes_) for m in members):
                raise ValueError('class orders differ')
        self.weights_ = np.full(self.n_trees_, 1. / self.n_trees_) if weights is None else np.asarray(weights, float)
        self._check_weights()
        self.blend_fit_ = None

    def _check_weights(self):
        if (self.weights_.shape != (self.n_trees_,) or not np.isfinite(self.weights_).all()
                or self.weights_.min() < 0 or not np.isclose(self.weights_.sum(), 1.)):
            raise ValueError('weights must lie on the probability simplex')

    def member_predictions(self, X):
        prediction = np.stack([m.predict_proba(X) if self.classification else m.predict(X)
                               for m in self.members], axis=0).astype(np.float64)
        if self.classification:
            prediction /= prediction.sum(axis=-1, keepdims=True)
        return prediction

    def predict_proba(self, X):
        if not self.classification:
            raise AttributeError('regression forest has no class probabilities')
        self._check_weights()
        p = np.tensordot(self.weights_, self.member_predictions(X), axes=(0, 0))
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        if self.classification:
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        self._check_weights()
        return np.tensordot(self.weights_, self.member_predictions(X), axes=(0, 0))

    def fit_blend(self, X_blend, y_blend, *, regularization=.02):
        """Fit constant simplex weights on a separate calibration/blend partition.

        Ridge shrinks toward uniform weights. No coefficient search is performed
        on these same observations. The caller owns partition provenance.
        Classification is a convex probability mixture, NOT a mixture of logits.
        Regression objective is normalized by blend-target variance only to define
        the penalty's scale; deployment predictions remain in original units.
        """
        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError('regularization must be finite and nonnegative')
        prediction = self.member_predictions(X_blend)
        y = np.asarray(y_blend)
        uniform = np.full(self.n_trees_, 1. / self.n_trees_)
        if self.classification:
            mapping = {v: i for i, v in enumerate(self.classes_)}
            try:
                label = np.array([mapping[v] for v in y])
            except KeyError as error:
                raise ValueError('unknown blend class') from error
            chosen = prediction[:, np.arange(len(y)), label]
            def primary(w):
                return -np.log(np.maximum(w @ chosen, 1e-15)).mean()
        else:
            variance = max(float(np.var(y)), 1e-12)
            def primary(w):
                residual = np.tensordot(w, prediction, axes=(0, 0)) - y
                return np.mean(residual ** 2) / variance
        def objective(w):
            return primary(w) + regularization * self.n_trees_ * np.sum((w - uniform) ** 2)
        result = minimize(objective, uniform, method='SLSQP', bounds=[(0., 1.)] * self.n_trees_,
            constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.}],
            options={'maxiter': 500, 'ftol': 1e-12})
        candidate = np.maximum(result.x, 0.); candidate /= candidate.sum()
        accepted = bool(result.success and np.isfinite(candidate).all() and
                        objective(candidate) <= objective(uniform) + 1e-10)
        self.weights_ = candidate if accepted else uniform
        self.blend_fit_ = dict(rows=len(y), regularization=regularization, accepted=accepted,
            solver_message=str(result.message), uniform_loss=float(primary(uniform)),
            weighted_loss=float(primary(self.weights_)), weights=self.weights_.tolist())
        return self

    def diagnostics(self, X, y):
        """Prediction-only diagnostics. Oracle numbers are NOT deployable scores."""
        pred = self.member_predictions(X)
        y = np.asarray(y)
        if self.classification:
            mapper = {v: k for k, v in enumerate(self.classes_)}
            encoded = np.array([mapper[v] for v in y])
            losses = -np.log(np.clip(pred[:, np.arange(len(y)), encoded], 1e-15, 1.))
            errors = (pred.argmax(-1) != encoded).astype(float)
            prediction = self.predict_proba(X)
            loss = log_loss(y, prediction, labels=self.classes_)
            disagreements = [float(np.mean(pred[i].argmax(1) != pred[j].argmax(1)))
                             for i in range(len(pred)) for j in range(i)]
            oracle = float(errors.min(0).mean())
        else:
            residual = pred - y
            # Squaring must precede output aggregation: opposite-sign errors on
            # separate targets must not cancel in the casewise oracle diagnostic.
            losses = residual ** 2
            if residual.ndim > 2:
                losses = losses.mean(axis=tuple(range(2, residual.ndim)))
            errors = residual.reshape(len(pred), -1)
            loss = math.sqrt(mean_squared_error(y, self.predict(X)))
            disagreements = []
            oracle = float(np.sqrt(losses.min(0).mean()))
        correlations = []
        for i in range(len(pred)):
            for j in range(i):
                if errors[i].std() > 1e-12 and errors[j].std() > 1e-12:
                    correlations.append(float(np.corrcoef(errors[i], errors[j])[0, 1]))
        return dict(trees=self.n_trees_, weights=self.weights_.tolist(), loss=float(loss),
            member_loss=[float(log_loss(y, p, labels=self.classes_)) if self.classification
                         else float(np.sqrt(mean_squared_error(y, p))) for p in pred],
            error_correlation_mean=float(np.mean(correlations)) if correlations else None,
            disagreement_mean=float(np.mean(disagreements)) if disagreements else None,
            oracle_casewise_error=oracle, oracle_is_deployable=False,
            examples=len(y))

    def jointly_refine(self, X_train, y_train, *, eval_set, epochs=128,
                       learning_rate=.001, weight_decay=1e-4, batch_size=512,
                       random_state=0, evaluate_every=8):
        """Jointly update member gates/readouts, retaining the best stopping score.

        Probabilities (classification) or inverse-standardized predictions
        (regression) are mixed. Feature maps and mixture weights are fixed. The
        method returns an independent forest and never mutates its source members.
        The initial ensemble is an eligible checkpoint. This is NOT the original
        physical/plastic controller: those remain in the native adaptive trainer.
        """
        if epochs < 1 or learning_rate <= 0 or batch_size < 1 or evaluate_every < 1:
            raise ValueError('invalid refinement controls')
        result = deepcopy(self)
        models = torch.nn.ModuleList([m.tree_.model_ for m in result.members])
        objective = result.members[0].tree_.objective_
        train_x, stop_x = [], []
        for member in result.members:
            train_x.append(member.tree_.preprocessor_.transform_x(member.encoder_.transform(X_train)))
            stop_x.append(member.tree_.preprocessor_.transform_x(member.encoder_.transform(eval_set[0])))
        weights = torch.as_tensor(result.weights_, dtype=torch.float32)
        if self.classification:
            mapping = {v: i for i, v in enumerate(self.classes_)}
            train_y = torch.tensor([mapping[v] for v in y_train], dtype=torch.long)
            stop_y = torch.tensor([mapping[v] for v in eval_set[1]], dtype=torch.long)
        else:
            train_y = torch.as_tensor(np.asarray(y_train), dtype=torch.float32).reshape(len(X_train), -1)
            stop_y = torch.as_tensor(np.asarray(eval_set[1]), dtype=torch.float32).reshape(len(eval_set[0]), -1)
        regression_scale = max(float(train_y.float().var(unbiased=False)), 1e-8) if not self.classification else 1.
        def combined(xs, idx=None):
            out = []
            for k, model in enumerate(models):
                raw = model(xs[k] if idx is None else xs[k][idx])
                if self.classification:
                    out.append(objective.response(raw))
                else:
                    pre = result.members[k].tree_.preprocessor_
                    # These fitted target coordinates may differ across independently fitted members.
                    out.append(raw * torch.as_tensor(pre.target_scale, dtype=raw.dtype) +
                               torch.as_tensor(pre.target_mean, dtype=raw.dtype))
            return torch.einsum('m,mno->no', weights, torch.stack(out))
        def loss(pred, target):
            if self.classification:
                return -pred[torch.arange(len(target)), target].clamp_min(1e-12).log().mean()
            return ((pred - target) ** 2).mean() / regression_scale
        opt = torch.optim.AdamW(models.parameters(), lr=learning_rate, weight_decay=weight_decay)
        generator = torch.Generator().manual_seed(random_state)
        history = []
        best_score, best_epoch = math.inf, -1
        best = None
        def evaluate(epoch):
            nonlocal best_score, best_epoch, best
            with torch.no_grad():
                train_score = float(loss(combined(train_x), train_y))
                score = float(loss(combined(stop_x), stop_y))
            history.append(dict(epoch=epoch, training_loss=train_score, stopping_loss=score))
            if score < best_score:
                best_score, best_epoch, best = score, epoch, deepcopy(models.state_dict())
        evaluate(0)
        for epoch in range(1, epochs + 1):
            rate = learning_rate * (.1 + .9 * .5 * (1. + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1))))
            for group in opt.param_groups:
                group['lr'] = rate
            order = torch.randperm(len(train_y), generator=generator)
            for indices in order.split(batch_size):
                opt.zero_grad(); value = loss(combined(train_x, indices), train_y[indices])
                if not torch.isfinite(value):
                    raise FloatingPointError('nonfinite joint-refinement objective')
                value.backward()
                torch.nn.utils.clip_grad_norm_(models.parameters(), 10., error_if_nonfinite=True)
                opt.step()
            if epoch % evaluate_every == 0 or epoch == epochs:
                evaluate(epoch)
        result.joint_last_state_ = deepcopy(models.state_dict())
        models.load_state_dict(best)
        for member in result.members:
            member.tree_.best_state_ = deepcopy(member.tree_.model_.state_dict())
        result.joint_history_ = history
        result.joint_best_epoch_ = best_epoch
        result.joint_best_score_ = best_score
        return result
