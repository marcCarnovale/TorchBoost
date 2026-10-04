"""Sklearn-style task wrappers, epoch-boundary checkpoints, and adaptation."""
from __future__ import annotations
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .config import ForestConfig
from .data import DataSplit, Preprocessor, check_x, sample_weights
from .forest import AdaptiveForest
from .objectives import Objective
from .serialization import portable_state
from .training import JointTrainer, model_snapshot, restore_model


class _AdaptiveEstimator(BaseEstimator):
    classification = False

    def __init__(self, config: ForestConfig | None = None, *, class_weight=None):
        self.config = config
        self.class_weight = class_weight

    def _training_weights(self, y, weights):
        result = sample_weights(weights, len(y))
        if self.classification and self.class_weight is not None:
            labels, counts = np.unique(y, return_counts=True)
            if self.class_weight == "balanced":
                mapping = {label: len(y) / (len(labels) * count) for label, count in zip(labels, counts)}
            elif isinstance(self.class_weight, dict):
                mapping = self.class_weight
                if set(mapping) - set(labels.tolist()):
                    raise ValueError("class_weight includes an unknown class")
            else:
                raise ValueError("class_weight must be None, 'balanced', or a mapping")
            result = result * np.asarray([mapping.get(value, 1.) for value in np.asarray(y).tolist()])
            result = sample_weights(result, len(y))
        return result

    def _make_split(self, value, default: DataSplit) -> DataSplit:
        if value is None:
            return default
        if len(value) not in (2, 3):
            raise ValueError("a split must be (X,y) or (X,y,sample_weight)")
        return self.preprocessor_.split(value[0], value[1], value[2] if len(value) == 3 else None)

    def _validate_split_roles(self, control_set, eval_set) -> None:
        cfg = self.config_
        adaptive = (cfg.structure.dynamic or cfg.plasticity.mode != "none"
                    or cfg.physics.mode in ("capacitor", "rlc") or cfg.online.enabled)
        if adaptive and control_set is None:
            raise ValueError("adaptive policies require an explicit control_set separate from final evaluation")
        if control_set is not None and eval_set is not None and control_set[0] is eval_set[0]:
            raise ValueError("control_set and eval_set must not be the same array")

    def fit(self, X, y, sample_weight=None, *, control_set=None, eval_set=None,
            custom_loss: Callable | None = None, selection_metric: Callable | None = None,
            greater_is_better: bool = False, stop_epoch: int | None = None):
        if hasattr(self, "trainer_"):
            self.trainer_.close()
        self.config_ = deepcopy(self.config or ForestConfig())
        self._validate_split_roles(control_set, eval_set)
        if greater_is_better and selection_metric is None:
            raise ValueError("greater_is_better requires an explicit custom selection metric")
        X = check_x(X)
        weights = self._training_weights(y, sample_weight)
        self.preprocessor_ = Preprocessor()
        self.preprocessor_.fit(X, y, classification=self.classification, weights=weights)
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = self.preprocessor_.output_dim
        if self.classification:
            self.classes_ = self.preprocessor_.classes.copy()
        self._custom_loss, self._selection_metric = custom_loss, selection_metric
        self._requires_custom_loss, self._requires_selection_metric = custom_loss is not None, selection_metric is not None
        self.objective_ = Objective(self.preprocessor_.task, self.n_outputs_, custom_loss)
        generator = torch.Generator().manual_seed(self.config_.random_state)
        model = AdaptiveForest(self.n_features_in_, self.n_outputs_, self.config_, generator=generator)
        train = self.preprocessor_.split(X, y, weights)
        with torch.no_grad():
            if self.classification:
                counts = torch.bincount(train.y, weights=train.weight, minlength=len(self.classes_))
                probabilities = (counts / counts.sum()).clamp_min(1e-7)
                bias = torch.log(probabilities[1] / probabilities[0]).reshape(1) if self.objective_.task == "binary" else probabilities.log()
                model.bias.copy_(bias.to(model.bias))
        control, selection = self._make_split(control_set, train), self._make_split(eval_set, train)
        self.trainer_ = JointTrainer(model, self.objective_, self.config_, generator,
                                    selection_metric=selection_metric, greater_is_better=greater_is_better)
        self.trainer_.fit(train, control, selection, stop_epoch=stop_epoch)
        self._select_model()
        return self

    def _select_model(self) -> None:
        self.model_ = restore_model(self.trainer_.best_snapshot, self.n_features_in_, self.n_outputs_, self.config_)
        self.model_.eval()
        self.best_epoch_ = self.trainer_.best_epoch
        self.history_ = self.trainer_.history

    @torch.no_grad()
    def _raw_predict(self, X, *, hard: bool = False) -> np.ndarray:
        check_is_fitted(self, "model_")
        x = self.preprocessor_.transform_x(X)
        self.model_.eval()
        output = torch.cat([self.model_(x[start:start + self.config_.batch_size].to(self.config_.device), hard=hard).cpu()
                            for start in range(0, len(x), self.config_.batch_size)])
        return output.numpy()

    def decision_function(self, X, *, hard: bool = False) -> np.ndarray:
        result = self._raw_predict(X, hard=hard)
        return result[:, 0] if self.n_outputs_ == 1 else result

    def predict(self, X, *, hard: bool = False) -> np.ndarray:
        prediction = self._raw_predict(X, hard=hard)
        if self.classification:
            probabilities = self.objective_.response(torch.from_numpy(prediction)).numpy()
            return self.classes_[probabilities.argmax(1)]
        value = self.preprocessor_.inverse_target(prediction)
        return value[:, 0] if self.n_outputs_ == 1 else value

    def resume_fit(self, X, y, sample_weight=None, *, control_set=None, eval_set=None,
                   stop_epoch: int | None = None):
        check_is_fitted(self, "trainer_")
        if self._requires_custom_loss and self._custom_loss is None:
            raise ValueError("supply the original custom_loss when loading this checkpoint")
        if self._requires_selection_metric and self._selection_metric is None:
            raise ValueError("supply the original selection_metric when loading this checkpoint")
        self._validate_split_roles(control_set, eval_set)
        train = self.preprocessor_.split(X, y, self._training_weights(y, sample_weight))
        control, selection = self._make_split(control_set, train), self._make_split(eval_set, train)
        self.trainer_.fit(train, control, selection, stop_epoch=stop_epoch)
        self._select_model()
        return self

    def adapt(self, X, y, sample_weight=None, *, control_set, eval_set, epochs: int,
              reset_control_reference: bool = True):
        """New supervised phase with fixed preprocessing and retained learned state.

        This is an explicit batch adaptation API, not an unrestricted one-sample
        streaming estimator. Classes/target dimensions may not change. Selection
        is restarted; incomparable old control-set trials are cancelled.
        """
        check_is_fitted(self, "trainer_")
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("adaptation epochs must be positive")
        self._validate_split_roles(control_set, eval_set)
        trainer = self.trainer_
        trainer.scheduler.collect(wait=True)
        for trial_id in list(trainer.tracker.trials):
            trainer.plastic.finish_trial(trial_id, retain=False)
        trainer.tracker.trials.clear()
        trainer.tracker.histories.clear()
        trainer.control_indices = None
        trainer.best_score = float("inf")
        trainer.best_snapshot = model_snapshot(trainer.model)
        trainer.best_epoch = trainer.epoch - 1
        if reset_control_reference:
            trainer.physical.reference = None
        self.config_.epochs = trainer.epoch + epochs
        train = self.preprocessor_.split(X, y, self._training_weights(y, sample_weight))
        selection = self._make_split(eval_set, train)
        trainer.best_score = trainer._selection_score(selection)
        trainer.fit(train, self._make_split(control_set, train), selection,
                    verify_fingerprints=False)
        self._select_model()
        return self

    def save(self, path: str | Path, *, include_training_state: bool = True) -> None:
        """Atomic trusted checkpoint; no raw training examples are stored.

        All data fingerprints, model topology, references, optimizer state,
        controller/learner histories and local RNG state are included when
        training state is requested. Resume supplies and verifies the data.
        """
        check_is_fitted(self, "model_")
        path = Path(path)
        payload = {"format": "torchboost.adaptive", "version": 1, "config": self.config_.to_dict(),
                   "classification": self.classification, "class_weight": self.class_weight,
                   "preprocessor": self.preprocessor_.state_dict(), "model": model_snapshot(self.model_),
                   "requires_custom_loss": self._requires_custom_loss,
                   "requires_selection_metric": self._requires_selection_metric,
                   "greater_is_better": self.trainer_.greater_is_better if hasattr(self, "trainer_") else False,
                   "trainer": self.trainer_.state_dict() if include_training_state and hasattr(self, "trainer_") else None}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(portable_state(payload), temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu", custom_loss: Callable | None = None,
             selection_metric: Callable | None = None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "torchboost.adaptive" or payload.get("version") != 1:
            raise ValueError("unsupported checkpoint format")
        actual_cls = AdaptiveForestClassifier if payload["classification"] else AdaptiveForestRegressor
        if cls not in (_AdaptiveEstimator, actual_cls):
            raise ValueError("checkpoint task does not match estimator class")
        config = ForestConfig.from_dict(payload["config"])
        config.device = device
        model = actual_cls(config, class_weight=payload["class_weight"])
        model.config_ = deepcopy(config)
        model.preprocessor_ = Preprocessor()
        model.preprocessor_.load_state_dict(payload["preprocessor"])
        model.n_features_in_ = len(model.preprocessor_.mean)
        model.n_outputs_ = model.preprocessor_.output_dim
        if model.classification:
            model.classes_ = model.preprocessor_.classes.copy()
        model._custom_loss, model._selection_metric = custom_loss, selection_metric
        model._requires_custom_loss = payload["requires_custom_loss"]
        model._requires_selection_metric = payload["requires_selection_metric"]
        model.objective_ = Objective(model.preprocessor_.task, model.n_outputs_, custom_loss)
        model.model_ = restore_model(payload["model"], model.n_features_in_, model.n_outputs_, model.config_)
        model.model_.eval()
        if payload["trainer"] is not None:
            generator = torch.Generator().manual_seed(config.random_state)
            model.trainer_ = JointTrainer(model.model_, model.objective_, model.config_, generator,
                                          selection_metric=selection_metric,
                                          greater_is_better=payload["greater_is_better"])
            model.trainer_.load_state_dict(payload["trainer"])
            model.best_epoch_, model.history_ = model.trainer_.best_epoch, model.trainer_.history
        return model

    def export_json(self, path: str | Path, *, hard: bool = True) -> dict:
        check_is_fitted(self, "model_")
        from .export import export_model
        result = export_model(self.model_, self.preprocessor_, hard=hard)
        Path(path).write_text(json.dumps(result, indent=2, allow_nan=False))
        return result

    def close(self) -> None:
        if hasattr(self, "trainer_"):
            self.trainer_.close()


class AdaptiveForestClassifier(ClassifierMixin, _AdaptiveEstimator):
    classification = True

    def predict_proba(self, X, *, hard: bool = False) -> np.ndarray:
        raw = torch.from_numpy(self._raw_predict(X, hard=hard))
        return self.objective_.response(raw).numpy()


class AdaptiveForestRegressor(RegressorMixin, _AdaptiveEstimator):
    classification = False
