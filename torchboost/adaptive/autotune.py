"""Problem-specific tree tuning, with independent fit/stop/rank roles.

The rank objective is a stability heuristic, not a confidence bound. Each
successive-fidelity fit starts afresh; histories are never joined into a fake
continued trajectory. The audit/test set is deliberately absent from this API.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
import time
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
    StratifiedKFold, StratifiedGroupKFold, train_test_split)
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from .single_tree import SingleTreeClassifier, SingleTreeRegressor, SingleTreeConfig


class FeatureMap(BaseEstimator):
    """Train-only mixed-type map; optional Gaussian ranks or piecewise encoding.

    PLE uses univariate quantile knots, a clipped ramp per nonempty interval,
    plus a standardized raw coordinate for extrapolation. This is a feature
    representation change, not a claim to remain oblique in raw coordinates.
    Unknown categories map to all-zero known-category indicators. Numeric
    missing indicators are always emitted, including all-missing fit columns.
    Knot/median fitting is unweighted; supervised fitting still honors weights.
    """
    def __init__(self, mode='raw', bins=8, categorical_columns=None, random_state=0):
        self.mode = mode
        self.bins = bins
        self.categorical_columns = categorical_columns
        self.random_state = random_state

    def _frame(self, X, fitting=False):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
            if frame.columns.has_duplicates:
                raise ValueError('feature names must be unique')
            if not fitting:
                if set(frame.columns) != set(self.columns_):
                    raise ValueError('feature schema does not match fitted columns')
                frame = frame.loc[:, self.columns_]
            return frame
        a = np.asarray(X)
        if a.ndim != 2:
            raise ValueError('X must be a two-dimensional table')
        columns = list(range(a.shape[1])) if fitting else self.columns_
        if len(columns) != a.shape[1]:
            raise ValueError('feature count changed')
        return pd.DataFrame(a, columns=columns)

    def _numeric(self, frame):
        if not self.numeric_:
            return np.empty((len(frame), 0)), np.empty((len(frame), 0))
        a = frame.loc[:, self.numeric_].to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(a).any():
            raise ValueError('infinite features are not supported')
        missing = np.isnan(a)
        return np.where(missing, self.median_, a), missing.astype(np.float64)

    def _categories(self, frame):
        # Prefixes distinguish the missing token from an actual string value.
        return frame.loc[:, self.categorical_].map(
            lambda v: 'MISSING:' if pd.isna(v) else 'VALUE:' + str(v)).to_numpy()

    def fit(self, X, y=None):
        if self.mode not in ('raw', 'quantile', 'ple'):
            raise ValueError('mode must be raw, quantile, or ple')
        if not isinstance(self.bins, int) or self.bins < 2:
            raise ValueError('bins must be an integer >= 2')
        frame = self._frame(X, fitting=True)
        if not len(frame) or not frame.shape[1]:
            raise ValueError('empty feature table')
        self.columns_ = frame.columns.tolist()
        self.n_features_in_ = frame.shape[1]
        if self.categorical_columns is None:
            self.categorical_ = [c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])]
        else:
            self.categorical_ = list(self.categorical_columns)
            if not set(self.categorical_) <= set(self.columns_):
                raise ValueError('unknown categorical column')
        self.numeric_ = [c for c in self.columns_ if c not in self.categorical_]
        raw = frame.loc[:, self.numeric_].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(raw).any():
            raise ValueError('infinite features are not supported')
        self.median_ = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.
                                 for v in raw.T])
        numeric, _ = self._numeric(frame)
        self.mean_ = numeric.mean(0)
        self.scale_ = np.maximum(numeric.std(0), 1e-8)
        self.quantile_ = None
        self.knots_ = []
        if self.numeric_ and self.mode == 'quantile':
            self.quantile_ = QuantileTransformer(n_quantiles=min(256, len(frame)),
                output_distribution='normal', subsample=None, random_state=self.random_state)
            self.quantile_.fit(numeric)
        if self.mode == 'ple':
            self.knots_ = [np.unique(np.quantile(v, np.linspace(0., 1., self.bins + 1)))
                           for v in numeric.T]
        self.onehot_ = None
        if self.categorical_:
            self.onehot_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                         dtype=np.float64)
            self.onehot_.fit(self._categories(frame))
        transformed = self.transform(frame)
        self.n_features_out_ = transformed.shape[1]
        self.fit_rows_ = len(frame)
        return self

    def transform(self, X):
        check_is_fitted(self, 'columns_')
        frame = self._frame(X)
        numeric, missing = self._numeric(frame)
        blocks = []
        if self.numeric_:
            base = ((numeric - self.mean_) / self.scale_ if self.quantile_ is None
                    else self.quantile_.transform(numeric))
            blocks.extend([base, missing])
            if self.mode == 'ple':
                for j, knots in enumerate(self.knots_):
                    if len(knots) > 1:
                        blocks.append(np.clip((numeric[:, j, None] - knots[:-1]) /
                                              np.diff(knots), 0., 1.))
        if self.categorical_:
            blocks.append(self.onehot_.transform(self._categories(frame)))
        result = np.concatenate(blocks, axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError('nonfinite mapped features')
        return result

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def take(X, indices):
    return X.iloc[indices] if isinstance(X, pd.DataFrame) else np.asarray(X)[indices]


@dataclass
class SearchFold:
    fit: np.ndarray
    stop: np.ndarray
    rank: np.ndarray

    def validate(self, n, groups=None, times=None):
        role_arrays = []
        for role in ('fit', 'stop', 'rank'):
            a = np.asarray(getattr(self, role))
            if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) or not len(a):
                raise ValueError('fold roles must be nonempty one-dimensional integer indices')
            if a.min() < 0 or a.max() >= n or len(np.unique(a)) != len(a):
                raise ValueError('invalid or duplicate fold row index')
            setattr(self, role, a)
            role_arrays.append(a)
        for i in range(3):
            for j in range(i):
                if np.intersect1d(role_arrays[i], role_arrays[j]).size:
                    raise ValueError('fit/stop/rank rows overlap')
                if groups is not None and np.intersect1d(np.asarray(groups)[role_arrays[i]],
                                                       np.asarray(groups)[role_arrays[j]]).size:
                    raise ValueError('group crosses fit/stop/rank roles')
        if times is not None:
            t = np.asarray(times)
            if not (max(t[self.fit]) < min(t[self.stop]) and max(t[self.stop]) < min(t[self.rank])):
                raise ValueError('temporal fold is not strictly forward ordered')
        return self

    def json(self):
        return {k: getattr(self, k).tolist() for k in ('fit', 'stop', 'rank')}


def make_search_folds(X, y, *, classification, n_splits=3, random_state=0,
                      groups=None, times=None):
    """Separate checkpoint selection from configuration ranking in every fold."""
    n, y = len(X), np.asarray(y)
    if n_splits < 2:
        raise ValueError('at least two ranking folds are required')
    folds = []
    if times is not None:
        unique = np.unique(times)
        if len(unique) < 12:
            raise ValueError('too few unique time points')
        for end_fraction in np.linspace(.6, 1., n_splits):
            end = int(round(end_fraction * len(unique)))
            score_start = max(1, end - max(2, len(unique) // 6))
            stop_start = max(1, score_start - max(2, len(unique) // 10))
            t = np.asarray(times)
            folds.append(SearchFold(np.where(t < unique[stop_start])[0],
                np.where((t >= unique[stop_start]) & (t < unique[score_start]))[0],
                np.where((t >= unique[score_start]) & (t <= unique[end - 1]))[0]).validate(n, times=t))
        return folds
    if groups is not None:
        splitter = (StratifiedGroupKFold(n_splits, shuffle=True, random_state=random_state)
                    if classification else GroupKFold(n_splits, shuffle=True, random_state=random_state))
        outer = splitter.split(np.zeros(n), y if classification else None, groups)
    else:
        splitter = (StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
                    if classification else KFold(n_splits, shuffle=True, random_state=random_state))
        outer = splitter.split(np.zeros(n), y if classification else None)
    for k, (remaining, rank) in enumerate(outer):
        if groups is None:
            fit, stop = train_test_split(remaining, test_size=.2, random_state=random_state + 73 + k,
                                         stratify=y[remaining] if classification else None)
        else:
            choices = GroupShuffleSplit(n_splits=20, test_size=.2, random_state=random_state + 73 + k)
            fit = stop = None
            for fi, si in choices.split(remaining, groups=np.asarray(groups)[remaining]):
                if not classification or len(np.unique(y[remaining[fi]])) == len(np.unique(y)):
                    fit, stop = remaining[fi], remaining[si]
                    break
            if fit is None:
                raise ValueError('cannot create group-disjoint training with all classes')
        folds.append(SearchFold(fit, stop, rank).validate(n, groups))
    return folds


@dataclass
class TreeCandidate:
    name: str
    tree: SingleTreeConfig = field(default_factory=SingleTreeConfig)
    representation: str = 'raw'
    bins: int = 8

    def json(self):
        return {'name': self.name, 'representation': self.representation,
                'bins': self.bins, 'tree': asdict(self.tree)}


def default_candidates(*, random_state=0):
    """A declared compact search space, not an assertion of optimal hyperparameters."""
    base = dict(epochs=512, batch_size=1024, learning_rate=.01,
                temperature=2., evaluate_every=8, random_state=random_state,
                weight_decay=1e-4, final_learning_rate_ratio=.1)
    specs = [
        ('raw_d2', 'raw', dict(depth=2)),
        ('raw_d4', 'raw', dict(depth=4)),
        ('raw_d6', 'raw', dict(depth=6)),
        ('raw_d8', 'raw', dict(depth=8)),
        ('rank_d4', 'quantile', dict(depth=4)),
        ('rank_d6', 'quantile', dict(depth=6)),
        ('rank_d8', 'quantile', dict(depth=8)),
        ('ple_d4', 'ple', dict(depth=4)),
        ('ple_d6', 'ple', dict(depth=6)),
        ('raw_regularized_d6', 'raw', dict(depth=6, temperature=1., learning_rate=.03,
                                           logit_penalty=.001, weight_decay=.001)),
        ('cart_leaf_d6', 'raw', dict(depth=6, temperature=1., readout='leaf', initializer='cart')),
        ('trinary_d4', 'raw', dict(depth=4, arity=3)),
    ]
    return [TreeCandidate(name, SingleTreeConfig(**(base | overrides)), representation)
            for name, representation, overrides in specs]


class MappedTree:
    """Exactly one trained tree with its own fitted feature representation."""
    def __init__(self, candidate: TreeCandidate, *, classification: bool):
        self.candidate = deepcopy(candidate)
        self.classification = classification

    def fit(self, X, y, *, eval_set, sample_weight=None):
        # Local import avoids a module cycle: the protected map subclasses the
        # legacy representation retained for frozen study/checkpoint replay.
        from .stable_features import StableFeatureMap
        self.encoder_ = StableFeatureMap(self.candidate.representation, self.candidate.bins,
                                        random_state=self.candidate.tree.random_state).fit(X)
        mapped = self.encoder_.transform(X)
        estimator = SingleTreeClassifier if self.classification else SingleTreeRegressor
        self.tree_ = estimator(deepcopy(self.candidate.tree)).fit(mapped, y, sample_weight,
            eval_set=(self.encoder_.transform(eval_set[0]), eval_set[1], *eval_set[2:]))
        self.n_features_in_ = self.encoder_.n_features_in_
        if self.classification:
            self.classes_ = self.tree_.classes_.copy()
        return self

    def predict(self, X, *, hard=False):
        return self.tree_.predict(self.encoder_.transform(X), hard=hard)

    def predict_proba(self, X, *, hard=False):
        if not self.classification:
            raise AttributeError('regression trees do not predict class probabilities')
        return self.tree_.predict_proba(self.encoder_.transform(X), hard=hard)

    def score_loss(self, X, y):
        if self.classification:
            return float(log_loss(y, self.predict_proba(X), labels=self.classes_))
        return float(math.sqrt(mean_squared_error(y, self.predict(X))))


@dataclass
class SearchConfig:
    budgets: tuple[int, ...] = (64, 512)
    finalists: int = 2
    reserve_deep_candidate: bool = True
    stability_weight: float = .5
    random_state: int = 0

    def __post_init__(self):
        if not self.budgets or any(not isinstance(b, int) or b < 1 for b in self.budgets):
            raise ValueError('budgets must be positive epoch counts')
        if list(self.budgets) != sorted(set(self.budgets)):
            raise ValueError('budgets must strictly increase')
        if self.finalists < 1 or not np.isfinite(self.stability_weight) or self.stability_weight < 0:
            raise ValueError('invalid finalist count or stability coefficient')


class TreeSearch(BaseEstimator):
    """Tune a single tree by independent-fold predictive outcomes.

    No test/audit input exists. ``callback`` receives each completed trial and
    enables durable logs without retaining hundreds of trained models in memory.
    """
    def __init__(self, *, classification=True, candidates=None, config=None):
        self.classification = classification
        self.candidates = candidates
        self.config = config

    def fit(self, X, y, *, folds=None, groups=None, times=None, callback: Callable | None = None):
        self.config_ = deepcopy(self.config or SearchConfig())
        c = self.config_
        self.candidates_ = deepcopy(self.candidates or default_candidates(random_state=c.random_state))
        names = [candidate.name for candidate in self.candidates_]
        if len(set(names)) != len(names):
            raise ValueError('candidate names must be unique')
        y = np.asarray(y)
        self.folds_ = (make_search_folds(X, y, classification=self.classification,
             random_state=c.random_state, groups=groups, times=times) if folds is None else deepcopy(folds))
        if len(self.folds_) < 2:
            raise ValueError('at least two ranking folds are required')
        for fold in self.folds_:
            fold.validate(len(X), groups, times)
            if self.classification and set(np.unique(y[fold.fit])) != set(np.unique(y)):
                raise ValueError('every fit fold must include all classes')
        self.records_, self.rung_rankings_ = [], []
        active = self.candidates_[:]
        started = time.monotonic()
        for rung, budget in enumerate(c.budgets):
            rankings = []
            for candidate in active:
                fold_scores = []
                records = []
                for fold_id, fold in enumerate(self.folds_):
                    choice = deepcopy(candidate)
                    choice.tree.epochs = budget
                    # A fresh fit per rung: do not imply checkpoint continuation.
                    choice.tree.schedule_epochs = None
                    choice.tree.random_state = c.random_state + 1009 * fold_id
                    choice.tree.__post_init__()
                    run_start = time.monotonic()
                    record = dict(candidate=candidate.name, rung=rung, epochs=budget, fold=fold_id,
                                  config=choice.json(), status='success')
                    try:
                        model = MappedTree(choice, classification=self.classification).fit(
                            take(X, fold.fit), y[fold.fit], eval_set=(take(X, fold.stop), y[fold.stop]))
                        prediction = (model.predict_proba(take(X, fold.rank)) if self.classification
                                      else model.predict(take(X, fold.rank)))
                        score = (log_loss(y[fold.rank], prediction, labels=model.classes_)
                                 if self.classification else math.sqrt(mean_squared_error(y[fold.rank], prediction)))
                        if not math.isfinite(score):
                            raise FloatingPointError('nonfinite ranking score')
                        record.update(rank_loss=float(score), best_epoch=model.tree_.best_epoch_,
                            stop_loss=model.tree_.best_score_, features=model.encoder_.n_features_out_,
                            parameters=sum(p.numel() for p in model.tree_.model_.parameters()),
                            fit_rows=len(fold.fit), stop_rows=len(fold.stop), rank_rows=len(fold.rank),
                            optimizer_steps=model.tree_.optimizer_steps_, history=model.tree_.history_)
                        fold_scores.append(float(score))
                        del model
                    except (ValueError, RuntimeError, FloatingPointError, MemoryError) as error:
                        record.update(status='failed', error=f'{type(error).__name__}: {error}')
                        fold_scores.append(math.inf)
                    record['seconds'] = time.monotonic() - run_start
                    self.records_.append(record); records.append(record)
                    if callback:
                        callback(deepcopy(record))
                valid = np.isfinite(fold_scores).all()
                mean = float(np.mean(fold_scores)) if valid else math.inf
                std = float(np.std(fold_scores, ddof=1)) if valid else math.inf
                rankings.append(dict(candidate=candidate.name, mean=mean, std=std,
                    objective=mean + c.stability_weight * std / math.sqrt(len(self.folds_)),
                    depth=candidate.tree.depth, representation=candidate.representation,
                    fold_scores=fold_scores, complete=bool(valid)))
            rankings.sort(key=lambda r: (r['objective'], r['depth'], r['candidate']))
            self.rung_rankings_.append(dict(rung=rung, epochs=budget, ranking=rankings))
            eligible = [r for r in rankings if r['complete']]
            if not eligible:
                raise RuntimeError('all tree candidates failed; inspect records_')
            if rung < len(c.budgets) - 1:
                selected = [r['candidate'] for r in eligible[:c.finalists]]
                if c.reserve_deep_candidate:
                    max_depth = max(r['depth'] for r in eligible)
                    sentinel = next(r['candidate'] for r in eligible if r['depth'] == max_depth)
                    if sentinel not in selected:
                        selected.append(sentinel)
                active = [next(v for v in self.candidates_ if v.name == name) for name in selected]
        self.ranking_ = rankings
        self.best_candidate_ = deepcopy(next(v for v in self.candidates_ if v.name == eligible[0]['candidate']))
        self.best_candidate_.tree.epochs = c.budgets[-1]
        self.fit_seconds_ = time.monotonic() - started
        return self

    def refit(self, X, y, *, eval_set, candidate=None, random_state=None, epochs=None):
        check_is_fitted(self, 'best_candidate_')
        choice = deepcopy(self.best_candidate_ if candidate is None else candidate)
        choice.tree.epochs = self.config_.budgets[-1] if epochs is None else epochs
        choice.tree.random_state = self.config_.random_state if random_state is None else random_state
        return MappedTree(choice, classification=self.classification).fit(X, y, eval_set=eval_set)

    def summary(self):
        check_is_fitted(self, 'best_candidate_')
        return dict(best=self.best_candidate_.json(), rankings=self.rung_rankings_,
            search_config=asdict(self.config_), folds=[f.json() for f in self.folds_],
            elapsed_seconds=self.fit_seconds_, fits=len(self.records_),
            failed_fits=sum(r['status'] != 'success' for r in self.records_))


class _AutoTree(BaseEstimator):
    classification = False
    def __init__(self, candidates=None, search_config=None):
        self.candidates = candidates
        self.search_config = search_config

    def fit(self, X, y, *, eval_set, folds=None, groups=None, times=None, callback=None):
        """Search X/y only; use explicit eval_set only for the final refit."""
        self.search_ = TreeSearch(classification=self.classification,
            candidates=self.candidates, config=self.search_config).fit(
                X, y, folds=folds, groups=groups, times=times, callback=callback)
        self.member_ = self.search_.refit(X, y, eval_set=eval_set)
        self.n_features_in_ = self.member_.n_features_in_
        if self.classification:
            self.classes_ = self.member_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self, 'member_')
        return self.member_.predict(X)


class AutoTreeClassifier(ClassifierMixin, _AutoTree):
    classification = True
    def predict_proba(self, X):
        check_is_fitted(self, 'member_')
        return self.member_.predict_proba(X)


class AutoTreeRegressor(RegressorMixin, _AutoTree):
    pass
