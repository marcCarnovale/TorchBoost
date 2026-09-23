# TorchBoost

An experimental PyTorch implementation of jointly trained differentiable soft decision-tree ensembles.

## Important terminology

Despite the historical repository name, the current implementation is **not classical gradient boosting**: its trees and attention network are optimized jointly with backpropagation rather than added sequentially to fit residuals. The name is retained for repository continuity while the algorithm and evidence are clarified.

The project is research code. It is not a drop-in replacement for XGBoost, LightGBM, or scikit-learn estimators, and it has not established competitive accuracy, speed, calibration, or interpretability.

## Implemented behavior

- differentiable oblique soft-tree routing;
- jointly trained ensembles with input-dependent tree weights;
- regression, binary classification, multiclass classification, and multitarget outputs;
- optional tree/feature dropout and several experimental regularizers;
- scheduled temperature hardening;
- early stopping and learning-rate scheduling; and
- a weight-magnitude feature-importance proxy.

Classification `forward` calls return logits. Use `predict_proba` for probabilities.

Rows containing any missing feature currently receive neutral `0.5` routing at every node. This is a conservative experimental behavior, not learned per-feature missing-value handling.

## Install

Python 3.10 or newer is required.

```bash
git clone https://github.com/marcCarnovale/TorchBoost.git
cd TorchBoost
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Minimal example

```bash
python examples/minimal.py
```

The basic API is:

```python
import torch
from torchboost import TorchBoostModel

model = TorchBoostModel(
    num_trees=8,
    input_dim=10,
    tree_depth=3,
    task_type="multiclass_classification",
    num_classes=5,
    dropout_rate=0.1,
)

logits = model(torch.randn(32, 10))
probabilities = model.predict_proba(torch.randn(32, 10))
```

`train_torchboost` provides a simple full-batch research training loop. For larger data or controlled experiments, write an explicit minibatch loop around the module instead.

## Verification

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest -q
python examples/minimal.py
```

The tests cover routing-probability invariants, output contracts, multitarget shapes, invalid configuration, and a tree-dropout backward-pass edge case.

## Known limitations

- This is a jointly trained ensemble, not sequential gradient boosting.
- Benchmark comparisons against established tree and neural-tabular methods have not yet been added.
- Several regularizers and hardening schedules remain experimental and need ablation studies.
- The feature-importance score is a normalized weight-magnitude proxy, not a validated attribution method.
- Missing values are handled at row level rather than with learned per-split routing.
- The included trainer is full batch and intended for small experiments.
- Public API and checkpoint compatibility are not yet stable.

The earlier README described a single training trace as possible grokking/deep double descent. That claim has been removed because the trace did not constitute a controlled, reproducible demonstration.

## License

MIT; see [`LICENSE`](LICENSE).
