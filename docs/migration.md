# Package boundary and migration

The source-of-truth cleanup was PR #1, head `b788ca331f8b70f446376d350c474af3f4741f4b`,
merged on main as `fe4dc7915dfea4e84a157e33e5dd79ab5404aa37`.
Its `torchboost.py` is preserved byte-for-byte as `torchboost/legacy.py`, Git blob
`95e9af7ae59100f8113c2599cb4d953256b19125`. Existing imports remain available:

```python
from torchboost import SoftTree, TorchBoostModel, train_torchboost
```

They intentionally use the characterized legacy behavior, including known defects. The six
cleanup tests are retained. Additional characterization tests expose disconnected pruning,
unconstrained residuals, dead flags, missing-value poisoning through attention, stochastic
diversity re-evaluation, and evaluated/checkpoint temperature mismatch. These tests are not
endorsements of those behaviors. New work should use the independently named APIs.

| Module | Owner/responsibility |
|---|---|
| `legacy.py` | Frozen compatibility/reference surface; no claims of real stagewise boosting |
| `objectives.py` | Analytic binary-logistic loss, gradient, Hessian, weighted intercept |
| `trees.py` | Binary soft routing, routing traces, CART warm start, coupled leaf solve |
| `stagewise.py` | Estimator, training-only candidate optimization, additive stage management |
| `metrics.py` | Stateless observations and bounded per-node temporal tracking |
| `control.py` | Explicit capacitor source/discharge/heat/cooling state transitions |
| `export.py` | Pure NumPy evaluator of versioned hard-tree JSON |
| `benchmarks/` | Prespecified comparisons, all raw runs, split identities and limitations |

`StagewiseBinaryClassifier` exposes sklearn-style fit/predict/predict_proba/decision_function,
get_params/set_params/clone behavior. This is not a claim of passing the entire sklearn estimator
check suite. Classification forward scores and probabilities are explicitly separate.

New checks use `model.save(path)` and `StagewiseBinaryClassifier.load(path)`, storing primitives
and tensors, loaded with `weights_only=True`. They restore predictions, retained controller
state, metrics and histories. They do not restore an unfinished optimizer, worker queues or
mid-stage data-loader position. Do not advertise exact training replay until that exists.

Hard JSON predictions are tested against `predict_proba(..., hard=True)`. A hard export is not
claimed to equal the original soft model: the benchmark records their probability discrepancy.
Train-fitted imputation/scaling and the left-on-zero tie rule are part of the export schema.

The planned plasticity, online scheduler, dynamic allocation and specialized heads will use
new explicit modules/configurations; no silent reinterpretation of legacy flags. Before a
legacy defect is fixed in-place, introduce an explicit versioned mode and migration tests.
