# TorchBoost

**Differentiable Newton boosting, with a research program in adaptive structure and plasticity.**

TorchBoost is being rebuilt from an AI-assisted 2024 overnight prototype into a reproducible
research project. The aim is an ensemble that learns where to specialize, grow, preserve,
reconsider and remove structure. The aim is ambitious; the claims below are limited to what
is implemented and tested.

## Two deliberately separate APIs

| API | What it does today |
|---|---|
| `StagewiseBinaryClassifier` | Adds differentiable binary soft trees sequentially using real logistic gradients/Hessians, coupled soft-leaf Newton solves, shrinkage and exact-loss backtracking. |
| `TorchBoostModel` | Preserved legacy jointly optimized soft-tree/attention ensemble. It is not classical stagewise gradient boosting. |

The new binary baseline includes weighted samples, training-only missing-value preprocessing,
minibatching, deterministic seeds, sklearn-style inference, checkpoint round-trips, hard-tree
JSON export, and independent split metrics. An optional capacitor controller injects charge
on controller-validation regression, discharges electrical energy into corrective heat, and
cools independently. Its temperatures affect routing softness, not the optimizer learning rate.

**Not implemented yet:** dynamic sparse growth/deletion, specialized multiclass attention heads,
plastic anchor yielding/breakage, local inductive momentum and learned online policies. They are
first-class requirements in the [research RFC](docs/research-program.md) and
[machine-readable feature ledger](docs/feature-ledger.json), not advertised features.

## Install and run

Python 3.10+; a supported PyTorch installation is required.

```bash
git clone https://github.com/marcCarnovale/TorchBoost.git
cd TorchBoost
python -m pip install -e '.[dev,benchmark]'
python examples/stagewise_binary.py
pytest -q
python -m benchmarks.run_binary --seeds 0 1 2
```

```python
from torchboost import StagewiseBinaryClassifier

model = StagewiseBinaryClassifier(
    n_estimators=40,
    max_depth=3,
    epochs_per_stage=20,
    init="random",        # "cart" is an explicitly disclosed hybrid warm start
    random_state=42,
)
model.fit(X_train, y_train, eval_set=(X_selection, y_selection))
probabilities = model.predict_proba(X_test)  # two columns in model.classes_ order
model.save("model.pt")
model.export_json("hard_model.json")
```

Accepted trees are immutable during later stages. The final model is the best evaluated prefix,
including the intercept-only candidate. A hard export matches explicit hard inference; it is
not promised to match soft predictions. The benchmark reports that discrepancy.

Enable the experimental controller only with a separate control split:

```python
model = StagewiseBinaryClassifier(controller={"cooling_law": "linear"})
model.fit(X_train, y_train,
          control_set=(X_controller, y_controller),
          eval_set=(X_selection, y_selection))
```

Do not feed the final test set to either adaptation or selection. The minimal controller affects
only the candidate tree and uses uniform node resistances. It is not the complete proposed
forest-wide electrical network.

## Evidence, not a leaderboard claim

The [recorded smoke benchmark](docs/benchmark-smoke.md) contains all 30 runs: two small binary
datasets, three fixed split seeds, four TorchBoost variants and XGBoost. It records AUC, NLL,
accuracy, balanced accuracy, calibration, runtime, selected stages and soft/hard discrepancy.
The results are encouraging on these splits, but XGBoost is substantially faster. There is no
matched tuning budget, broad dataset coverage, or established state-of-the-art claim. Peak
training memory has not been measured; serialized tensor bytes are not a substitute.

The new solver allocates a complete binary tree and a dense leaf covariance matrix. It is a
small-data reference, not the promised deep sparse engine. CPU tests and comparisons are
recorded; GPU throughput and distributed operation remain unvalidated.

## Research direction

The distinctive hypotheses are **corrective heat allocation**, **evidence-earned plastic
anchors whose pullback can yield or break**, and **real dynamic growth/pruning**, combined with
learned output specialization. These mechanisms must remain independently switchable and
falsifiable. The [RFC](docs/research-program.md) states equations, owners, prior art and acceptance
gates; the [migration guide](docs/migration.md) explains compatibility and known legacy defects.

Metric collection is independent of plasticity:

    training -> SplitMetricsCollector -> PerformanceTracker
                                         -> OnlineScheduler [planned]
                                              -> PlasticityModule [planned]

A shared learner must preserve node-specific histories and outcomes. Frozen means preserved,
not removed. Thawing means reconsidering, not resetting. Temperature, learning rate, momentum,
plastic consolidation and structural existence are different controls.

## Legacy compatibility and provenance

`from torchboost import SoftTree, TorchBoostModel, train_torchboost` still resolves to the
characterized cleanup implementation, preserved byte-for-byte in `torchboost/legacy.py`.
Known issues remain there deliberately as a reference, including disconnected pruning and
snapshot timing. New code does not silently reuse those semantics. Research ideas originated
in the maintainer's design conversations; generated code is subject to the same tests and
review standards as any other implementation. See [AGENTS.md](AGENTS.md).

MIT license; see [LICENSE](LICENSE).
