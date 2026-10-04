# TorchBoost

**Function-preserving neural–tree architecture expansion for tabular learning.**

TorchBoost is an experimental PyTorch research system for expanding a trained tabular neural network into a richer tree-structured model **without changing the inherited function at the moment of expansion**.

The current research result is a neural→tree residual adapter, not a claim of tabular state of the art.

## Result at a glance

A pretrained ReLU MLP is embedded exactly as depth-zero affine tree layers. Each hidden layer can then grow a zero-at-birth residual-tree refinement while the inherited MLP backbone and head remain frozen.

| Evidence | Result |
|---|---|
| HIGGS development comparison | held-out architecture scales beat a matched fixed-scale adapter |
| Predeclared HIGGS replication | favorable learned-vs-fixed direction in **3/3** independent seeds |
| Matched scale-source control | held-out-scale beats TRAIN-scale on the corrected canonical control |
| External transfer | residual adapter beats its inherited MLP on **7/7** public OpenML datasets in both NLL and AUC |
| Strong tree baselines | CatBoost, XGBoost, and LightGBM still win on the external benchmark |
| Fresh HIGGS shadow audit | **unopened** |

### HIGGS adapter evidence

Initial 500k-row development result:

- fixed-scale ranking NLL/AUC: `0.57106812 / 0.76946808`
- held-out-scale ranking NLL/AUC: `0.57049915 / 0.76991939`
- delta: **−0.00056897 NLL / +0.00045131 AUC**

The direction reproduced in all three predeclared independent seeds; mean learned-minus-fixed ranking effect was approximately **−0.000230371 NLL / +0.000293058 AUC**.

The corrected matched scale-source control then compared the *same five scale parameters* when optimized from TRAIN versus a disjoint ARCHITECTURE-SELECTION pool, with matched update cadence and a separate CHECKPOINT-SELECTION pool. On that canonical control:

- TRAIN-scale ranking: `0.571734560 NLL / 0.768675883 AUC`
- held-out-scale ranking: `0.570802453 NLL / 0.769270809 AUC`
- heldout − TRAIN: **−0.000932107 NLL / +0.000594926 AUC**

This is one corrected causal-control seed, not yet a replicated causal claim.

### External transfer

The frozen transfer study evaluated **7 public OpenML binary datasets × 5 predeclared seeds = 35/35 completed cells**, using TRAIN-only preprocessing and dataset-clustered inference.

Against the inherited MLP, the learned residual adapter improved **7/7 datasets** in both metrics:

- mean dataset ΔNLL: **−0.004773**, 95% dataset-bootstrap CI `[-0.007932, -0.002308]`
- mean dataset ΔAUC: **+0.003448**, 95% CI `[+0.001291, +0.006955]`

Held-out learned scales versus fixed residual scales were much weaker cross-dataset:

- mean ΔNLL: `−0.000391`, 5/7 dataset wins, CI crossing zero
- mean ΔAUC: `+0.000042`, 4/7 wins, CI crossing zero

The current architecture **does not beat** CatBoost, XGBoost, or LightGBM on this external benchmark. That negative result is part of the evidence, not hidden.

Exact SHAs, Actions run/job/artifact IDs, per-seed values, invalidated controls, and claim boundaries are maintained in [`research/RESULTS.md`](research/RESULTS.md).

## What is claimed

Current evidence supports the following narrow statement:

> A trained tabular MLP can be embedded exactly in a tree-expandable architecture, zero-at-birth residual-tree capacity can be added without perturbing the inherited predictor, and training that residual capacity improves the inherited MLP across the seven-dataset external transfer benchmark. On HIGGS, held-out learning of residual architecture weights also reproducibly beats a matched fixed-scale adapter, and a corrected canonical control favors held-out over TRAIN-sourced scale updates.

It does **not** establish broad tabular state of the art, superiority over tuned tree ensembles, or a final unseen HIGGS result.

## Architecture

For a dense hidden layer

```text
h(x) = W x + b
```

TorchBoost represents the layer exactly as a depth-zero affine tree packet. Structural growth adds a residual refinement

```text
h_new(x) = h(x) + alpha * r(x),     r(x) = 0 at birth.
```

Therefore `h_new == h` at the growth event, regardless of the positive architecture scale `alpha`. Subsequent training updates only newborn routing/residual packets; the inherited predictor can remain frozen.

The HIGGS adapter uses five positive layerwise residual scales. The corrected scale-source control separates:

- **TRAIN** — residual packet/routing optimization;
- **ARCHITECTURE-SELECTION** — held-out scale optimization;
- **CHECKPOINT-SELECTION** — retained-checkpoint choice;
- **RANKING** — evaluation only;
- **SHADOW AUDIT** — sealed final evaluation only.

## Reproducibility and audit discipline

The research program treats a configured mechanism, an activated mechanism, and a beneficial mechanism as different claims.

- Every headline quantitative result is tied to a source SHA, Actions run, job, and artifact.
- Failed and invalid controls remain documented.
- The fresh HIGGS shadow audit at rows `[9,600,000, 10,100,000)` is locked in `experiments/higgs_shadow_protocol.json` and remains **unopened**.
- The previously opened legacy final-500k audit is retained only for frozen historical comparisons.
- No shadow metric is used for architecture, seed, checkpoint, or hyperparameter selection.

See:

- [`research/RESULTS.md`](research/RESULTS.md) — auditable evidence ledger
- [`research/NEURAL_TREE_ADAPTER_NOTE.md`](research/NEURAL_TREE_ADAPTER_NOTE.md) — paper-shaped architecture note and claim boundary
- [`research/higgs_scale_source_control_protocol.md`](research/higgs_scale_source_control_protocol.md) — corrected matched causal control
- [`research/external_adapter_benchmark_protocol.md`](research/external_adapter_benchmark_protocol.md) — frozen 7-dataset × 5-seed transfer study
- [`experiments/higgs_shadow_protocol.json`](experiments/higgs_shadow_protocol.json) — sealed HIGGS shadow protocol

## Broader experimental system

The repository also contains secondary research paths for:

- affine-residual “power” trees;
- grouped-oblique routing;
- OOF-selected forests;
- progressive/boosted power trees;
- dynamic growth/pruning;
- evidence-earned plastic anchors;
- capacitor/RLC-inspired adaptive controllers and topology-normalized physical state.

These mechanisms are exploratory and are **not** promoted to the same evidentiary status as the neural→tree residual-adapter results.

## Install

```bash
git clone https://github.com/marcCarnovale/TorchBoost.git
cd TorchBoost
python -m pip install -e '.[dev,benchmark]'
pytest -q
```

## Minimal unified example

```python
from torchboost.adaptive import UnifiedConfig, UnifiedProgressiveClassifier

cfg = UnifiedConfig(
    n_trees=1,
    linear_values=True,
    proposal_mode="hist_newton",
)

model = UnifiedProgressiveClassifier(cfg)
model.fit(
    X_train,
    y_train,
    control_set=(X_control, y_control),
    eval_set=(X_selection, y_selection),
)
p = model.predict_proba(X_test)
```

Final test data must never be supplied as controller or selection data.

## Status

This is an active research repository. The current strongest evidence is the function-preserving residual-adapter program above; historical progressive-forest experiments remain in the repository for provenance and secondary research.
