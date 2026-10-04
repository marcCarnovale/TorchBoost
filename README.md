# TorchBoost

**Function-preserving neural–tree architecture expansion for tabular learning.**

TorchBoost is an experimental PyTorch research system for studying whether strong
tabular neural and tree predictors can be embedded inside a shared trainable
architecture and then expanded without destroying the inherited function.

## Main research result

The strongest current evidence is a HIGGS neural→tree hybrid result, not the
historical progressive-forest machinery.

A canonical five-layer ReLU MLP is represented exactly as depth-zero affine
tree layers. Each hidden layer can then grow a zero-at-birth residual-tree
refinement, so structural capacity is introduced **function-preservingly**.
The inherited MLP backbone and head stay frozen while only the newborn routing
and residual packets are trained.

On the 500k-row HIGGS development protocol, held-out learning of five positive
layerwise residual scales beat an otherwise identical fixed-scale adapter:

- fixed-scale ranking NLL/AUC: `0.57106812 / 0.76946808`
- held-out-scale ranking NLL/AUC: `0.57049915 / 0.76991939`
- delta: **−0.00056897 NLL / +0.00045131 AUC**

The result then reproduced directionally in all three predeclared independent
seeds. The mean replicated held-out-minus-fixed effect was approximately
**−0.000230371 NLL / +0.000293058 AUC**. Exact SHAs, run IDs, jobs, artifacts,
and per-seed values are recorded in `research/RESULTS.md`.

A corrected matched causal control is predeclared in
`research/higgs_scale_source_control_protocol.md`: the same five scales are
trained either from TRAIN or from a held-out ARCHITECTURE-SELECTION half with
matched update cadence, while a disjoint CHECKPOINT-SELECTION half chooses the
retained checkpoint. This tests whether the gain is specifically associated
with held-out architecture allocation rather than merely adding five trainable
supervised parameters.

The fresh HIGGS shadow audit at rows `[9,600,000, 10,100,000)` remains
**unopened**. External transfer is evaluated separately on seven public OpenML
datasets with five seeds, TRAIN-only numeric/categorical preprocessing, and
dataset-clustered inference.

### What is and is not claimed

Current evidence supports a narrow statement: a pretrained tabular MLP can be
expanded with zero-at-birth residual-tree structure, and held-out optimization
of the residual architecture weights reproducibly improves the hybrid on the
HIGGS development protocol.

It does **not** yet establish broad tabular state of the art, superiority over
properly tuned CatBoost/XGBoost/LightGBM, or successful optimization across the
entire architecture space described elsewhere in this repository.

## Broader experimental system

### Single power tree — secondary model-tree research path

A node can contribute an affine residual

\[
r_v(x)=b_v+x^\top\beta_v,
\]

so a path accumulates coarse-to-fine predictive corrections rather than only constant leaf values.
Hard Newton/model-tree proposals can be released into differentiable routing and trained with
hierarchical regularization. This substantially increases the capacity of one tree before ensembling.

### OOF-selected forest

`OOFForest` trains candidate members on inner bags, scores them using out-of-fold predictions,
retains only the strongest members, refits survivors on independent outer bags, and supports
uniform, inverse-loss, or softmax OOF-performance weighting. Member selection is therefore based on
held-out behavior rather than unconditional averaging.

### Progressive / boosted power trees

`UnifiedProgressiveClassifier` and `UnifiedProgressiveRegressor` add corrective trees sequentially.
Older trees can slow with age instead of being permanently frozen. The native trainer can combine
this with plastic anchors, dynamic structure, online local control, and capacitor/RLC thermal control.

## Axis-aligned by default; oblique routing is explicit

For ordinary tabular data, arbitrary rotations across unrelated columns are usually a poor prior.
The default proposal is axis-aligned. Experimental `proposal_mode="grouped_oblique"` restricts an
oblique gate to one declared semantic feature group; singleton groups recover ordinary axis splits,
and groups may overlap. This is intended for domain-informed or strongly evidenced feature groups,
not indiscriminate rotation of all columns.


## Data-adaptive experimental design

Power-tree structural learning defaults to data-adaptive complexity control. On larger fitting sets,
candidate splits can be screened cheaply, ranked by K-fold out-of-fold improvement, and then refit on
all data. Final affine packets can be averaged across large parameter-estimation bags. Cross-fitting
uses a capped design sample on very large nodes so structural validation does not make tree
construction quadratic in data size.

On smaller data, the same policy increases required rows per affine parameter, strengthens local
ridge pressure, and reduces feasible depth. A single permanent holdout is not the default: it wastes
scarce data and was empirically too conservative in development tests.

This is separate from final model selection: controller, selection, ranking, and audit data remain
distinct where the experiment protocol provides them.

## Adaptive mechanisms

The native engine implements:

- evidence-earned anchors and elastic pullback;
- yielding, permanent reference motion, work hardening, damage/breakage, recovery, and optional locks;
- capacitor and RLC corrective-energy controllers with independent cooling;
- thermal softening and thaw/reopening;
- local momentum variants, including circuit-state coupling;
- real growth/pruning with optimizer/controller/plastic-state migration;
- split-specific observations and an online local policy with delayed outcomes;
- hierarchical, feature, structural, routing, and output regularization.

Physics is **not** assumed to improve an underpowered predictor. Current development first makes the
base tree/forest statistically strong, then evaluates control mechanisms on long nonstationary
curricula where retention, selective reopening, and recovery can actually matter.

## Physics defaults and topology scaling

The ordinary routing temperature is 1.0. Physics-enabled experiments now use a neutral resting
temperature of 1.0 so enabling the controller does not silently sharpen every gate.

`PhysicsConfig(topology_normalization=True)` derives per-node resistance, inductance, heat capacity,
and cooling from whole-controller time constants. When topology changes, stored thermal and
inductive energy are preserved while component values are recalibrated. This prevents the same
controller configuration from changing meaning merely because a tree grows.

## Evidence and protocols

- `research/RESULTS.md` — auditable SHAs, Actions runs, jobs, artifacts, and
  valid/invalid evidence status.
- `research/NEURAL_TREE_ADAPTER_NOTE.md` — paper-shaped statement of the
  architecture, current HIGGS evidence, controls, and claim boundary.
- `research/higgs_scale_source_control_protocol.md` — corrected causal
  heldout-scale vs TRAIN-scale control.
- `research/external_adapter_benchmark_protocol.md` — frozen seven-dataset
  transfer study with TRAIN-only preprocessing and dataset-clustered inference.
- `experiments/higgs_shadow_protocol.json` — unopened final HIGGS shadow lock.

Secondary development evidence exists for affine-residual power trees,
grouped-oblique routing, recurring-domain memory/control, and
topology-normalized physical controllers. Those experiments remain exploratory
and are not promoted to the same evidentiary status as the replicated HIGGS
adapter result.

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
    n_trees=1,                 # single power tree
    linear_values=True,
    proposal_mode="hist_newton",
)

model = UnifiedProgressiveClassifier(cfg)
model.fit(
    X_train, y_train,
    control_set=(X_control, y_control),
    eval_set=(X_selection, y_selection),
)
p = model.predict_proba(X_test)
```

Final test data must never be supplied as controller or selection data.

## Research discipline

A configured mechanism, an activated mechanism, and a beneficial mechanism are three different
claims. Experiments record intervention timing, charge/temperature, anchor admission, structural
events, and selected checkpoints so late or inactive mechanisms are not credited for earlier model
quality. Negative controls are retained.

The draft research branch remains under active development. See `docs/research-program.md` and the
current experiment scripts for protocols and known limitations.