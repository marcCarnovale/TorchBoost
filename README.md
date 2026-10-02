# TorchBoost

**Adaptive differentiable model trees, selected forests, and progressive ensembles.**

TorchBoost is a research system for tabular learning that combines a strong statistical backbone
with explicit mechanisms for memory, plasticity, structural change, and physically motivated
control. The project is experimental: mechanisms are kept independently switchable and claims are
limited to recorded tests and experiments.

## Current model families

### Single power tree — current default research path

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

## Current evidence

### HIGGS architecture-space result

The current research branch treats CatBoost and a five-layer ReLU MLP as
calibrated corners inside a larger TorchBoost architecture space rather than
defining TorchBoost as one historical progressive-forest configuration.

On the canonical 500k-row, 21-feature HIGGS protocol:

- the depth-zero compositional TorchBoost endpoint reproduces the canonical MLP
  training path exactly;
- imported numerical CatBoost symmetric trees reproduce the fitted CatBoost
  predictor to numerical precision;
- growing zero-at-birth affine tree residual adapters in all five hidden layers,
  freezing the inherited MLP backbone/head, and training only the residual
  packets/routing improves materially over the MLP anchor;
- most importantly, held-out differentiable learning of the five residual
  scales improves over an otherwise identical fixed-scale adapter.

The decisive differentiable-adapter study is GitHub Actions run
`36525742176`, source SHA
`265a8e040a060080d198bb5fce5578a1ccc1825f`.  Both arms begin at residual
scale `sigmoid(-2) = 0.1192029` and receive the same residual/routing training
budget.  The fixed-scale adapter reached ranking NLL/AUC
`0.57106812 / 0.76946808`; the held-out learned-scale adapter reached
`0.57049915 / 0.76991939`, an improvement of `-0.00056897` NLL and
`+0.00045131` AUC from architecture-scale learning itself.  Relative to the
canonical MLP anchor in the same study, the learned adapter improved ranking by
about `-0.00304` NLL and `+0.00445` AUC.

The learned residual scales were approximately
`[0.1329, 0.0923, 0.0644, 0.0423, 0.0385]`: a data-selected taper from a
slightly strengthened first-layer correction to progressively smaller deeper
corrections.  This is evidence that differentiable held-out architecture
learning can discover a better hybrid point inside this restricted adapter
family, not merely that TorchBoost can represent one.

The replicated adapter mechanism is now centralized in
`torchboost/adaptive/residual_adapter.py`; HIGGS and external-transfer studies
use the same function-preserving model surgery rather than maintaining separate
copies.

A frozen external-transfer benchmark is also running across seven public
OpenML binary datasets and five predeclared seeds. It compares the learned
adapter against its MLP anchor, the identical fixed-scale adapter, CatBoost,
XGBoost, and LightGBM under train-only preprocessing and paired
train/selection/ranking splits. The workflow emits machine-readable per-cell
records plus an aggregate report with paired deltas, win/tie/loss counts,
standard errors, and deterministic bootstrap intervals. See
`research/external_adapter_benchmark_protocol.md`. No external benchmark
result is claimed here until that frozen matrix completes.

These are **selection/ranking development results**.  The fresh shadow audit
defined in `experiments/higgs_shadow_protocol.json` remains unopened.  No
claim of final unseen-test or broad benchmark superiority is made from these
numbers.

Recorded development results also include:

- one affine-residual power-tree configuration beating a much larger CatBoost ensemble on a controlled
  context-dependent affine problem; this is a development result, not a broad leaderboard claim;
- grouped-oblique routing helping when a related feature block is deliberately rotated, while hurting
  when the natural axis-aligned representation is already correct;
- recurring-domain A→B→A experiments where adaptive memory/control can improve return performance,
  while A→B→C can punish excessive retention;
- current topology-normalized recurring A→B→A→B→A screens where capacitor control improves over the
  matched no-control model on two development seeds. These are mechanism-development results, not
  evidence of general superiority.

The project target is stronger than XGBoost: comparisons should include CatBoost and other strong
tabular references. Physics/control improvements are expected to be incremental; the statistical
backbone must earn competitiveness on its own.

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