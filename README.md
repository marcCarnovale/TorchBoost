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

Recorded development results include:

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