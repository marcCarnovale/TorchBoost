# Current research status — 2026-09-24

## Statistical backbone

The current development default is a single affine-residual power tree. OOF-selected forests and
progressive additive power trees are supported as separate model families. Oblique routing is
experimental and group-constrained; ordinary axis-aligned proposals remain the default.

## Physics corrections in this revision

1. Neutral resting routing temperature: physics no longer changes the resting geometry merely by
   being enabled.
2. `topology_normalization=True`: whole-controller discharge, inductive, cooling, and heat-capacity
   scales are converted to per-node components.
3. Topology recalibration preserves stored thermal and inductive energy when component values change.
4. Histories record event counts, cumulative charge injection, and peak temperature.

Focused tests include energy preservation and topology-invariant time-scale contracts.

## Recurring-domain development screen

Sequence: A -> B -> A -> B -> A. Separate generated samples are used at each stage and fixed audit
sets measure both A and B. These are short mechanism-development runs (16 updates/stage), not a
general benchmark.

| seed | model | cumulative current-domain loss proxy | final A audit |
|---|---|---:|---:|
| 17 | no controls | 3.48018 | 0.73905 |
| 17 | topology-normalized capacitor | 3.39323 | 0.69137 |
| 29 | no controls | 3.57359 | 0.76862 |
| 29 | topology-normalized capacitor | 3.51490 | 0.72752 |

On seed 17, plasticity without electrical control was numerically indistinguishable from no controls
at this horizon. Capacitor control was stronger than the tested RLC configuration. This argues for
tuning the physical dynamics rather than assuming additional components help.

## Transient shock screen

A -> noisy/corrupted A -> A, 24 updates/stage. On seed 17 the full adaptive package had slightly
higher loss during the shock but lower loss after recovery (0.71311 versus
0.72002). Peak temperature stayed below the thaw threshold, so this result
cannot be attributed to thermal thaw.

## Interpretation

The corrected capacitor signal is promising on recurring regimes, but the physics is not yet
established. The next experiments must be much longer and include matched simple controls:
scheduled thaw, validation-triggered nonphysical thaw, capacitor, RLC, plasticity-only, and the full
system. CatBoost is the primary strong tabular reference; physics is expected to contribute
incrementally after the base predictor is competitive.


## Topology-recalibration follow-up

The first normalization implementation preserved each branch's stored energy separately, which could
create local temperature spikes when new nodes changed per-node heat capacity. The controller now
preserves **system** thermal and inductive energy under one global rescaling while retaining relative
state patterns; new nodes enter at the pre-change mean temperature. The focused suite remains 49/49.

On seed 17 after this correction (16 updates/stage):

| controller | cumulative current-domain loss proxy | final returned-A audit | peak temperature |
|---|---:|---:|---:|
| no controls | 3.48018 | 0.73905 | 1.000 |
| fixed non-electrical heat pulse | 3.36976 | 0.70043 | 1.992 |
| topology-normalized capacitor | 3.37751 | **0.67664** | 2.535 |

The simple fixed pulse has slightly lower cumulative loss while the capacitor has materially better
final returned-domain loss. Therefore the present experiment does **not** establish that electrical
control dominates a simple thaw schedule. It does show that the corrected capacitor behavior remains
useful after removing the topology-energy confound. Longer matched experiments are required.


## Base-model diagnostic against CatBoost

A fresh smaller-data context-dependent affine task (4,096 fitting rows) exposes a current weakness.
The ordinary affine model-tree proposal reaches low training loss quickly but its selection loss
deteriorates after the first few checkpoints. At 96 updates its selected audit NLL was 0.69773 versus
0.68717 for a 256-tree CatBoost reference. Extending to 192 updates did not help: the selected
checkpoint remained step 16 while later training continued to overfit.

An experimental honest node-split proposal (`proposal_holdout=.25`) prevented the same aggressive
split overfit but overcorrected: it selected the intercept-only model (audit 0.69274). This option is
therefore retained as experimental and defaults to zero.

Interpretation: the base-model gap is currently **proposal/generalization quality**, not insufficient
training duration. The next statistical work should use cross-fitted or regularized model-tree split
selection rather than relying on physics to rescue an overfit proposal.


## Cross-fitted structural design and data-adaptive complexity

The affine model-tree builder now separates **candidate screening**, **structural selection**, and
**final parameter estimation**.

For data-rich nodes:

1. candidate thresholds are screened using a cheap conditional affine Newton gain;
2. only the strongest candidates receive K-fold cross-fitted evaluation;
3. structural cross-fitting can use a capped design sample for scalability;
4. after the structure is selected, affine node parameters are estimated from multiple large bags
   and averaged;
5. the accepted differentiable tree is then trained on the full fitting set.

For data-poor nodes, `auto_complexity=True` increases the minimum rows per affine parameter, increases
ridge pressure locally, limits feasible depth, and can avoid cross-fitting when folds would be too
small. Thus the model does not spend the same degrees of freedom at N=600 as at N=12,000.

A soft information-criterion split cost remains in addition to cross-fitting. The single fixed holdout
proposal remains available only as an experimental comparison; it was too conservative in the earlier
small-data probe.

### Scaling screen

Fresh 16-feature context-dependent affine problem, one development seed per size, shallow diagnostic
budget (depth 2, 16 differentiable updates):

| fitting rows | in-sample proposal audit | auto design audit | CatBoost reference |
|---:|---:|---:|---:|
| 1,000 | 0.68807 | 0.68807 | 0.70675 |
| 4,096 | 0.68232 | **0.68070** | 0.67593 |
| 12,000 | 0.68600 | 0.68713 | **0.65820** |

The shallow 12k model is under-capacity, not overfit. Increasing the same auto-designed single tree to
depth 4 and training it for 512 updates changes the result materially.

### Deep single-tree comparison

At 12,000 fitting rows, depth 4, 512 updates:

| seed | TorchBoost single tree audit | selected CatBoost reference audit |
|---:|---:|---:|
| 71 | **0.62449** | 0.65820 |
| 72 | **0.61870** | 0.66114 |

The CatBoost reference search here is bounded (depth 6/8, 256 trees, L2=20; selected by the separate
selection split), not exhaustive. TorchBoost's best checkpoints are at updates 480 and 512,
respectively. The result therefore supports continued long training of the deep single tree; it does
not establish a broad best-in-class claim.

The important design conclusion is that **experimental design and model capacity must scale together**:
cross-fitting/bagging alone cannot rescue an under-capacity shallow tree, while deep flexible trees on
small data need stronger automatic complexity control.

## Overnight verification follow-up — 2026-09-25

All figures below come from separated development/evaluation or selection/ranking/audit runs on the
research branch; the blocking CatBoost ratchet remains unchanged.

### Long-horizon deep-tree control

At seed 71 and 2,048 updates, the no-control single power tree selected step 544 with audit NLL
0.633576; its last-iterate audit NLL was 0.644543. The full adaptive package selected the same step
with audit NLL 0.633818 and deteriorated to 0.692901 at the last iterate, despite 87 admitted anchors,
37 growth events, and nonzero electrical injection. Thus the current full controller does not rescue
late stationary-task drift and is slightly worse at the selected checkpoint in this run.

### Thermal-dose fairness control

Direct generic feedback was tuned on development seed 11 only. The development candidate chosen to
match the oracle pulse peak temperature transferred to evaluation seeds 17/29 with mean peak
temperature 1.4829, substantially below the pulse's 1.9917. Its mean gains versus no control were
1.66% on cumulative current-domain loss and 3.92% on returned-A loss; the unconstrained
performance-selected direct controller achieved 3.62% and 7.69% but ran hotter (mean peak
temperature 2.6228). Therefore the existing temperature-match experiment does not yet isolate
controller quality from thermal dose across seeds; a budget-normalized direct controller is the next
fair comparison.

### Covertype system benchmark

On binary Covertype (80k train / 10k control / 10k selection / 10k ranking / 20k audit), the
progressive 24x5 TorchBoost forest won the TorchBoost ranking split and reached audit NLL 0.34114,
substantially improving over the single power tree. A separately ranked CatBoost depth-10,
1,024-tree model reached 0.20218 audit NLL, leaving a large real-data gap. This negative result is
retained.

### Central-force diagnostic

At seed 83, 32k fitting examples, depth 6, and 1,024 updates, generic TorchBoost training recovered an
in-range inverse-power exponent 1.9713 with radial alignment 0.9972, but held-out farther-radius
transfer remained poor (alignment 0.7775, exponent 2.7519). CatBoost had strong OOD radial alignment
0.9911 but OOD exponent 0.7168. Neither model has yet demonstrated a globally transferable
inverse-square law.
