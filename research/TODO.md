# TorchBoost Research TODO

Last updated: 2026-09-27

This is the persistent research-priority document for the
`research/unified-progressive-2026-09-24` program.

The central change in strategy is that TorchBoost is an **architecture space**,
not the historical 40--50-tree progressive configuration.  CatBoost and the
canonical MLP are now tested representation endpoints inside that space.  The
research goal is to calibrate those endpoints and then identify which
TorchBoost degrees of freedom improve on them.

See also `research/architecture-corners.md`.

## Current evidence

### Frozen HIGGS references

Canonical run: `36261511922`.

At 500k training rows (legacy opened audit):

| family | audit NLL | audit AUC |
| --- | ---: | ---: |
| CatBoost | 0.57871951 | 0.76182001 |
| MLP | 0.57205481 | 0.76757062 |
| historical TorchBoost | 0.60685535 | 0.72780999 |

The historical TorchBoost configuration is **not** the definition of the
TorchBoost architecture space.  It is one point in that space.

### Scaling experiments already answered

- Constant-exposure correction `36283928979`: restoring roughly two
  differentiable sample passes did not materially close the gap.  500k moved
  only slightly; 1M was essentially flat/slightly worse; 3M exceeded the
  six-hour hosted-runner budget.  Do not rerun unchanged.
- High-capacity rolling experiment `36288563045`: more trees/depth/nominal
  optimizer exposure did not solve the problem.  500k and 1M were worse than
  historical Progressive TorchBoost; the 1M trajectory destabilized.  The
  rolling approximation had also removed useful global adaptation semantics.
  Do not interpret this as a negative result for the full TorchBoost space.
- Unified mechanism screen `36323333930`:
  - hard histogram-Newton alone was poor;
  - releasing the same construction into differentiable oblique routing
    improved ranking NLL by about 0.0215 and AUC by about 0.0275;
  - global scalar-rate refitting improved training but failed the selection
    gate, so scalar reweighting alone is not the missing coordination;
  - affine-oblique arms hit the six-hour timeout and produced no valid final
    result.  Do not blindly rerun the same implementation/schedule.

### Important optimization diagnosis

The old mechanism screen's reported aggregate sample passes were not comparable
to MLP epochs.  Proposal sampling and differentiable-SGD sampling were coupled:
with `row_subsample=.20`, refinement repeatedly sampled from the proposal pool
instead of the whole training distribution.  With a small active window, an
individual tree received far less than a full-data MLP-like training regime.

Future code must track separately:

1. proposal rows;
2. SGD/refinement rows;
3. unique-row coverage;
4. examples seen per live parameter block;
5. full-data-equivalent passes per component;
6. global/end-to-end polish exposure.

## P0 — calibrate the two exact endpoints

Nothing hybrid is interpretable until these are done.

### P0.1 MLP endpoint: training equivalence

Representation containment is already tested: a depth-zero affine tree layer is
a dense affine layer, and `CompositionalTreeNetwork.from_mlp` reproduces MLP
logits exactly.  The canonical 21 -> 300 x 5 -> 1 corner has exactly 368,101
trainable parameters.

Next:

- [x] Add a canonical-HIGGS MLP-corner training harness. Launched via `HIGGS endpoint calibration`.
- [ ] Match the frozen MLP exactly: same initialization, standardization,
      width/depth, ReLU, dropout, AdamW, LR, weight decay, batch size, epoch
      count, batch ordering, clipping and selection rule.
- [ ] Verify initial logits agree with the ordinary MLP.
- [ ] Verify one-step gradients/updates agree under controlled dropout masks.
- [ ] Verify the complete 500k learning curve and ranking metric agree to a
      tight numerical tolerance.
- [ ] Save the trained depth-zero TorchBoost checkpoint as the **neural anchor**.
- [ ] Record wall time, parameter count and memory so hybrid costs can be
      compared fairly.

**Exit criterion:** the depth-zero TorchBoost endpoint reproduces the canonical
MLP result/learning curve.  If it does not, fix the implementation before doing
hybrid research.

### P0.2 CatBoost endpoint: fitted-model equivalence

Representation containment is already tested: numerical CatBoost symmetric
trees can be imported into `ObliviousSoftForest`, with CI testing raw-prediction
agreement.

Next:

- [~] Produce/save the exact frozen 500k CatBoost model (JSON) under the
      canonical data/split/seed/configuration. Calibration job launched; mark complete when artifact lands.
- [ ] Import it into TorchBoost and verify raw logits, ranking NLL and ranking
      AUC agree to numerical tolerance.
- [ ] Save the imported model as the **boosting anchor**.
- [ ] Measure hard TorchBoost inference cost and memory versus CatBoost.
- [ ] Keep import support explicit about unsupported categorical/non-numerical
      CatBoost split types rather than claiming broader exactness.

**Exit criterion:** the trained CatBoost predictor is a reproducible TorchBoost
checkpoint with indistinguishable predictions.

### P0.3 Native boosting calibration

Exact import proves representation containment, not optimizer equivalence.

- [ ] Implement/finish a TorchBoost-native symmetric histogram-Newton boosting
      mode with the same structural scale as the frozen comparator:
      ~1536 trees, depth 10, hard axis-aligned shared-by-depth routing, small
      shrinkage, strong leaf L2, CatBoost-scale numerical binning.
- [ ] Separate fast proposal construction from any differentiable machinery.
- [ ] Benchmark native hard-corner training against CatBoost at 500k.
- [ ] Do not require the Python implementation to match CatBoost runtime before
      testing hybrids; predictive calibration is the first objective.

**Exit criterion:** either native hard-corner training is competitive with the
CatBoost predictor, or the gap is documented as an optimizer/constructor gap
while the exact imported CatBoost anchor remains available for hybrid work.

## P1 — controlled departures from each calibrated endpoint

The rule is **anchor + one named relaxation**.  A candidate that hurts
selection/ranking does not replace its anchor.

### P1.A From CatBoost toward TorchBoost

Start from the exact imported CatBoost anchor, so tree-builder quality cannot
confound the experiment.

- [ ] A1: hard imported CatBoost -> soft gates only.
- [ ] A2: soft -> oblique routing, initialized exactly at the axis-aligned
      CatBoost splits.
- [ ] A3: add affine residual packets initialized at zero, preserving the
      CatBoost function.
- [ ] A4: allow short local differentiable refinement on **full-distribution
      SGD**, distinct from proposal sampling.
- [ ] A5: periodic global polish of tree parameters, not merely scalar rates.
- [ ] A6: optional attention/input-dependent coefficients only after A1--A5
      establish value.
- [ ] At every step keep the original anchor checkpoint available and require
      selection/ranking improvement before advancing.

This trajectory answers: *given a predictor already as good as CatBoost, which
neural/differentiable freedoms add value?*

### P1.B From MLP toward TorchBoost

Start from the exact trained depth-zero MLP anchor.

- [ ] B1: function-preserving growth of zero residual tree branches.
- [ ] B2: train only the new residual branches; keep the original MLP function
      available as the zero-refinement checkpoint.
- [ ] B3: release routing from hard/neutral initialization into soft oblique
      specialization.
- [ ] B4: compare scalar versus affine/vector residual packets.
- [ ] B5: introduce Newton/boosting proposals for new residual branches.
- [ ] B6: test sparse/dynamic topology and pruning only after specialization
      improves ranking.
- [ ] B7: compare full end-to-end updates with rolling/frozen approximations,
      measuring the predictive cost of each approximation.

This trajectory answers: *given a predictor already as good as the MLP, can
tree structure/specialization improve it or achieve the same quality more
efficiently?*

## P1.C Fix exposure semantics before progressive experiments

- [ ] Split `proposal_subsample` from `refinement_subsample`.
- [ ] Default differentiable refinement to the full training distribution even
      when proposals use a small row sample.
- [ ] Add diagnostics for per-tree/per-layer unique row coverage and effective
      full-data passes.
- [ ] Add a periodic global-polish phase with an explicit compute budget.
- [ ] Make freezing/rolling an optimization approximation that can be toggled
      against full end-to-end training, not an implicit definition of the
      architecture.
- [ ] Checkpoint long HIGGS jobs by stage so a hosted-runner timeout does not
      destroy six hours of computation.

## P2 — meet in the middle

Only after at least one departure from each endpoint improves its anchor:

- [ ] Define a compact coordinate/config object for the interpolation axes:
      composition, routing hardness, routing geometry, topology, packet type,
      construction mode, optimization scope, exposure, capacity and
      regularization.
- [ ] Run matched hybrids from both directions and determine whether they
      converge toward the same useful region.
- [ ] Test latent/compositional state between tree blocks if additive tree
      outputs remain the limiting factor.
- [ ] Scale the best fixed hybrid to 1M, then 3M only after 500k evidence shows a
      meaningful gain.
- [ ] Compare quality *and* compute/memory/latency, not NLL alone.

## P3 — adaptive mechanisms after the backbone is competitive

Physics/plasticity remain research mechanisms, but they are not the current
explanation for the HIGGS gap.

- [ ] Reintroduce plasticity only on a competitive calibrated backbone.
- [ ] Test electrical/thermal controllers on problems/regimes where adaptation,
      forgetting, structural churn or long-horizon stability gives them a
      plausible advantage.
- [ ] Require mechanism-specific ablations against a matched nonphysical
      scheduler/control.
- [ ] Keep dynamic allocation/pruning experiments, but do not use them to mask
      a weak base predictor.

## Experiment discipline

### HIGGS data

The fresh shadow audit is locked in
`experiments/higgs_shadow_protocol.json`:

- rows `[9,600,000, 10,100,000)`;
- 500,000 rows;
- **do not open during endpoint calibration or architecture search**.

The legacy final-500k audit has already been opened and is historical comparison
data only.

- [ ] Use train + selection for fitting/model selection.
- [ ] Use ranking for architecture comparison.
- [ ] Freeze one final architecture/configuration before opening the shadow
      audit.
- [ ] Record the exact commit, environment, data SHA, split, seed, parameter
      count, examples seen and wall time for every serious run.

### Stop rules

Do **not** spend large compute merely because a mechanism exists.

Stop/redirect an arm when:

- it cannot reproduce the baseline it claims to generalize;
- increasing capacity/exposure does not improve selection/ranking;
- a cheaper ablation explains the gain;
- the implementation repeatedly reaches runner timeout without checkpoints;
- it explores only another nearby point in the old progressive-forest
  neighborhood.

## Immediate execution order

1. **MLP training-equivalence harness.**
2. **Canonical CatBoost model export + exact TorchBoost import/checkpoint.**
3. **CatBoost-anchor A1/A2 experiments: soft then oblique release.**
4. **MLP-anchor B1/B2 experiments: function-preserving growth then residual
   branch training.**
5. Fix proposal-vs-refinement sampling and checkpointing in the progressive
   trainer.
6. Compare the first successful departures from both anchors.
7. Only then decide whether affine packets, global end-to-end polish or latent
   composition deserves the next major HIGGS campaign.
8. Do not rerun 3M or the timed-out affine screen unchanged.

## Current infrastructure status

At the time this roadmap was written:

- branch endpoint head before this document:
  `6d87070c1b165f938d4290ebf4d2e39670e26f42`;
- CI run `36345055761` is green;
- lint, Python 3.11/3.12 tests, mechanism monitor and CatBoost performance
  ratchet all pass;
- exact CatBoost/MLP representation-corner tests are part of the green test
  suite.

The next substantial work should therefore be endpoint **training calibration**,
not more infrastructure repair.
