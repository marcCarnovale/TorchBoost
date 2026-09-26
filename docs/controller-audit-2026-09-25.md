# Controller audit and continuation protocol — 2026-09-25

## Verified defects and evidence classification

At source `94f59709e181147a13af696eaacd24bdc25eb650`, the experimental
`DirectFeedbackController.advance` injected heat before the parent's replay
validation. Repeating `(loss=2, step=1)` after losses 1 then 2 increased the
one-node temperature from 1.149531981659531 to 1.299531981659531 and the external
heat total from 0.03 to 0.06. Its state dictionary also omitted the accumulated
external heat. These are reproducible controller defects, not evidence that the
historical training runs necessarily exercised the defective replay path.

The replacement validates/replays before heating, includes external heat in a
whole-step energy ledger, preserves the augmented history on replay, and saves
and restores the source total. Normal monotonically advancing temperature
updates retain the existing heating/cooling sequence. The controller remains
experimental; no production default or electrical solver is replaced.

A separate protocol defect invalidates the phrase **untouched evaluation audit**
for the older recurring-controller comparisons. `experiments/long_regimes.py`
defines one global `AUD` dictionary from seeds 9000/9001. Both the development
candidate selection and the supposedly held-out training-seed runs score those
same observations through `run_direct` / `lr.run`. Changing training seeds to
17/29 does not make those audit observations independent of selection. Keep all
old numbers, but treat them as development evidence, not independent controller
confirmation. This does not establish that circuits never help.

The old deep direct-feedback arm also used different thermal settings from the
deep capacitor arm. The replacement protocol fixes cooling time, total heat
capacity, temperature ceiling, thaw scale, plastic softening, LR coupling, and
spatial allocation across capacitor/RLC/direct/cooling controls.

## Tested replacement

`experiments/deep_controller_protocol.py` retains the context-dependent affine
power-tree task and uses 12,000 train / 1,800 control / 1,000 selection / 1,000
ranking / 1,200 audit observations. The initial partition is different from the
older selection-2,000 protocol; scores are not numerically interchangeable.
Development seed 97 never scores audit. Its fixed direct-gain grid is ranked
both by development predictive ranking loss and, independently, by thermal
matching to capacitor. Confirmation seeds 101/103 generate new data and freeze
the development choices. Only the ranking winner and the preregistered no-control
reference receive confirmation audit scores.

Thermal matching requires both peak excursion and the right-endpoint integrated,
capacity-weighted excursion in ambient-to-thaw units to fall within 20% of the
capacitor target. The closest candidate is not silently called matched when this
criterion fails. Integrated thermal energy, above-thaw exposure, source heat,
resistor heat, cooling, venting, and ledger errors are retained as diagnostics.
The sums use controller dt and are not exact continuous-time ODE integrals.

Direct replay/checkpoint/source-ledger tests cover 1/7/63-node topologies and
venting. The existing optimizer-normalization test had an invalid rescaled
fixture (ambient 10 above its default ceiling 5); the ceiling now rescales with
thaw temperature, while the original invariance assertion is unchanged. These
normalization, effective-count, leaf-evidence, and new controller contracts are
added to the blocking Python matrix rather than merely existing outside CI.

## Remaining research order

The workflow performs long-horizon deep-controller development and confirmation
first, then a development-only Covertype screen: 12/24/32 trees crossed with
depths 5/6/7, against CatBoost depths 6/8/10/12 at up to 2,048 iterations. Audit
is not scored in that screen. Total budgets and times differ and are reported;
this is not an equal-compute comparison. Negative and partial outputs survive
failed jobs through always-uploaded artifacts.

Native generic-reg forest replication follows on seeds 67/73. A scheduled
post-interpolation objective change must be disclosed; effective-count reduction
alone is not double descent. The five-arm epicycle experiment remains **not yet
implemented/executed**. Its primary regularization must be the same generic
machinery used for ordinary unknown problems. Fourier/order/radial/inverse-power
quantities remain diagnostics in primary arms. The domain-specific Fourier
penalty remains a separate planned secondary positive control, not cancelled.

The CatBoost ratchet definition, dataset, thresholds, and assertions are unchanged.
The prior negative physics mechanism monitor remains nonblocking and unchanged.

## Bounded dose refinement

When the coarse direct-gain grid misses the capacitor dose, `calibrate_controller_dose.py` permits at most three development-only refinements using the geometric mean of the peak and integrated-exposure ratios. Predictive scores never choose these gains; the original ranking-selected gain remains frozen. The uncalibrated result is preserved in a separate file, and source/input checksums bind the refinement. Confirmation never recalibrates on its own losses or thermal outcomes. A failed 20% joint match remains explicitly failed.


## Remaining loss-unit confound

The existing uncapped loss-to-charge source is linear in positive loss surprise,
but capacitor energy and dissipated heat are quadratic in charge. The direct
source is linear in surprise **as heat**. Starting from zero charge/current and
without clipping, rescaling every loss by a positive factor a therefore scales
capacitor heat by a^2 but direct heat by a. A constant additive loss offset
cancels in both EMA-surprise signals.

`tests/test_controller_loss_scale.py` characterizes this explicitly: multiplying
all losses by ten gives capacitor heat ratio 100.00000000000017 and direct heat
ratio 10.000000000000007. This is not a violation of the capacitor solver's energy
identity. It is a remaining loss-source normalization/comparator confound. A
fixed development gain need not preserve dose when loss-surprise amplitudes
change on a new task. Do not call the physics fully loss-scale normalized or
reinterpret a failed confirmation dose match after seeing predictive results.
