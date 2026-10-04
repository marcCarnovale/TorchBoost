# Overcomplete forest compression: mechanism-isolation result

This experiment isolates one hypothesis behind the planned TorchBoost
overparameterized forest. It is **not** the final TorchBoost forest and is **not
yet evidence of deep double descent**.

## Question

Can a redundant interpolating collection of trees move, under only generic
post-interpolation regularization, toward a lower-complexity decomposition while
preserving interpolation and improving held-out performance?

The experiment in `experiments/overcomplete_forest_mechanism.py` builds a pool
of 128 ordinary CART trees: 64 shallow/intermediate trees and 64 fully grown
interpolating trees. Scalar front coefficients are learned jointly. Tree dropout
is used before the interpolation checkpoint and switched off when the
post-interpolation simplification phase begins.

The task is a noisy generic piecewise tabular rule. No task-specific structure
is supplied to the regularizer.

The three post-interpolation conditions are:

- **none** — predictive loss plus tiny ordinary coefficient decay;
- **L2** — additional coefficient ridge;
- **generic hierarchy** — coefficient ridge plus realized-contribution
  shrinkage and a generic structural cost proportional to log leaf count.

There is no Fourier, orbital, inverse-square, target-rule, or other
domain-specific regularization.

## Three-seed development result

At the interpolation checkpoint, all three variants are identical because their
post-interpolation regularizers have not yet been activated. Across seeds
77/78/79:

- training classification accuracy: **100%**;
- mean audit NLL: **6.5837**;
- mean participation-ratio effective tree count: **100.71**.

After continued optimization:

| post-interpolation pressure | mean train NLL | mean audit NLL | mean audit accuracy | mean effective tree count | shallow contribution mass |
|---|---:|---:|---:|---:|---:|
| none | 2.59e-8 | 1.8618 | 82.87% | 65.20 | 0.28% |
| coefficient L2 | 2.38e-5 | 1.1767 | 83.08% | 72.03 | 3.57% |
| generic hierarchy | 0.1198 | **0.4155** | **85.33%** | **50.46** | **30.34%** |

The generic hierarchy condition retained **100% training classification
accuracy on every seed**, despite allowing the training NLL to rise away from
the degenerate huge-margin interpolating solution. Its audit NLL fell by about
93.7% relative to the interpolation checkpoint and its effective tree count
fell by about half.

This is exactly the qualitative post-interpolation selection mechanism we
wanted to see: predictive fit has already saturated in classification error, so
generic regularization can reward a less redundant structural decomposition.

## What this does and does not establish

It **does** show, in a controlled surrogate, that learned front coefficients and
ordinary structural penalties can exploit redundant forest solutions after
interpolation. L2 alone is much weaker and does not reduce the participation
count as effectively.

It **does not** yet establish deep double descent. The learning curve does not
show the required clean first descent -> interpolation-associated peak -> second
descent. Nor does the experiment allow the underlying trees themselves to
deform; only their front coefficients move.

The next TorchBoost-native experiment should therefore:

1. use actual trainable power trees rather than a frozen CART pool;
2. deliberately maintain redundant solution paths with bagging/feature/tree
   dropout before interpolation;
3. make front coefficients trainable in the native progressive forest;
4. activate only generic parameter/hierarchy/contribution pressure after
   interpolation;
5. track train loss, selection/audit loss, effective tree count, contribution
   entropy, structural depth, and tree correlations continuously;
6. test whether structural plasticity lets the trees themselves deform toward
   the lower-complexity solution rather than merely reweighting a fixed basis.

For the orbital/epicycle study the same generic machinery should be used
unchanged. Fourier/epicycle order and inverse-power exponent remain diagnostics
only; the domain-informed Fourier penalty remains a secondary positive control.
