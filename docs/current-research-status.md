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