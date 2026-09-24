# Unified progressive native training

## What is being replaced

The previous packed progressive loop did not implement actual per-stage row
subsampling: its fraction often only changed a nonbinding batch-size cap.
It did not consistently apply feature masks to older trees during joint updates,
and several advertised penalties were absent from rolling refinement. Its
Newton-output experiment also did not provide an adequate full-objective
acceptance safeguard. Do not interpret those old experiments as clean ablations
of the intended mechanisms. The old APIs remain reproducibility references;
use `torchboost.adaptive.unified_progressive` for this implementation.

## Statistical identity

This is a sum of score-valued native trees. Classification applies its sigmoid
or softmax after summation. It is not a mixture of class probabilities. Each
proposal is fitted to the current full loss gradient and Hessian, including
multiclass off-diagonal terms. Greedy histogram splits enforce row count,
Hessian mass, leaf ridge and split-gain costs. Exact training-loss backtracking
checks the proposed new correction; it is not a guarantee about validation.

Native hard routing and residual ancestor values preserve the proposal's leaf
predictions. Values train first; routing can then stay hard, adjust thresholds,
or become fully oblique. Leaf-only and residual modes start from the same
hard function, so representation and hierarchical penalties can be studied
separately. Parameter counts and resulting optimization paths still differ.

Each new stage gets a fixed sampled row pool. Training minibatches are sampled
from that pool. The feature mask is part of the native tree and survives joint
refinement and checkpoints, including older stages. Hard interaction groups
restrict each summand's inputs: they do not imply additive probabilities after
a nonlinear classification link.

## One adaptive engine

The progressive trainer inherits the native `JointTrainer`; it does not copy
physical or plastic laws into an unrelated optimization loop. Observation,
tracking, online proposals, plasticity, structural actions and actual optimizer
state migration run through the existing owners. Native structural penalties,
feature penalties, diversity, soft monotonicity, schedules and separate dropouts
are available here. Sampled soft monotonicity penalties are not global proofs.

Optimizer moments persist across additions. Old nodes get age-decaying learning
rates; nodes outside the rolling window freeze without losing contributions.
Thermal events can temporarily reopen old nodes. Routing temperature does not
implicitly become the optimizer learning rate or momentum coefficient.
A genuine RLC state can influence the separately selected circuit-momentum
optimizer; compare it with the same RLC controller and fixed momentum.

References are not automatically awarded at birth. Imported anchors require
an explicit comparator flag; ordinary admission requires equivalent passes,
updates and optionally useful local observations. Equivalent passes measure
exposures, not a guarantee that every unique row was seen. Admission is not
consolidation. Yield, damage, recovery and earned reference movement remain
separate native transitions.

Custom hierarchy penalties normalize depth/node budgets, with a learned
allocation option and a penalty on allocation coefficients. Penalties sum over
trees so appending a zero correction cannot dilute existing regularization.
This does not prove the learned budget can never favor a cheap, unhelpful node.

## Checkpoint and cache contracts

CPU stage-boundary continuation preserves last weights, optimizer/controller
state and independent sampler generators. Best selected weights are separate.
Ordered data fingerprints reject resuming on different data. Mid-stage resume,
cross-hardware bitwise equality and arbitrary callbacks are not promised.
Only load trusted research checkpoints: the current format uses Python-backed
PyTorch serialization.

Frozen predictions are a derived bounded cache keyed by live topology, model
state and routing mode. Caches are disabled for dropout, input-dependent heads,
input-gradient constraints and trace-dependent penalties. Actual hard pruning
uses native structural transactions and releases associated ownership.

## Conditional output fitting

The optional refit uses a damped data Hessian as a preconditioner and includes
the regularizer gradient. It is not advertised as the Hessian of every nonlinear
regularizer. Its line search evaluates the full declared objective and reverts
on failure. Accepted readout changes deliberately reset their Adam moments.
The reference solve is dense and limited to 1,024 leaf/output coordinates.

## Evidence and remaining scope

Each experiment records enabled settings, executed updates and examples,
selected checkpoint, intervention counts, failures and source hashes. Online
rewards remain observational: simultaneous learning and local actions can
confound causal attribution. Enabled-but-inactive arms are not evidence that a
mechanism's intended effect was exercised.

The hard proposal builder is binary. The native tree remains the shared
variable-arity/dynamic backend. This does not implement new arbitrary circuit
components, sparse accelerator kernels, structural RL, Kalman/PID control,
evolutionary search, or every analogy discussed historically. Those are not
silently counted as complete. The two synthetic contexts and reused diamonds
are development tasks, not a broad independent frontier benchmark.
