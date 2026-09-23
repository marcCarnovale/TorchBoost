# TorchBoost engineering and research rules

Scientific correctness and reproducibility outrank feature count or favorable results.
The 2024 prototype was AI-assisted; do not hide that provenance or preserve generated errors.
User-authored requirements and corrections outrank historical assistant implementations.

Characterize behavior before changing it. Preserve the legacy source blob unless a separately
specified migration is approved. New APIs must distinguish joint attention ensembles from true
stagewise boosting. Tests exposing a legacy defect do not endorse the defect.

Follow `docs/research-program.md` and maintain `docs/feature-ledger.json`. Planned mechanisms are
not implemented features. Every promotion needs a software owner, equation, tests and ablation.

Keep observations out of plasticity: training -> collector -> tracker -> shared per-run online
scheduler -> plasticity/controller. Preserve split identities; never erase them into one global
success label. Avoid process-global registries, disconnected metric gradients and random fake metrics.

Heat adds capacity for reconsideration; independent cooling removes heat. Temperature is not LR.
A frozen component still contributes predictions. Thawing must not silently reset weights.
Ideal inductors store energy, not resistor heat. Account for charge and energy separately.

Plastic consolidation, yield, restoring-bond breakage and structural deletion are distinct.
Identity must be a defined no-op function or valid square map; never invent an identity for a
nonsquare gate vector. Detach stored anchors. Large cancelling oscillations still count as motion.

Dense masks are not sparse memory savings. Remove optimizer/controller/tracker state along with
physically deleted parameters. Never weaken tests or hide numerical instability to obtain green CI.

Benchmark all seeds and variants with declared splits, objectives, versions, budgets and limitations.
Do not tune on reported test results. Do not claim grokking, double descent, SOTA, convergence or
novelty from anecdotes. Inference and visualization must not mutate physical or plastic state.
