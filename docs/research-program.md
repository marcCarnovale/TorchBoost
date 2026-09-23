# RFC 001 — Adaptive differentiable forests

Status: foundation implemented; larger research program proposed.

## Mandate and provenance

Make TorchBoost a flagship research and engineering project: powerful, interpretable at the
mechanism level, reproducible, and explicit about what the evidence does and does not show.
The original was an AI-assisted overnight prototype in 2024. The 2026 rebuild preserves the
ideas while replacing unverified generated implementations with mathematical specifications,
characterization tests, and controlled experiments. AI assistance is not hidden.

Sources: the maintainer's September 2026 handoff and design requirements from the historical
[conversation](https://chatgpt.com/share/6ab32650-81d8-83e8-a167-9261f34847e2).
The handoff reports full transcript recovery; this implementation does not claim to have
independently re-decoded every message. User corrections are authoritative over old generated code.

The thesis is an adaptive differentiable forest with four independently testable systems:
output specialization, structural growth/pruning, local physical control, and learned plastic
anchors. Stagewise boosting is one statistical backbone, not a replacement for the retained
jointly trained attention ensemble. Neither backbone is inherently superior.

## Implemented vertical slice

`StagewiseBinaryClassifier` fits binary logistic loss with one new soft oblique tree per stage.
For accepted ensemble scores F, p=sigmoid(F), g=p-y, and h=p(1-p), a candidate f minimizes

    sum_i w_i [g_i f(x_i) + 0.5 h_i f(x_i)^2] / sum_i w_i
      + 0.5 lambda_leaf ||v||_2^2 + 0.5 lambda_gate mean(W^2).

Old stages, including temperatures, remain frozen and retain their contributions. A training-only
backtracking search accepts eta*f only when the exact logistic loss decreases. Selection data
choose an evaluated stage prefix; the intercept-only model is also eligible. Analytic Hessians
are tested against autograd. A declared numerical curvature floor is logged.

Soft leaves overlap. With routing matrix R and leaf vector v, the conditional Newton solve is

    [R^T diag(w*h) R / sum(w) + lambda_leaf I] v = -R^T(w*g)/sum(w).

It is a coupled solve, not independent hard-leaf Newton updates. `init='cart'` is a disclosed
hybrid warm start; `init='random'` is an ablation. `curvature='first_order'` explicitly substitutes
unit surrogate curvature and is not advertised as a Hessian.

This initial solver is dense and depth-limited. It is not the sparse-growth implementation.
Inputs and cached scores stay on CPU; minibatches move to the configured device. GPU behavior
has not been performance-validated. Checkpoints are for inference/inspection, not exact
mid-optimizer resume.

## Boundaries and ownership

    training loop -> SplitMetricsCollector -> PerformanceTracker
                                               -> OnlineScheduler (planned)
                                                    -> PlasticityModule (planned)
    controller-only validation -> CapacitorController -> candidate gate temperatures

`SplitMetricsCollector` observes detached routing traces and optimizer updates regardless of
whether deformation occurs. Statistics preserve stage/node identities. `PerformanceTracker`
maintains bounded per-node histories. Neither component owns plasticity or fabricates success
labels. The later online scheduler will be shared within a run, not a process-global singleton.
It may use observations from all splits to choose different settings per split.

The minimal controller uses uniform node resistances. Data-guided allocation, delayed causal
attribution, and CPU-side asynchronous proposals are not implemented yet. No background worker
may mutate live GPU state: versioned proposals must be applied by training at safe boundaries.

## Required semantic distinctions

- Temperature controls routing softness, not optimizer learning rate. Higher temperature can
  restore gradients in saturated gates, but does not universally increase gradient magnitude.
- Freeze preserves predictions; it does not deactivate a node. Thaw does not reinitialize it.
- Heating enables structural reconsideration; pruning independently decides existence.
- Gradient accumulation normally reuses `.grad`; it does not require K gradient copies.
  Momentum needs a parameter-sized history. Per-tree coefficients do not eliminate that storage.
- Routing entropy is uncertainty, not useful information. The collector separately measures
  weighted branch/label mutual information conditional on reaching a node. This is descriptive,
  not causal importance, and is not sufficient evidence of deformation success.
- Inference, visualization, and regularization queries must not advance control/plasticity state.

## Capacitor controller: implemented experimental baseline

A positive regression above a smoothed reference injects bounded nonnegative charge. First
observation only initializes the reference. Improvement stops injection relative to that
reference; it does not erase residual charge. Let G=sum_j 1/R_j, C>0, dt>0. The update is

    V=Q/C; I_j=V/R_j; P_j=V*I_j
    Q_next=Q*exp(-G*dt/C)
    E_released=(Q^2-Q_next^2)/(2*C)
    heat_j=E_released*(1/R_j)/G.

Thus the discharged electrical energy becomes resistor heat exactly, with no duplicated budget.
Logged current/power are initial instantaneous values; integrated heat is generally not P*dt.
Heat raises T by heat/c. Cooling is independent: exact Newton cooling for the linear variant,
or an implicit passive update of c*dT/dt=-k*(T^4-T_ambient^4) for the radiative variant.
The combined update is operator splitting in dimensionless model units, not an exact simultaneous
ODE solution. State buffers serialize. Tests check source/discharge/heat/cooling accounting.

Only the current candidate is controlled. Accepted stages never reheat in this baseline. A
future joint fine-tuning phase must be explicitly named rather than silently breaking boosting.
The small benchmark does not isolate heating from changed cooling schedules: add a cooling-only
matched control and gain-zero ablation before attributing any benefit to corrective heat.

## Plasticity: priority research specification, not yet an implementation

Separate **consolidation**, **yielding**, **damage/breakage**, and **structural deletion**.
A node should not remain forever quadratically tied to its original no-op configuration.

For node parameters theta, detached anchor A, elastic stiffness K, and bond integrity d in [0,1],
one candidate constitutive design is

    U_elastic = 0.5*d*||theta-A||_K^2
    stress = d*K*(theta-A)
    yield_function = ||stress|| - Y_eff
    plastic_rate = mobility * positive(yield_function/Y_scale)^m
    A_next = A + dt*plastic_rate*stress/(||stress||+epsilon).

Work hardening increases Y with accumulated plastic strain. Optional temperature softening
lowers effective Y; a zero softening coefficient disables it. Recovery decays hardening after
inactivity using elapsed increments, not repeated subtraction of the entire idle duration.
A separate damage law can reduce d under sustained supported pressure; at breakage the old
pullback releases. Re-anchoring, healing, and terminal locking require explicit policies.
Breaking an anchor is not deleting the node.

Evidence-based consolidation is another process:

    e_next = decay*e + evidence(persistence, utility, small total update path)
    z = sigmoid(sharpness*(e-threshold))
    A_next = (1-rate*z)*A + rate*z*stop_gradient(theta_candidate).

Large cancelling oscillations count against evidence. A detached stored candidate is essential:
using the current differentiable theta as its own reference degenerates into weakened weight
decay rather than memory. Stable bad configurations must not consolidate merely because they
are frozen or have no traffic. These are proposed learning laws inspired by mechanics, not a
claim that parameter space is an actual material or that empirical improvement is guaranteed.

A literal identity matrix applies only to square transformations. An oblique routing vector is
not square. Structural identity must instead be a real no-op/bypass function, a residual branch
with zero contribution, or a separately defined square feature transform. Specify it before
implementing Frobenius-to-identity pruning.

## Growth, output specialization, and inductive control

Dynamic topology must allocate and delete modules and their optimizer/controller/tracker state.
Start with binary topology; retain trinary/n-ary support as a requirement, not a rejected idea.
Separate structural gates from routing temperature. Support explore/evaluate/consolidate/reopen,
rolling subtree and copse schedules, depth-normalized penalties, within-level split competition,
and composable named/CustomPruning policies. Test actual released memory and export fidelity.

Later output heads support ordinary multiclass softmax, one-v-rest auxiliaries, multitarget
regression and learned per-head tree attention without hard-coding one tree per class. Compare
against the unchanged joint baseline and stagewise objective abstractions.

An ideal inductor stores energy; it does not dissipate resistor heat. Begin with local gradient
EMA as an independent optimizer experiment. An optional delayed mapping

    E_t=0.5*L*||m_{t-1}||^2
    beta_t=beta_min+(beta_max-beta_min)*sigmoid(a+b*log1p(E_t))
    m_t=beta_t*m_{t-1}+(1-beta_t)*g_t

is a designed feedback law, not a consequence of Ohm's law. Bound beta away from one and reset
or decay stale momentum under reversals. Compare with fixed EMA, AdamW, and a derived dissipative
RLC system; do not apply a second momentum filter unnoticed. Rare sparse random sparks remain
an optional budgeted perturbation experiment, not an assumed diversity benefit.

## Acceptance gates

1. Preserve legacy behavior and expose defects with characterization tests.
2. Establish the statistical baseline and reproducible multi-seed comparison (current slice).
3. Validate metrics and capacitor accounting, then matched controller ablations.
4. Implement detached plastic anchors, yielding and breakage with controlled perturbation tests.
5. Add dynamic topology plus optimizer-state migration and memory profiling.
6. Add shared online experimental scheduling after split-specific outcomes are credible.
7. Extend task heads, broad tabular benchmarks, robustness, calibration, adaptation and retention.

Every claim needs an owner, equation, test, and ablation. Use distinct training, controller,
selection, and final-test roles. Repeated adaptive development requires fresh final test data.
Report all seeds, failures, runtime, peak memory, soft/hard fidelity, and structural churn.
Benchmark against XGBoost, LightGBM, CatBoost, histogram boosting, MLPs and differentiable-tree
prior art. No broad superiority, theoretical convergence, or novelty claim is established yet.

## Primary prior art to compare, not ignore

- [XGBoost objective derivation](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).
- [NODE](https://arxiv.org/abs/1909.06312): differentiable oblivious ensembles.
- [GRANDE](https://arxiv.org/abs/2309.17130): gradient-based decision-tree ensembles.
- [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796) and
  [Synaptic Intelligence](https://arxiv.org/abs/1703.04200): learned-weight preservation.
- [RigL](https://arxiv.org/abs/1911.11134): dynamic sparse training.

These references constrain novelty claims. The proposed combination needs separate literature
review and measured advantages; physical naming alone is not a contribution.
