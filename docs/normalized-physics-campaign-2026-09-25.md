# Normalized-energy physics campaign — 2026-09-25

## Scope and verified starting point

This continuation starts from research head `0c8a229fa5f822868ff4f5fffceaac78e760b611`. Its blocking lint, Python 3.11/3.12, and CatBoost ratchet passed in Actions run 36149233946. The nonblocking physics monitor failed; neither its threshold nor the CatBoost gate is changed. Read `controller-reproduction-2026-09-25.md`: small controller rankings differed across local and Actions software stacks. Do not pool those rankings as equivalent replications.

The new source and driver are experimental files. No production learner, solver, optimizer, or default has changed. The blocking matrix adds 32 new contracts to its prior 87. All 119 passed locally, and the unchanged CatBoost ratchet separately passed. Remote results must be checked on the actual new commit before calling its CI green.

## Common dimensionless observation and energy input

`experiments/normalized_energy_controller.py` uses a causal EW loss-innovation signal. The innovation is standardized using the prior EW innovation scale; warmup records without heating. A zero-variation history has an explicit unit-shock fallback. This is not a significance test on independent observations. Positive affine transformations of an externally supplied loss history preserve this normalized signal, apart from floating-point error.

For clipped positive drive u, the requested energy per accepted controller advance is

    E_requested = rate * dt * H_total * (T_thaw - T_ambient) * u.

The direct arm supplies external heat. The capacitor/RLC arms increase capacitor stored energy, preserving the charge sign when RLC charge is negative. Source clipping and rejected energy are logged; previously stored energy is not erased when passive oscillation exceeds the source charge limit. The existing passive circuit solver, cooling, and energy accounting are retained. Whole-step ledgers include the new source and remain checked.

Tests cover loss-unit transformations, 1/7/63-node topology scaling, temperature-unit/heat-capacity scaling, clipping/venting, negative charge, replay, invalid inputs, and checkpoint continuation. Identical exogenous losses give equal energy input but intentionally different thermal responses. In feedback training the losses themselves can differ, so equal configured rates do not guarantee exactly equal total energy. This is not a claim that every part of the learner or observation allocator is loss-unit invariant.

## Deeper stationary and localized-change driver

`experiments/deep_local_physics.py` builds a random hierarchical local-affine task with separate train/control/selection/ranking/audit streams. The local pilots use 16,000 training rows, 768 control, 1,200 selection, 1,200 ranking, and 2,400 unopened audit rows. Ground-truth changed-region indicators never enter the source, loss, or regularizer.

All arms start from one shared 256-update trained model and reset their optimizers at the same fork. They then perform 1,536 native refinement updates. The stationary sequence is A/A/A; the recurring sequence is A/B/A, where B changes only a quarter of the input regions. Controllers receive no regime identity or oracle boundary. The driver resets checkpoint selection at a phase boundary, but not the optimizer, physical state, tracker, or anchors. The continuation observes the entire 768-row control set at each observation; it does not use the configured 256-row control-sampling shortcut.

The teacher and allowed model depth are six, but the auto-designed pilot model actually has **21 nodes, maximum depth five, and one tree**. This is not a fully grown depth-six model, an overcomplete forest, or a demonstrated selective-local-heating system: source allocation is uniform in this isolate. Structure is fixed during refinement. The ordinary generic residual/leaf/linear penalties and optimization settings are shared. Thermal LR coupling is zero, isolating thermal routing/plastic effects from an LR increase. Yield threshold is specified through cold elastic strain, and thermal softening through ambient-to-thaw units without changing the production plasticity law.

Development ranks the mean NLL over periodic checkpoints and never prepares audit predictions. Confirmation requires a source/config-bound completed development protocol and a fresh seed; it only opens the ranking winner and no-control reference on audit, including the matched trajectory metric and final selected checkpoint.

## Completed local development: activity is not yet value

Twenty continuation fits completed: five arms crossed with two regimes and two release policies. Each policy/regime shares one warm fit. All use source rate 0.5 and data seed 211. Lower ranking-trajectory NLL is better.

| Release / regime | None | Plasticity | Direct heat | Capacitor | RLC |
|---|---:|---:|---:|---:|---:|
| persistent-harm / stationary | .574156339 | .573139861 | .573063870 | .573064618 | .573065857 |
| persistent-harm / recurring | .573690231 | .573215549 | .573240725 | .573236326 | .573233664 |
| stress / stationary | .574156339 | .573695347 | .573753797 | .573749214 | .573747441 |
| stress / recurring | .573690231 | .573540923 | .573643285 | .573627835 | .573628001 |

The small stationary gain from heating over plasticity is development-only and does not support a production promotion. Recurring pilots favor unheated plasticity. Relaxing release from persistent-harm to stress produces vastly more plastic flow, but worsens these ranking metrics.

The causal diagnostic is concrete. In the stationary direct-heat arm, persistent-harm release allows **1 yielding observation and blocks 3,109 releases**, despite peak normalized temperature excursion 1.160 and maximum stress/yield ratio 29.0. Stress release gives **3,017 yielding observations and zero blocked releases** but worse ranking NLL (.573754 vs .573064). These are node-observation counts, not independent nodes, and temperature crossing a threshold alone is not proof that a frozen node thawed. The result identifies a real actuation gate while falsifying the simple claim that removing it necessarily improves learning. Maximum whole-step energy errors are around 1e-17.

The first persistent-harm pilots ran an earlier driver before confirmation binding and the release-policy CLI were added. Their exact executed driver is preserved in the downloadable raw capsule, and its SHA is in the machine-readable record. The stress pilots ran the committed driver. Unchanged no-control scores across these runs are an additional consistency check, not a substitute for source provenance. The Actions campaign reruns all choices using the same final source and pinned environment.

## Bounded next execution and limitations

`experiments/physics_campaign.py` screens source rates 0.1/0.5 and release policies persistent-harm/stress on development seed 211, then freezes the best thermal-family configuration before fresh seeds 223/227. Every run includes no-control and plasticity-only arms. This is a matched-configuration mechanism screen, not an independently optimized best-static-baseline or equal-compute claim. Further tuning of time constants, local allocation, truly deep representations, and simple scheduled controls remains needed before asserting practical circuit superiority.

The workflow runs stationary and recurring campaigns first, then Covertype development seed 61, then native generic-forest seeds 79/89. Partial logs and results are uploaded even on failure. Jobs launched are not results completed.

## Completed prior Actions studies, recovered in this pass

Source `05c4ec6fe7385cbde0e94d409935c0f6f187c320`, workflow 36145015457:

* Covertype: the 12/24/32 by depth 5/6/7 screen selected the requested 24-by-5 candidate, retaining 20 trees, with ranking NLL **.324087501**. CatBoost depth 12 with 2,048 trees ranked at **.152093544**. Audit remains closed. Deeper TorchBoost candidates did not win; this preserves a substantial negative real-data result against a stronger reference. Budgets/times are unequal.
* Native forest seeds 67/73 both retain 100% training classification accuracy. At continuation step 400, hierarchy reduces selection effective count **17.973 -> 8.674** and **20.724 -> 10.169**, with selection NLL **2.049984 -> 1.681627** and **2.211207 -> 1.821789**. Hierarchy wins both independent rankings; winner audit NLL is 1.764794 and 1.900412. Those absolute errors remain poor relative to the early selection minima .404611/.376910. No-complexity-pressure continuation is worse at 2.289136/2.498563 selection NLL.

The native artifacts contain `claim_deep_double_descent=true`. Preserve that raw heuristic flag but do **not** treat it as an established claim: the model is already interpolating at diagnostic step zero, so first interpolation onset inside progressive construction was not localized, and the continuation objective is changed. The justified result is replicated generic post-interpolation recovery and effective-complexity reduction, not a clean fixed-objective onset-localized double-descent demonstration. Lower effective count is not physical tree deletion or measured inference savings.

The source/raw-file/archive hashes and principal scalar summaries are in `results/normalized-physics-2026-09-25.json`. The downloaded prior archives were SHA256-verified against GitHub artifact metadata. The raw local capsule retains full traces, configs, environments, and both executed driver versions.

## Orbital contract remains unchanged

The actual five-arm epicycle warm-start experiment has not been executed. Its primary must use the exact shared ordinary-problem generic machinery, including learned tree coefficients and realized-contribution complexity diagnostics where applicable. Fourier/order, inverse-square, radial, and Kepler-specific penalties remain forbidden in primary. They cannot guide selection through diagnostic scores either. The no-regularization arm must also disable optimizer weight decay. Frozen representation and genuinely cold-start controls remain required; the domain-informed Fourier-energy/order positive control remains a separate **retained secondary** experiment. Do not conflate central-force approximation or generic forest recovery with that orbital experiment.
