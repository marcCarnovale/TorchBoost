# Adaptive architecture-space training

## Goal

TorchBoost should not choose between a "tree pole" and an "MLP pole".
Those are two calibrated corners inside a much larger architecture space.
The superiority experiment is therefore:

> start from a strong calibrated anchor, generate function-preserving local
> architectural mutations, spend a small matched compute budget on each,
> retain mutations that improve held-out selection/ranking, and continue
> training in the region that earns evidence.

The shadow audit in `experiments/higgs_shadow_protocol.json` remains sealed.
Architecture decisions use only training + selection + ranking.

## Architecture coordinates

A state is described by independently mutable coordinates rather than one
interpolation scalar:

1. **composition**: additive forest / residual composition / latent sequential
   composition / mixtures of these;
2. **capacity**: trees, layers, widths, depth, live nodes, packet rank;
3. **routing hardness**: hard / temperature-soft / annealed / mixed;
4. **routing geometry**: axis aligned / grouped oblique / full oblique;
5. **packet type**: scalar leaf / vector leaf / affine residual / low-rank
   affine / neural packet;
6. **construction**: histogram-Newton / gradient proposal / inherited split /
   learned routing;
7. **aggregation**: fixed additive / learned scalar rate / attention /
   input-dependent mixture;
8. **optimization scope**: newborn only / active window / reopened specialists /
   full end-to-end polish;
9. **exposure**: proposal rows and differentiable refinement rows tracked
   separately, with full-data-equivalent passes per live parameter block;
10. **regularization/control**: sparsity, structural penalties, plasticity,
    thermal/electrical controls, pruning and freeze/reopen policy.

CatBoost and the canonical MLP are named anchor states in this coordinate
system, not opposite ends of a line.

## Required anchor properties

### Neural anchor

The depth-zero compositional network must train through the identical native
MLP floating-point path before structural release.  The endpoint therefore
keeps contiguous `[out,in]` weights and `F.linear` evaluation while depth
zero.  The first growth operation migrates those parameters into the tree root
packet function-preservingly and rebuilds the optimizer.

### Boosting anchor

The imported CatBoost model remains an exact hard symmetric-tree checkpoint.
Relaxations (softening, oblique rotation, affine packets, end-to-end polish)
must begin from that checkpoint, so constructor quality is not confounded with
the value of the added mechanism.

## Architecture mutation protocol

At a checkpoint, create a small portfolio of *named* mutations.  Every mutation
must either preserve the current function at birth or provide an explicit
baseline delta.

Examples from the MLP anchor:

- add zero residual children to one or more affine layers;
- release a subset of those branches to soft oblique routing;
- replace scalar residual packets by affine/low-rank packets;
- add a stagewise Newton proposal as a residual specialist;
- reopen only the highest-utility layer versus full end-to-end polish.

Examples from the CatBoost anchor:

- soften imported splits without changing the initial hard predictor;
- rotate selected axis-aligned splits into grouped/full oblique gates;
- add zero affine residual packets to selected trees;
- release a bounded active window versus global differentiable polish;
- add latent/compositional blocks above the additive predictor.

## Evidence-driven allocation

A mutation receives a fixed pilot budget measured in:

- optimizer updates;
- examples seen;
- full-data-equivalent passes;
- wall time;
- trainable parameters;
- peak memory.

Selection NLL is the primary admission signal; ranking NLL/AUC is the
architecture comparison signal.  A mutation is not retained merely because it
lowers training loss.

For each parent state, evaluate:

- parent continuation for the same compute;
- each mutation with identical data/exposure budget;
- optionally a combined mutation only after its individual mechanisms show
  positive evidence.

The controller maintains an archive of non-dominated states rather than a
single scalar ranking.  Quality, compute, memory and inference cost are tracked
separately.  For the HIGGS superiority claim, predictive quality is primary,
but efficiency gains are retained rather than discarded.

## Search dynamics

Use successive-halving / bandit-style resource allocation over architecture
mutations:

1. generate local candidates;
2. pilot each candidate under matched exposure;
3. reject clear regressions;
4. allocate more compute to promising candidates;
5. periodically include the unchanged parent as a control;
6. checkpoint every admitted architecture transition;
7. run occasional full end-to-end polish so rolling/frozen approximations do
   not silently define the model class.

This is architecture adaptation *during training*, not offline grid search over
a static family.

## First superiority campaign

After exact MLP training equivalence is green:

1. train and save the 500k neural anchor;
2. branch the same checkpoint into:
   - MLP continuation control,
   - zero-residual growth + newborn-only refinement,
   - zero-residual growth + full end-to-end refinement,
   - zero-residual growth + oblique release;
3. match examples-seen and optimizer-pass budgets;
4. select only from selection + ranking;
5. admit the best non-regressing mutation;
6. repeat from the admitted state with the next coordinate mutations.

In parallel, run the imported CatBoost anchor through soft/oblique/affine
relaxations.  The purpose is to discover whether successful trajectories from
both anchors move toward a common region of architecture space.

## Success criterion

The meaningful result is not "TorchBoost can imitate CatBoost and an MLP."
It is:

> a single adaptive training system contains both strong baseline corners and
> can use held-out evidence to move into a mixed architecture that beats the
> stronger calibrated anchor under a controlled compute/exposure protocol.

Only after a final architecture/configuration is frozen is the fresh shadow
audit opened.
