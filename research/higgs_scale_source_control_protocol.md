# HIGGS scale-source causal control protocol

Status: predeclared before interpreting this control study.

## Question

The replicated HIGGS study established that held-out learning of five positive
layerwise residual scales improves over an otherwise matched fixed-scale
adapter. This control asks whether that benefit is specifically associated with
held-out architecture allocation, rather than merely granting the adapter five
additional trainable supervised parameters.

## Frozen arms

All arms share the same canonical MLP anchor, one zero-at-birth residual-tree
refinement in each hidden layer, frozen inherited backbone/head, residual
optimizer, initialization, minibatch order, adapter epochs, and ranking set.

1. **fixed**: residual scales remain fixed at `sigmoid(-2)`.
2. **train-scale**: the same five scale parameters are trainable and updated
   from TRAIN on the same post-warmup cadence as the held-out arm.
3. **heldout-scale**: the same five scale parameters are updated from SELECTION
   on that matched cadence.

Residual routing/packet parameters are trained from TRAIN in all three arms.
The scale optimizer, learning rate, clipping rule, warmup, batch size, update
count, and update ordering are matched between train-scale and heldout-scale.

At every scheduled scale update, the residual step occurs first. Each
trainable-scale arm then performs a separate scale-only forward/backward pass.
The train-scale arm draws from a fixed 100k-row subset of TRAIN; the
heldout-scale arm draws from the 100k-row ARCHITECTURE-SELECTION subset.
The scale-only source pools therefore have the same size. The two arms also
receive matched scale-only example exposure and differ only in whether the
scale data came from TRAIN or held-out development data. Residual training
continues to use all 500k TRAIN rows.

## Primary comparison

Primary quantity:

`heldout-scale ranking NLL - train-scale ranking NLL`.

Negative is favorable to held-out architecture allocation. Ranking AUC is a
secondary directional measure.

The fixed-scale and MLP-anchor comparisons remain useful context but do not
answer this causal question by themselves.

## Data discipline

The existing 200k-row development SELECTION block is deterministically split
before this corrected control is interpreted:

- first 100k rows: **ARCHITECTURE-SELECTION**, used only for heldout-scale
  gradient updates;
- second 100k rows: **CHECKPOINT-SELECTION**, used to choose the retained
  checkpoint for all three arms.

TRAIN supplies residual updates in every arm and scale updates only in the
train-scale arm. This prevents the heldout-scale optimizer from consuming the
same examples used to select its checkpoint.

For trainable-scale arms, checkpoints before the first actual scale update are
ineligible. This prevents a pre-warmup checkpoint from masquerading as a
train-scale result.

RANKING is evaluation only. The fresh HIGGS shadow audit at rows
`[9,600,000, 10,100,000)` remains unopened and must not be accessed by this
study.

## Interpretation

A heldout-scale win over train-scale supports the narrower claim that allocating
residual capacity using held-out predictive evidence adds value beyond simply
making the five scale coefficients trainable.

A loss or tie does not invalidate the replicated learned-vs-fixed result; it
would instead weaken the stronger causal interpretation of why that result
occurs.
