# TorchBoost architecture corners and interpolation program

TorchBoost should be evaluated as an architecture space, not as one small
progressive-forest configuration.

## Exact endpoints

### Boosting corner

The CatBoost endpoint is an additive ensemble of hard numerical symmetric
(oblivious) trees.  The frozen HIGGS comparator uses 1536 depth-10 trees,
learning rate 0.05 and L2 leaf regularization 20.  The TorchBoost
ObliviousSoftForest representation can import the resulting numerical CatBoost
JSON model exactly in hard mode.  Therefore the CatBoost predictor is a literal
point in the TorchBoost representation space.

Exact representability does not imply that TorchBoost's own optimizer can find
that point.  Native training from this corner is a separate optimization
benchmark.

### Neural corner

A TorchBoost residual node with no children and an affine packet computes

    v + x W.

Consequently one depth-zero affine tree is exactly a dense affine layer.
Composing five such layers with ReLU and dropout and adding the final linear
readout gives the canonical HIGGS MLP exactly.  For 21 inputs, five width-300
hidden layers and one scalar output, the corner has 368,101 trainable
coefficients, exactly matching the baseline MLP.

Growing a zero-residual child refinement preserves the represented function.
Thus the MLP can be embedded first and tree structure added without paying an
initial performance penalty.

## Independent interpolation axes

Every experiment should name its coordinates on these axes.

| Axis | Boosting corner | Neural corner | Hybrid directions |
| --- | --- | --- | --- |
| composition | additive score corrections | five sequential hidden layers | additive + latent/compositional blocks |
| routing | hard | no routing at depth zero | soft, temperature-controlled |
| split geometry | axis aligned | dense affine layer | oblique/grouped oblique |
| topology | symmetric depth-10 trees | depth-zero tree layers | ragged/dynamic refinements |
| local packet | scalar leaf | affine hidden packet | affine residual experts |
| construction | stagewise | all layers present | proposal then release |
| optimization | frozen previous trees | full end-to-end AdamW | rolling + periodic global polish |
| data exposure | boosting sampler | 20 full epochs | proposal sample distinct from SGD sample |
| width/capacity | 1536 trees | width 300 | independently scalable |
| regularization | leaf L2 / bootstrap | dropout / weight decay | hierarchy, tree dropout, plasticity |
| adaptive physics | off | off | optional only after backbone is competitive |

## Required calibration

1. The exact imported CatBoost endpoint must reproduce CatBoost raw predictions.
2. The exact MLP endpoint must reproduce MLP logits.
3. A TorchBoost-native boosting trainer should be calibrated against the
   CatBoost endpoint before hybrid conclusions are drawn.
4. A depth-zero compositional TorchBoost network should train under the same
   optimizer, batches, epochs and initialization as the MLP and reproduce its
   learning curve within numerical tolerance.
5. Hybrid experiments should move one or a small number of named axes away from
   a calibrated endpoint.
6. No new HIGGS architecture may open the locked shadow audit until its
   configuration is fixed from train/selection/ranking.

## High-value trajectories

Boosting -> hybrid:
hard imported symmetric forest -> soften gates -> allow oblique rotations ->
affine residual packets -> periodic global refinement -> compositional state.

Neural -> hybrid:
exact depth-zero MLP -> function-preserving tree growth -> train new residual
branches -> specialize routing -> sparse/dynamic topology -> boosting/Newton
proposal initialization.

The two trajectories should eventually meet.  Their meeting point, rather than
the historical 40-tree progressive configuration, is the main TorchBoost
architecture hypothesis.
