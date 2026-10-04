# External transfer: mechanism decomposition

Status: secondary analysis plan written after the first transfer cells became
available. It is **not** a replacement for the frozen primary benchmark
protocol and must be labeled post hoc in any report.

The external study contains two scientifically distinct questions:

1. **Residual-tree expansion effect**
   [
   Delta_{mathrm{adapter}} =
   mathrm{NLL}(	ext{fixed adapter})-mathrm{NLL}(	ext{MLP anchor}).
   ]

   This asks whether function-preserving tree expansion adds useful predictive
   capacity to the inherited neural model.

2. **Held-out architecture-scale effect**
   [
   Delta_{mathrm{scale}} =
   mathrm{NLL}(	ext{learned-scale adapter})-
   mathrm{NLL}(	ext{fixed adapter}).
   ]

   This asks whether held-out learning of layerwise residual scales improves
   over using the frozen perturbative scale.

These effects must not be conflated. A dataset may support useful residual-tree
expansion while preferring the fixed scale.

For final transfer reporting, compute both quantities per dataset by averaging
the five seed-level paired differences within dataset, then summarize across
datasets with:

- mean and median dataset effect;
- dataset win/tie/loss count;
- dataset-cluster bootstrap interval;
- all per-dataset means.

The originally frozen learned-adapter comparisons against MLP, fixed adapter,
CatBoost, XGBoost, and LightGBM remain the primary reported benchmark outputs.
This decomposition is a transparent secondary interpretation prompted by the
emerging transfer behavior, not a newly preregistered success criterion.
