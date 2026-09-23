# Public Portfolio Engineering Rules

- Scientific correctness and reproducibility outrank feature count or favorable results.
- Characterize current behavior and write invariant/regression tests before changing algorithms.
- Do not call this gradient boosting unless sequential residual fitting is implemented and tested.
- Do not claim grokking, double descent, superiority, interpretability, or robustness from anecdotal runs.
- Benchmarks require fixed seeds, stated data splits, appropriate baselines, and complete results.
- Do not hide instability with clipping, broad exception handling, or weakened assertions.
- Keep the README explicit about experimental status and known limitations.
