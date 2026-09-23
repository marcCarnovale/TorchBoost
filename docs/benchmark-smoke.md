# Binary smoke benchmark — all variants

This records a development smoke test, not a tuned state-of-the-art comparison.

Protocol: two small binary datasets, seeds 0/1/2, approximately 50% training, 15% controller,
15% selection and 20% final test. Each method sees the same training and selection rows.
Only the capacitor variant uses controller labels. No method is tuned on these test scores.

All variants have a 32-stage budget and depth 3. TorchBoost uses 12 inner epochs per stage;
XGBoost uses histogram trees. This is neither equal compute nor equal parameter count.
CART initialization is disclosed; the random-initialized variant is separate.

Values below are mean ± sample standard deviation across the three recorded splits.

| Dataset | Variant | AUC ↑ | NLL ↓ | Fit seconds ↓ |
|---|---|---:|---:|---:|
| breast_cancer | torchboost_capacitor_cart | 0.9859 ± 0.0115 | 0.1229 ± 0.0190 | 0.9367 ± 0.0100 |
| breast_cancer | torchboost_first_order_cart | 0.9883 ± 0.0099 | 0.2514 ± 0.0130 | 0.8153 ± 0.0658 |
| breast_cancer | torchboost_newton_cart | 0.9850 ± 0.0125 | 0.1384 ± 0.0243 | 1.0265 ± 0.3990 |
| breast_cancer | torchboost_newton_random | 0.9913 ± 0.0076 | 0.1062 ± 0.0263 | 0.7008 ± 0.0110 |
| breast_cancer | xgboost_hist | 0.9820 ± 0.0082 | 0.1476 ± 0.0229 | 0.0228 ± 0.0033 |
| synthetic_imbalanced | torchboost_capacitor_cart | 0.9544 ± 0.0070 | 0.2541 ± 0.0203 | 1.7013 ± 0.1725 |
| synthetic_imbalanced | torchboost_first_order_cart | 0.9320 ± 0.0185 | 0.3727 ± 0.0212 | 1.6387 ± 0.4740 |
| synthetic_imbalanced | torchboost_newton_cart | 0.9526 ± 0.0086 | 0.2585 ± 0.0264 | 1.5263 ± 0.2760 |
| synthetic_imbalanced | torchboost_newton_random | 0.9579 ± 0.0045 | 0.2388 ± 0.0131 | 1.3311 ± 0.0274 |
| synthetic_imbalanced | xgboost_hist | 0.9343 ± 0.0134 | 0.3010 ± 0.0300 | 0.0168 ± 0.0030 |

## Interpretation and limitations

The random-initialized Newton variant has the highest mean AUC in these two recorded comparisons.
This does not establish broad superiority, a recommended default, or statistical significance.
The datasets are small; three split seeds do not constitute broad independent replications.
XGBoost is substantially faster. Timings are single-process CPU measurements and include
TorchBoost metrics collection; first-run setup affects timing. No GPU claim is made.

The capacitor variant uses a different cooling schedule as well as corrective heat. Total
corrective heating is very small in these runs. Its results are therefore not evidence that
the heating mechanism causes an improvement. A matched cooling-only and gain-zero ablation
is required before that interpretation. The random-initialized baseline also outperforms it
on the reported mean AUC values.

Raw records include balanced accuracy, Brier score, 10-bin ECE, inference timing, model state
size, selected stage count and mean soft/hard probability discrepancy. A hard export is tested
against explicit hard inference, not claimed identical to soft inference. Peak memory was not
measured: tensor state bytes must not be mislabeled as peak training memory.

[Raw results and environment](../benchmarks/results/binary_smoke.json) retain all 30 runs,
full estimator configurations, dataset fingerprints, split-index digests and descriptive summaries.
The exact indices are reconstructed by the versioned split function and checked against the digests.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m benchmarks.run_binary --seeds 0 1 2
```

Current run: PyTorch 2.10.0+cpu, sklearn 1.8.0, XGBoost 3.1.3.
Do not select new hyperparameters using this published test table; use development data and a fresh final holdout.
