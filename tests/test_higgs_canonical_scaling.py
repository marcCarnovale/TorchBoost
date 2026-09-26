from experiments.higgs_canonical_scaling import (
    AUDIT_START,
    CANONICAL_TRAIN_STOP,
    LOW_FEATURES,
    RANKING_ROWS,
    SELECTION_ROWS,
    TOTAL_ROWS,
    TRAIN_POOL_STOP,
    smoke,
)


def test_canonical_higgs_boundaries_match_uci_protocol():
    assert TOTAL_ROWS == 11_000_000
    assert CANONICAL_TRAIN_STOP == 10_500_000
    assert TOTAL_ROWS - AUDIT_START == 500_000
    assert TRAIN_POOL_STOP + SELECTION_ROWS + RANKING_ROWS == CANONICAL_TRAIN_STOP
    assert LOW_FEATURES == 21


def test_scaling_smoke_metric_is_finite_and_nontrivial():
    result = smoke(3)
    assert 0 < result["nll"] < 1
    assert .5 < result["auc"] <= 1
