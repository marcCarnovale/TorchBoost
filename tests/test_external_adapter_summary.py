import pytest

from experiments.summarize_external_adapter_benchmark import summarize


def record(dataset, seed, learned_nll, baseline_nll):
    def model(nll, auc):
        return {"ranking": {"nll": nll, "auc": auc}}

    return {
        "protocol_version": "external-adapter-v2-frozen",
        "status": "completed",
        "dataset": dataset,
        "seed": seed,
        "source_sha": "deadbeef",
        "mlp": model(baseline_nll, 0.70),
        "fixed_adapter": model(baseline_nll, 0.70),
        "learned_adapter": model(learned_nll, 0.71),
        "catboost": model(baseline_nll, 0.70),
        "xgboost": model(baseline_nll, 0.70),
        "lightgbm": model(baseline_nll, 0.70),
    }


def test_cross_dataset_inference_clusters_seed_replicates():
    rows = [
        record("dataset-a", 1, 0.4, 0.5),
        record("dataset-a", 2, 0.4, 0.5),
        record("dataset-a", 3, 0.4, 0.5),
        record("dataset-b", 1, 0.8, 0.5),
    ]
    summary = summarize(rows)
    comparison = summary["learned_adapter_paired_comparisons"]["mlp"]
    clustered = comparison["dataset_clustered_nll_delta"]
    cells = comparison["cell_level_seed_variation"]["nll_delta"]

    assert summary["inferential_unit"] == "dataset"
    assert clustered["n"] == 2
    assert cells["n"] == 4
    assert clustered["mean"] == pytest.approx(0.1)
    assert cells["mean"] == pytest.approx(0.0)


def test_dataset_win_counts_are_not_seed_win_counts():
    rows = [
        record("dataset-a", 1, 0.4, 0.5),
        record("dataset-a", 2, 0.4, 0.5),
        record("dataset-a", 3, 0.4, 0.5),
        record("dataset-b", 1, 0.8, 0.5),
    ]
    comparison = summarize(rows)["learned_adapter_paired_comparisons"]["mlp"]

    assert comparison["dataset_nll_wins_ties_losses"] == {
        "wins": 1,
        "ties": 0,
        "losses": 1,
    }
    assert comparison["cell_level_seed_variation"]["nll_wins_ties_losses"] == {
        "wins": 3,
        "ties": 0,
        "losses": 1,
    }
