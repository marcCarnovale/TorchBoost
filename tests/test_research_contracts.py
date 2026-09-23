import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_blob_is_preserved_exactly():
    content = (ROOT / "torchboost/legacy.py").read_bytes()
    blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    assert blob == "95e9af7ae59100f8113c2599cb4d953256b19125"


def test_feature_ledger_has_owners_and_preserves_the_research_program():
    ledger = json.loads((ROOT / "docs/feature-ledger.json").read_text())
    features = ledger["features"]
    by_id = {row["id"]: row for row in features}
    assert len(by_id) == len(features)
    for row in features:
        assert row["status"] in ledger["status_vocabulary"]
        assert row["owner"] and row["description"] and row["acceptance"]
        for path in row["tests"]:
            assert (ROOT/path).is_file()
    for key in ("plastic-breakage", "plastic-yield", "plastic-consolidation", "dynamic-topology",
                "specialized-heads", "online-scheduler", "trinary-routing", "inductive-momentum"):
        assert by_id[key]["status"] == "planned"


def test_recorded_benchmark_contains_all_variants_without_nonfinite_metrics():
    data = json.loads((ROOT / "benchmarks/results/binary_smoke.json").read_text())
    assert len(data["records"]) == 30
    keys = {(r["dataset"], r["seed"], r["model"]) for r in data["records"]}
    assert len(keys) == 30
    for row in data["records"]:
        assert np.isfinite([row[k] for k in ("auc", "log_loss", "accuracy", "brier", "ece")]).all()
        assert row["peak_memory_bytes"] is None
    assert len(data["summary"]) == 10
