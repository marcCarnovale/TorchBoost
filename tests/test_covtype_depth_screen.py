import json

import numpy as np

from experiments import covtype_depth_screen as screen


def test_depth_screen_smoke_runs_real_models_without_scoring_audit(tmp_path, monkeypatch):
    original_score = screen.score
    lengths = []
    # Mark the audit rows while retaining the exact native smoke data shapes.
    original_make = screen.make_classification
    audit = None

    def marked_data(**kwargs):
        nonlocal audit
        x, y = original_make(**kwargs)
        audit = np.asarray(x[2100:], dtype="float32").copy()
        return x, y

    def guarded_score(model, split):
        assert audit is not None
        assert not np.array_equal(split[0], audit)
        lengths.append(len(split[0]))
        return original_score(model, split)

    monkeypatch.setattr(screen, "make_classification", marked_data)
    monkeypatch.setattr(screen, "score", guarded_score)
    path = tmp_path / "screen.json"
    result = screen.run(seed=59, out=path, smoke=True)
    assert result["status"] == "completed"
    assert result["audit_evaluated"] is False
    assert len(result["torchboost_candidates"]) == len(result["catboost_candidates"]) == 2
    assert len(lengths) == 8
    assert json.loads(path.read_text())["status"] == "completed"
