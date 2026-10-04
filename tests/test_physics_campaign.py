import json
from pathlib import Path

from experiments import physics_campaign


def test_campaign_freezes_selection_before_fresh_confirmation(tmp_path, monkeypatch):
    calls = []

    def fake_run(seed, depth, ntrain, warm_updates, phase_updates, rate, stiffness,
                 strain, regime, out, **kwargs):
        calls.append((seed, rate, kwargs["release_policy"]))
        if kwargs.get("confirmation"):
            frozen = json.loads((tmp_path / "frozen-selection.json").read_text())
            assert frozen["status"] == "selected_before_confirmation"
            assert seed in (223, 227)
            assert Path(kwargs["protocol"]).exists()
            assert rate == 0.1
            assert kwargs["release_policy"] == "stress"
        score = rate + (0.0 if kwargs["release_policy"] == "stress" else 1.0)
        result = {"arms": {name: {"ranking_trajectory_nll": score}
                           for name in ("direct", "capacitor", "rlc")}}
        Path(out).write_text(json.dumps(result))
        return result

    monkeypatch.setattr(physics_campaign, "run", fake_run)
    result = physics_campaign.campaign(tmp_path, "stationary", 3, 1000, 8, 16)
    assert result["status"] == "completed"
    assert len(calls) == 6
    assert [c[0] for c in calls] == [211, 211, 211, 211, 223, 227]
