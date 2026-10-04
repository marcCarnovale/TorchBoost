import json
from pathlib import Path


def test_shadow_protocol_range_remains_locked():
    protocol = json.loads(Path("experiments/higgs_shadow_protocol.json").read_text())
    assert protocol["shadow_audit"]["range"] == [9_600_000, 10_100_000]
    assert protocol["shadow_audit"]["rows"] == 500_000
    assert "Do not evaluate" in protocol["shadow_audit"]["rule"]


def test_superseded_freeze_cannot_authorize_shadow_open():
    freeze = json.loads(Path("research/higgs_adapter_shadow_freeze.json").read_text())
    assert freeze["status"] == "superseded_do_not_execute"
    assert freeze["shadow_open_authorized"] is False
    assert freeze["shadow_opened"] is False
