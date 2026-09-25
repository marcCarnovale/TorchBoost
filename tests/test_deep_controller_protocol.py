from dataclasses import asdict

import numpy as np
import pytest

from experiments.deep_controller_protocol import (
    candidate_config,
    controller_type,
    partition,
)
from torchboost.adaptive import training


def test_all_five_splits_are_disjoint_and_new_seeds_change_data():
    x, y, rows, manifest = partition(97)
    indices = np.concatenate(list(rows.values()))
    assert len(np.unique(indices)) == len(x) == len(y) == 17000
    assert [len(rows[k]) for k in rows] == [12000, 1800, 1000, 1000, 1200]
    _, _, _, changed = partition(101)
    assert manifest["audit"]["sha256"] != changed["audit"]["sha256"]


def test_direct_and_capacitor_share_thermal_and_plastic_settings():
    cap = candidate_config("cap", 97, 32)
    direct = candidate_config("direct", 97, 32, 0.25)
    a = asdict(cap.native.physics)
    b = asdict(direct.native.physics)
    for field in ("mode", "charge_gain", "max_injection"):
        a.pop(field)
        b.pop(field)
    assert a == b
    assert cap.native.plasticity == direct.native.plasticity
    assert cap.regularizers == direct.regularizers
    assert cap.native.structure == direct.native.structure


def test_controller_patch_restores_even_on_failure():
    original = training.PhysicalController
    with pytest.raises(RuntimeError), controller_type(True):
        assert training.PhysicalController is not original
        raise RuntimeError("test interruption")
    assert training.PhysicalController is original
