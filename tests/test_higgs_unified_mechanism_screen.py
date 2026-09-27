import json
from pathlib import Path

import numpy as np

from experiments.higgs_unified_mechanism_screen import (
    BATCH,
    DEPTH,
    NTRAIN,
    TREES,
    UPDATES,
    VARIANTS,
    config_for,
)
from torchboost.adaptive.config import StructureConfig
from torchboost.adaptive.unified_progressive import (
    UnifiedConfig,
    UnifiedProgressiveClassifier,
    default_native,
)


def test_shadow_audit_is_locked_outside_selection_ranking():
    protocol=json.loads(
        (Path(__file__).parents[1]/"experiments"/"higgs_shadow_protocol.json").read_text()
    )
    shadow=protocol["shadow_audit"]["range"]
    selection=protocol["development"]["selection"]
    ranking=protocol["development"]["ranking"]
    assert shadow==[9_600_000,10_100_000]
    assert shadow[1]==selection[0]
    assert selection[1]==ranking[0]
    assert protocol["locked_before_next_mechanism_screen"] is True


def test_mechanism_screen_contract_is_audit_blind_and_compute_matched():
    assert NTRAIN==500_000
    assert (TREES,DEPTH,UPDATES,BATCH)==(48,6,32,2048)
    assert TREES*UPDATES*BATCH/NTRAIN>6.
    for name in VARIANTS:
        cfg=config_for(name,17)
        assert cfg.n_trees==TREES
        assert cfg.depth==DEPTH
        assert cfg.native.batch_size==BATCH
        assert cfg.gate_release==("hard" if name=="hist_hard" else "oblique")
        assert cfg.linear_values==name.startswith("affine")


def test_global_stage_rate_refit_is_selection_gated():
    rng=np.random.default_rng(9)
    x=rng.normal(size=(700,6)).astype("float32")
    y=(x[:,0]+.7*x[:,1]-.4*x[:,2]+.2*x[:,3]*x[:,4]>0).astype(int)
    native=default_native()
    native.batch_size=128
    native.structure=StructureConfig(
        dynamic=False,initial_depth=0,max_depth=2,max_nodes=31
    )
    cfg=UnifiedConfig(
        n_trees=4,updates_per_stage=3,depth=2,bins=6,min_samples_leaf=8,
        shrinkage=.3,age_decay=.8,active_window=2,row_subsample=.8,
        feature_subsample=1.,warm_value_updates=1,gate_release="oblique",
        checkpoint_every=2,native=native,random_state=9,
    )
    model=UnifiedProgressiveClassifier(cfg).fit(
        x[:500],y[:500],eval_set=(x[500:600],y[500:600])
    )
    model.refit_stage_rates(
        x[:500],y[:500],eval_set=(x[500:600],y[500:600]),max_iter=5
    )
    record=model.rate_refit_
    assert np.isfinite(record["selection_before"])
    assert np.isfinite(record["selection_after"])
    assert len(record["rates_before"])==len(record["rates_after"])
    if record["accepted"]:
        assert record["selection_after"]<record["selection_before"]
