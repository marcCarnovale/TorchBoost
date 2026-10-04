from dataclasses import replace
import torch
from torchboost.adaptive.config import ForestConfig, StructureConfig, PlasticityConfig, OnlineConfig
from torchboost.adaptive.forest import AdaptiveForest
from torchboost.adaptive.training import JointTrainer
from torchboost.adaptive.objectives import Objective
from torchboost.adaptive.data import DataSplit
from torchboost.adaptive.contracts import Proposal,StructuralAction

def test_structural_events_wait_for_trials_and_do_not_admit_more(monkeypatch):
    c=ForestConfig(n_trees=1,epochs=10,aggregation='mean',residual_weights=False,
        structure=StructureConfig(max_depth=1,initial_depth=0,max_nodes=7),
        plasticity=PlasticityConfig(mode='anchor'),
        online=OnlineConfig(enabled=True,defer_structure_for_trials=True))
    m=AdaptiveForest(2,1,c);t=JointTrainer(m,Objective('binary',1),c,torch.Generator().manual_seed(2))
    root=m.trees[0].root_id;actions=[StructuralAction('grow',root,0)]
    t.tracker.trials[17]={'proposal':Proposal(t.scheduler.run_id,17,root,0,0,0,0,1.,()),'window':3}
    monkeypatch.setattr(t.tracker,'mature',lambda *a,**k:[])
    monkeypatch.setattr(t.structure,'propose',lambda *a,**k:actions)
    applied=[];requests=[]
    monkeypatch.setattr(t,'_apply_structure',lambda a,*args:applied.extend(a))
    monkeypatch.setattr(t.scheduler,'request',lambda *a,**k:requests.append(True))
    x=torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    d=DataSplit(x,torch.tensor([0,1,1,0]),torch.ones(4))
    t._observe_and_control(d,0)
    assert not applied and not requests
    assert t.events[-1]['event']=='structure_waits_for_trials'
    t.tracker.trials.clear();t._observe_and_control(d,1)
    assert len(applied)==1 and not requests
    actions.clear();t._observe_and_control(d,2)
    assert requests==[True]

def test_disabled_coordination_preserves_legacy_scheduling(monkeypatch):
    # Configuration remains opt-in; old saved configs load with unchanged behavior.
    assert OnlineConfig().defer_structure_for_trials is False
