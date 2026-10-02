from copy import deepcopy
import torch
from experiments.scaling_followups import make_resumable
from test_scaling_controls import small,cfg
from torchboost.adaptive.forest import AdaptiveForest
from torchboost.adaptive.scaling import StepBudgetTrainer
from torchboost.adaptive.objectives import Objective
from torchboost.adaptive.config import ScheduleConfig

def test_positive_tail_resume_matches_uninterrupted_original_schedule():
    pre,splits=small();original=cfg('rlc');original.schedules={'learning_rate':ScheduleConfig(kind='cosine',low=.01,high=.002)};long=deepcopy(original);long.epochs=8
    a=StepBudgetTrainer(AdaptiveForest(4,1,long),Objective('binary',1),long,torch.Generator().manual_seed(3),updates_per_block=2)
    a.schedule.config=original;a.fit_steps(*splits)
    b=StepBudgetTrainer(AdaptiveForest(4,1,original),Objective('binary',1),original,torch.Generator().manual_seed(3),updates_per_block=2)
    b.fit_steps(*splits);c,_=make_resumable({'config':original.to_dict(),'preprocessor':pre.state_dict(),'trainer':b.state_dict()},8);c.fit_steps(*splits)
    assert a.optimizer_steps==c.optimizer_steps==16
    for key,val in a.model.state_dict().items():assert torch.equal(val,c.model.state_dict()[key]),key
    assert [r['selection_loss'] for r in a.history]==[r['selection_loss'] for r in c.history]

    assert all(abs(r["learning_rate"]-.002)<1e-12 for r in c.history[3:])
    assert abs(c.history[0]["learning_rate"]-.01)<1e-12
