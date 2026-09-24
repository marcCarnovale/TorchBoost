from copy import deepcopy
import torch
from test_scaling_controls import cfg,small
from torchboost.adaptive.operating_scales import OperatingScales
from torchboost.adaptive.scaling import StepBudgetTrainer
from torchboost.adaptive.forest import AdaptiveForest
from torchboost.adaptive.objectives import Objective

def test_response_change_preserves_zero_source_learning_exactly():
    pre,splits=small();models=[];curves=[]
    for capacity in [.1,.003]:
        config=cfg('none');config.physics=OperatingScales(total_heat_capacity=capacity,charge_gain=0).physics(7,mode='capacitor')
        tr=StepBudgetTrainer(AdaptiveForest(4,1,config),Objective('binary',1),config,torch.Generator().manual_seed(3),updates_per_block=3)
        tr.fit_steps(*splits);models.append(deepcopy(tr.model.state_dict()));curves.append([h['selection_loss'] for h in tr.history]);tr.close()
    assert curves[0]==curves[1]
    for key in models[0]:assert torch.equal(models[0][key],models[1][key])
