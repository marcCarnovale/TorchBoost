from copy import deepcopy
import numpy as np
import pytest
import torch
from torchboost.adaptive.single_tree import SingleTreeConfig,SingleTreeRegressor
from torchboost.adaptive.scaling import migrate_single_tree
from test_scaling_controls import cfg,small,obs
from torchboost.adaptive.config import PlasticityConfig
from torchboost.adaptive.plasticity import PlasticityModule

def test_trinary_multitarget_predictor_survives_native_import():
    rng=np.random.default_rng(41);x=rng.normal(size=(80,4));y=np.stack([x[:,0]+x[:,1],x[:,0]*x[:,1]],axis=1)
    fitted=SingleTreeRegressor(SingleTreeConfig(depth=2,arity=3,readout='residual',epochs=2,batch_size=32)).fit(x[:50],y[:50],eval_set=(x[50:65],y[50:65]))
    config=cfg('rlc');config.structure.arity=3
    trainer,pre=migrate_single_tree(fitted,config,updates_per_block=2)
    with torch.no_grad():np.testing.assert_allclose(trainer.model(pre.transform_x(x)).numpy(),fitted.model_(pre.transform_x(x)).numpy(),atol=2e-6)
    trainer.fit_steps(pre.split(x[:50],y[:50]),pre.split(x[50:65],y[50:65]),pre.split(x[65:],y[65:]))
    assert trainer.selected_model()(pre.transform_x(x)).shape==(80,2)
    trainer.close()

def test_positive_utility_resets_adverse_release_streak():
    config=PlasticityConfig(mode='full',release_policy='persistent_harm',release_patience=3,stiffness=10,yield_threshold=.01)
    module=PlasticityModule(config);parameters={'n':{'v':torch.nn.Parameter(torch.ones(3))}};module.synchronize(parameters)
    for step,utility in enumerate([-.1,-.1,.1,-.1,-.1,-.1]):
        event=module.advance(parameters,{'n':obs(step,utility)},{'n':1.},step,progress=.1)['events'][0]
        assert event['release_allowed']==(step==5)
    assert event['flow_fraction']>0
