import numpy as np
import torch
from torchboost.adaptive.single_tree import SingleTreeClassifier,SingleTreeConfig
from experiments.single_tree_gate_pressure import gate_pressure


def test_gate_pressure_matches_autograd_and_does_not_mutate():
    torch.set_num_threads(1)
    rng=np.random.default_rng(91);x=rng.normal(size=(35,4));y=(x[:,0]+x[:,1]>.1).astype(int)
    m=SingleTreeClassifier(SingleTreeConfig(depth=2,epochs=5,temperature=2.)).fit(x,y)
    state={k:v.clone() for k,v in m.model_.state_dict().items()}
    result=gate_pressure(m,x,y)
    xt=m.preprocessor_.transform_x(x);empirical={j:[] for j in range(m.model_.n_internal)}
    for i in range(len(x)):
        value=m.model_(xt[i:i+1])[0,0]
        grad=torch.autograd.grad(value,m.model_.routing_bias)[0]
        for j in empirical:empirical[j].append(float(grad[j].square().sum()))
    for row in result['nodes']:
        expected=np.sqrt(np.mean(empirical[row['node']]))
        np.testing.assert_allclose(row['prediction_jacobian_rms'],expected,rtol=2e-5,atol=1e-8)
    for k,v in m.model_.state_dict().items():torch.testing.assert_close(v,state[k],atol=0,rtol=0)
