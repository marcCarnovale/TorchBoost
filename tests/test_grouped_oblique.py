
import numpy as np, torch
from torchboost.adaptive.config import ForestConfig,StructureConfig
from torchboost.adaptive.data import DataSplit
from torchboost.adaptive.newton_builder import BuilderConfig,build_grouped_oblique_model_tree

def test_grouped_oblique_gate_has_no_cross_group_coefficients():
    rng=np.random.default_rng(4);x=rng.normal(size=(160,6)).astype("float32")
    y=((x[:,0]+x[:,1]>0) ^ (x[:,4]>0)).astype("float32")
    data=DataSplit(torch.tensor(x),torch.tensor(y),torch.ones(len(y)))
    native=ForestConfig(n_trees=1,aggregation="additive",residual_weights=False,node_linear_values=True,
        structure=StructureConfig(max_depth=2,initial_depth=0,dynamic=True,max_nodes=31))
    cfg=BuilderConfig(depth=1,linear_values=True,linear_l2=1.)
    tree,record=build_grouped_oblique_model_tree(data,torch.zeros(len(y),1),"binary",0,native,cfg,torch.ones(6),torch.Generator().manual_seed(2),((0,1),(4,5)),1.,2,8)
    for split in record["splits"]:
        node=tree.get(split["node_id"]);allowed=set(split["group"])
        used=set(torch.where(node.routing_weight.abs().sum(0)>1e-10)[0].tolist())
        assert used <= allowed

def test_feature_groups_validate():
    from torchboost.adaptive.unified_progressive import UnifiedConfig
    try: UnifiedConfig(proposal_mode="grouped_oblique",linear_values=True,feature_groups=())
    except ValueError: pass
    else: raise AssertionError("missing groups accepted")