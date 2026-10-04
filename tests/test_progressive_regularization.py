
import numpy as np
import torch
from torchboost.adaptive.progressive import (
    ProgressiveConfig, _new_packed, _tree_regularization,
    residual_cart_initialize, _feature_mask, _mask_tree_gradients
)

def test_regularization_is_real_and_depth_sensitive():
    cfg=ProgressiveConfig(depth=3,leaf_l2=.1,depth_shrinkage=.2)
    tree,_=_new_packed(4,1,cfg,3)
    with torch.no_grad():
        tree.values.fill_(1.)
    reg=float(_tree_regularization(tree,cfg))
    assert reg > .1
    with torch.no_grad():
        tree.values.zero_()
    assert float(_tree_regularization(tree,cfg)) == 0.

def test_feature_subsample_limits_cart_and_gradient_directions():
    cfg=ProgressiveConfig(depth=2,feature_subsample=.5)
    tree,_=_new_packed(6,1,cfg,4)
    g=torch.Generator().manual_seed(7)
    mask=_feature_mask(6,.5,g,tree.values.device)
    x=torch.randn(128,6,generator=torch.Generator().manual_seed(8))
    y=(2*x[:,0]-x[:,1]).reshape(-1,1); w=torch.ones(128)
    residual_cart_initialize(tree,x,y,w,8.,feature_mask=mask)
    assert torch.count_nonzero(tree.routing_weight[...,~mask]) == 0
    tree.routing_weight.grad=torch.ones_like(tree.routing_weight)
    _mask_tree_gradients(tree,mask)
    assert torch.count_nonzero(tree.routing_weight.grad[...,~mask]) == 0

def test_residual_readout_accepts_cart_residual_warm_start():
    cfg=ProgressiveConfig(depth=2,readout="residual")
    tree,_=_new_packed(3,1,cfg,9)
    x=torch.randn(64,3,generator=torch.Generator().manual_seed(10))
    y=x[:,0,None];w=torch.ones(64)
    residual_cart_initialize(tree,x,y,w,8.)
    assert torch.isfinite(tree(x)).all()
    assert tree.values.shape[0] == tree.n_nodes


def test_binary_newton_refit_is_finite():
    from torchboost.adaptive.progressive import _newton_refit_binary
    cfg=ProgressiveConfig(depth=2)
    tree,_=_new_packed(3,1,cfg,12)
    x=torch.randn(96,3,generator=torch.Generator().manual_seed(13))
    y=(x[:,0]>0).float()
    w=torch.ones(96)
    base=torch.zeros(96,1)
    before=torch.nn.functional.binary_cross_entropy_with_logits(base+.3*tree(x),y[:,None])
    _newton_refit_binary(tree,base+.3*tree(x),x,y,w,.3,1e-3)
    after=torch.nn.functional.binary_cross_entropy_with_logits(base+.3*tree(x),y[:,None])
    assert torch.isfinite(after)
    assert after <= before + 1e-6
