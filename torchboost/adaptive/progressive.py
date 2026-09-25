"""Progressive additive differentiable trees with CART/Newton warm starts.

The ensemble is a literal sum of trees.  New trees are fitted to the current
loss gradient/curvature.  Earlier trees remain trainable, but their optimizer
learning rates decay geometrically with age.  This interpolates between strict
stagewise boosting (old_tree_lr_decay=0) and joint forest training (=1).
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .single_tree import SingleTreeConfig, PackedSingleTree, cart_initialize
from .forest import AdaptiveForest
from .data import Preprocessor, sample_weights
from .objectives import Objective


@dataclass
class ProgressiveConfig:
    n_trees: int = 8
    depth: int = 4
    stage_updates: int = 256
    batch_size: int = 512
    learning_rate: float = 0.01
    new_tree_shrinkage: float = 0.3
    old_tree_lr_decay: float = 0.25
    weight_decay: float = 1e-4
    cart_strength: float = 8.0
    routing_temperature: float = 1.0
    anchor_strength: float = 0.0
    # Explicit tree regularization.  These act on the predictive decomposition,
    # not merely on optimizer parameter norms.
    leaf_l2: float = 0.0
    depth_shrinkage: float = 0.0
    route_balance: float = 0.0

    newton_l2: float = 1e-3
    min_child_mass: float = 0.0
    row_subsample: float = 1.0
    feature_subsample: float = 1.0
    readout: str = 'leaf'
    # CART is an embedded hard proposal.  Hold routing fixed for a configurable
    # fraction of each stage, then release it into the richer oblique model.
    cart_value_updates: int = 32
    gradient_clip: float = 10.0
    min_improvement: float = 1e-6
    patience_stages: int = 3
    random_state: int = 0

    def __post_init__(self):

        for name in ("learning_rate","new_tree_shrinkage","cart_strength","routing_temperature","gradient_clip"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name) <= 0: raise ValueError(f"{name} must be positive")
        if not 0 <= self.old_tree_lr_decay <= 1: raise ValueError("old_tree_lr_decay must lie in [0,1]")
        if self.readout not in ("leaf","residual"): raise ValueError("readout must be leaf or residual")

            if not math.isfinite(getattr(self,name)) or getattr(self,name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("row_subsample","feature_subsample"):
            if not math.isfinite(getattr(self,name)) or not 0 < getattr(self,name) <= 1:
                raise ValueError(f"{name} must lie in (0,1]")



def _depth_weights(model: PackedSingleTree) -> torch.Tensor:
    """Per-node output penalty weights; deeper residual increments shrink more."""
    if model.readout == "leaf":
        return model.values.new_ones(len(model.values))
    depths = []
    for i in range(model.n_nodes):
        # Complete k-ary breadth-first index: count ancestors.
        d, q = 0, i
        while q:
            q = (q - 1) // model.arity
            d += 1
        depths.append(d)
    return model.values.new_tensor(depths)

def _tree_regularization(tree: PackedSingleTree, cfg: ProgressiveConfig) -> torch.Tensor:
    z = tree.values.new_zeros(())
    if cfg.leaf_l2:
        z = z + cfg.leaf_l2 * tree.values.square().mean()
    if cfg.depth_shrinkage:
        weights = 1. + cfg.depth_shrinkage * _depth_weights(tree)
        z = z + cfg.depth_shrinkage * (weights[:,None] * tree.values.square()).mean()
    return z


def _routing_mass_penalty(tree: PackedSingleTree, x: torch.Tensor, cfg: ProgressiveConfig) -> torch.Tensor:
    if cfg.min_child_mass <= 0 or tree.n_internal == 0:
        return tree.values.new_zeros(())
    mass, prob = tree.routing(x)
    reach = mass[:, :tree.n_internal]
    child = reach[...,None] * prob
    effective = child.sum(0) / max(1, len(x))
    # Soft analogue of minimum child weight: penalize tiny effective children.
    return cfg.min_child_mass * torch.relu(cfg.min_child_mass - effective).square().mean()


def _sample_indices(n: int, batch: int, fraction: float, generator: torch.Generator) -> torch.Tensor:
    size = min(batch, n, max(1, int(math.ceil(n * fraction))))
    return torch.randint(n, (size,), generator=generator)

def _feature_mask(d: int, fraction: float, generator: torch.Generator, device) -> torch.Tensor | None:
    if fraction >= 1:
        return None
    k = max(1, int(math.ceil(d * fraction)))
    order = torch.randperm(d, generator=generator)[:k]
    mask = torch.zeros(d, dtype=torch.bool)
    mask[order] = True
    return mask.to(device)

def _mask_tree_gradients(tree: PackedSingleTree, mask: torch.Tensor | None) -> None:
    if mask is not None and tree.routing_weight.grad is not None:
        tree.routing_weight.grad[..., ~mask] = 0


@torch.no_grad()
def _newton_refit_binary(tree: PackedSingleTree, base_score: torch.Tensor, x: torch.Tensor,
                         y: torch.Tensor, weight: torch.Tensor, rate: float, l2: float) -> None:
    """One exact IRLS/Newton solve for tree output values with routing held fixed.

    The solve uses the current full-ensemble probabilities, so it is the
    differentiable-tree analogue of XGBoost's regularized leaf Newton step.
    """
    if tree.output_dim != 1:
        return
    mass, _ = tree.routing(x)
    design = mass if tree.readout == 'residual' else mass[:, tree.n_internal:]
    # Include the current tree in the base only through its old output; remove it
    # so the solve replaces, rather than double-counts, the tree's contribution.
    old = tree(x)
    score_without = base_score - rate * old
    score = base_score
    prob = score.sigmoid()
    g = (prob - y[:,None]).double()
    h = (prob * (1-prob)).clamp_min(1e-5).double()
    A = (rate * design).double()
    w = weight[:,None].double()
    # IRLS target relative to score_without.
    z = (score - g/h - score_without).double()
    H = A.T @ (w*h*A) + l2*torch.eye(A.shape[1],dtype=torch.float64,device=A.device)
    rhs = A.T @ (w*h*z)
    try:
        value = torch.linalg.solve(H,rhs)
    except RuntimeError:
        return
    if torch.isfinite(value).all():
        tree.values.copy_(value.to(tree.values))

    def forward(self,x):
        out=self.bias.expand(len(x),-1)
        for r,t in zip(self.rates,self.trees): out=out+r*t(x)
        return out


def _new_packed(input_dim, output_dim, cfg: ProgressiveConfig, seed: int):
    sc=SingleTreeConfig(depth=cfg.depth,arity=2,epochs=1,learning_rate=cfg.learning_rate,
        temperature=cfg.routing_temperature,readout=cfg.readout,initializer='random',batch_size=cfg.batch_size,
        random_state=seed,cart_strength=cfg.cart_strength,max_nodes=max(8191,2**(cfg.depth+1)-1))
    native=AdaptiveForest(input_dim,output_dim,sc.forest_config(),generator=torch.Generator().manual_seed(seed))
    return PackedSingleTree(native,readout=cfg.readout),sc


@torch.no_grad()
def residual_cart_initialize(model: PackedSingleTree, x: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, strength: float, min_leaf: int=5, feature_mask: torch.Tensor | None=None):
    """Fit a hard CART regression tree to Newton/gradient targets and embed it exactly in topology.

    The embedded routing is a finite-temperature soft approximation; missing lower
    levels duplicate the terminal CART prediction, so added topology is a no-op.
    """
    if model.arity != 2: raise ValueError('binary model required')
    xx=x.detach().cpu().numpy(); yy=target.detach().cpu().numpy(); ww=weight.detach().cpu().numpy()
    columns = np.arange(xx.shape[1]) if feature_mask is None else np.flatnonzero(feature_mask.detach().cpu().numpy())
    xx_fit = xx[:, columns]
    cart=DecisionTreeRegressor(max_depth=model.depth,min_samples_leaf=min_leaf,random_state=0)
    cart.fit(xx_fit,yy,sample_weight=ww)
    model.bias.zero_(); model.values.zero_(); t=cart.tree_
    def fill(i,k):
        if i>=model.n_internal:
            out_index = i-model.n_internal if model.readout=='leaf' else i
            model.values[out_index].copy_(torch.as_tensor(t.value[k].reshape(-1),dtype=model.values.dtype)); return
        feat=t.feature[k]
        if feat>=0:
            feat=int(columns[feat]); model.routing_weight[i].zero_();model.routing_weight[i,0,feat]=-strength/2;model.routing_weight[i,1,feat]=strength/2
            model.routing_bias[i,0]=strength*t.threshold[k]/2;model.routing_bias[i,1]=-strength*t.threshold[k]/2
            fill(2*i+1,t.children_left[k]);fill(2*i+2,t.children_right[k])
        else:
            fill(2*i+1,k);fill(2*i+2,k)
    fill(0,0)


class _ProgressiveEstimator(BaseEstimator):
    classification=False
    def __init__(self,config=None): self.config=config
    def fit(self,X,y,sample_weight=None,*,eval_set=None):
        cfg=deepcopy(self.config or ProgressiveConfig());cfg.__post_init__();self.config_=cfg
        w=sample_weights(sample_weight,len(X));self.preprocessor_=Preprocessor();self.preprocessor_.fit(X,y,classification=self.classification,weights=w)
        train=self.preprocessor_.split(X,y,w);valid=train if eval_set is None else self.preprocessor_.split(*eval_set)
        self.objective_=Objective(self.preprocessor_.task,self.preprocessor_.output_dim);self.n_features_in_=train.x.shape[1]
        if self.classification:self.classes_=self.preprocessor_.classes.copy()
        if self.classification:
            counts=torch.bincount(train.y,weights=train.weight,minlength=len(self.classes_));p=(counts/counts.sum()).clamp_min(1e-7)
            bias=(p[1]/p[0]).log().reshape(1) if len(p)==2 else p.log()
        else:bias=(train.y*train.weight[:,None]).sum(0)/train.weight.sum()
        self.model_=ProgressiveSum(bias,cfg.learn_tree_rates);self.history_=[];self.stage_states_=[];self.best_score_=float(self.objective_.weighted_loss(self.model_(valid.x),valid.y,valid.weight).detach());self.best_state_=deepcopy(self.model_.state_dict());self.best_stage_=0
        rng=torch.Generator().manual_seed(cfg.random_state);stale=0
        for stage in range(cfg.n_trees):
            tree,sc=_new_packed(self.n_features_in_,self.objective_.output_dim,cfg,cfg.random_state+1009*stage)
            feature_mask=_feature_mask(self.n_features_in_,cfg.feature_subsample,rng,tree.values.device)
            with torch.no_grad():
                current=self.model_(train.x)
                if stage==0 and self.classification and cfg.readout=='leaf':
                    # A true CART-like first model: supervised hard partition embedded into the soft model.
                    cart_initialize(tree,train,self.objective_,sc)
                    rate=1.0
                else:
                    if self.objective_.task=='binary':
                        p=current.sigmoid();g=p-train.y[:,None];h=(p*(1-p)).clamp_min(1e-4);target=-g/h;rw=train.weight[:,None]*h
                    elif self.objective_.task=='multiclass':
                        p=current.softmax(1);g=p-torch.nn.functional.one_hot(train.y.long(),self.objective_.output_dim);target=-g;rw=train.weight[:,None].expand_as(target)
                    else:
                        target=train.y-current;rw=train.weight[:,None].expand_as(target)
                    residual_cart_initialize(tree,train.x,target,rw.mean(1),cfg.cart_strength,feature_mask=feature_mask)
                    rate=cfg.new_tree_shrinkage
            self.model_.append(tree,rate)
            anchors=[{n:p.detach().clone() for n,p in t.named_parameters()} for t in self.model_.trees[:-1]]
            # One group per tree.  Oldest tree gets the smallest LR; the new tree gets full LR.
            groups=[];n=len(self.model_.trees)
            for j,t in enumerate(self.model_.trees):
                age=n-1-j;groups.append({'params':list(t.parameters()),'lr':cfg.learning_rate*(cfg.old_tree_lr_decay**age)})

            for update in range(cfg.stage_updates):
                idx=_sample_indices(len(train.x),cfg.batch_size,cfg.row_subsample,rng)
                opt.zero_grad(set_to_none=True);pred=self.model_(train.x[idx]);loss=self.objective_.weighted_loss(pred,train.y[idx],train.weight[idx])
                if cfg.anchor_strength and anchors:
                    penalty=pred.new_zeros(())
                    for j,t in enumerate(self.model_.trees[:-1]):
                        for name,p in t.named_parameters(): penalty=penalty+(p-anchors[j][name]).square().mean()
                    loss=loss+cfg.anchor_strength*penalty/max(1,len(anchors))

                loss.backward()
                # Preserve the CART partition briefly: values adapt first, then routing flips to differentiable training.
                if update < cfg.cart_value_updates:
                    for p in (tree.routing_weight,tree.routing_bias): p.grad=None
                else:
                    _mask_tree_gradients(tree,feature_mask)
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(),cfg.gradient_clip,error_if_nonfinite=True);opt.step()

            self.stage_states_.append(deepcopy(self.model_.state_dict()))
            if va < self.best_score_-cfg.min_improvement:
                self.best_score_=va;self.best_stage_=n;self.best_state_=deepcopy(self.model_.state_dict());stale=0
            else:stale+=1
            if stale>=cfg.patience_stages:break
        # Reconstruct the best prefix, because state_dict shapes grow stage by stage.

        else:self.model_.bias.data.copy_(self.best_state_['bias'])
        self.model_.eval();self.n_estimators_=best_trees
        return self
    @torch.no_grad()
    def _raw(self,X):
        check_is_fitted(self,'model_');x=self.preprocessor_.transform_x(X);return torch.cat([self.model_(b) for b in x.split(2048)]).numpy()
    def predict(self,X):
        raw=self._raw(X)
        if self.classification:return self.classes_[self.predict_proba(X).argmax(1)]
        y=self.preprocessor_.inverse_target(raw);return y[:,0] if y.shape[1]==1 else y

class ProgressiveTreeClassifier(ClassifierMixin,_ProgressiveEstimator):
    classification=True
    def predict_proba(self,X): return self.objective_.response(torch.from_numpy(self._raw(X))).numpy()

class ProgressiveTreeRegressor(RegressorMixin,_ProgressiveEstimator): pass

@dataclass
class RollingBoostConfig(ProgressiveConfig):
    active_window: int = 4
    joint_updates: int = 8
    joint_every: int = 4
    def __post_init__(self):
        super().__post_init__()
        if self.active_window < 1 or self.joint_updates < 0 or self.joint_every < 1: raise ValueError('invalid rolling schedule')

class RollingBoostClassifier(ClassifierMixin, BaseEstimator):
    """Efficient progressive sum: fit each new tree against cached old scores.

    Every ``joint_every`` stages, only the most recent ``active_window`` trees are
    jointly refined.  Within that window, older trees receive geometrically lower
    learning rates.  Trees that age out are stable contributors, not deleted.
    """
    def __init__(self,config=None): self.config=config
    def fit(self,X,y,sample_weight=None,*,eval_set=None):
        cfg=deepcopy(self.config or RollingBoostConfig());cfg.__post_init__();self.config_=cfg
        w=sample_weights(sample_weight,len(X));self.preprocessor_=Preprocessor();self.preprocessor_.fit(X,y,classification=True,weights=w)
        train=self.preprocessor_.split(X,y,w);valid=train if eval_set is None else self.preprocessor_.split(*eval_set)
        self.objective_=Objective(self.preprocessor_.task,self.preprocessor_.output_dim);self.classes_=self.preprocessor_.classes.copy();self.n_features_in_=train.x.shape[1]
        counts=torch.bincount(train.y,weights=train.weight,minlength=len(self.classes_));p=(counts/counts.sum()).clamp_min(1e-7);bias=(p[1]/p[0]).log().reshape(1) if len(p)==2 else p.log()
        self.model_=ProgressiveSum(bias);rng=torch.Generator().manual_seed(cfg.random_state);self.history_=[]
        with torch.no_grad():train_score=self.model_(train.x).detach();valid_score=self.model_(valid.x).detach()
        best=float(self.objective_.weighted_loss(valid_score,valid.y,valid.weight));best_stage=0;best_state=deepcopy(self.model_.state_dict());stale=0
        for stage in range(cfg.n_trees):
            tree,sc=_new_packed(self.n_features_in_,self.objective_.output_dim,cfg,cfg.random_state+1009*stage)
            feature_mask=_feature_mask(self.n_features_in_,cfg.feature_subsample,rng,tree.values.device)
            with torch.no_grad():
                if stage==0 and cfg.readout=='leaf':
                    cart_initialize(tree,train,self.objective_,sc);rate=1.
                else:
                    if self.objective_.task=='binary':
                        pp=train_score.sigmoid();g=pp-train.y[:,None];h=(pp*(1-pp)).clamp_min(1e-4);target=-g/h;rw=train.weight[:,None]*h
                    else:
                        pp=train_score.softmax(1);g=pp-torch.nn.functional.one_hot(train.y.long(),self.objective_.output_dim);target=-g;rw=train.weight[:,None].expand_as(target)
                    residual_cart_initialize(tree,train.x,target,rw.mean(1),cfg.cart_strength,feature_mask=feature_mask);rate=cfg.new_tree_shrinkage
            # Fit ONLY the new correction. Old prediction is a detached cache.
            opt=torch.optim.AdamW(tree.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
            for update in range(cfg.stage_updates):
                idx=_sample_indices(len(train.x),cfg.batch_size,cfg.row_subsample,rng);opt.zero_grad(set_to_none=True)
                loss=self.objective_.weighted_loss(train_score[idx]+rate*tree(train.x[idx]),train.y[idx],train.weight[idx])
                loss=loss+_tree_regularization(tree,cfg)+_routing_mass_penalty(tree,train.x[idx],cfg)
                loss.backward()
                if update<cfg.cart_value_updates:
                    tree.routing_weight.grad=None;tree.routing_bias.grad=None
                else:
                    _mask_tree_gradients(tree,feature_mask)
                torch.nn.utils.clip_grad_norm_(tree.parameters(),cfg.gradient_clip,error_if_nonfinite=True);opt.step()
            if cfg.newton_refit and self.objective_.task=='binary':
                with torch.no_grad():
                    candidate_score=train_score+rate*tree(train.x)
                _newton_refit_binary(tree,candidate_score,train.x,train.y,train.weight,rate,cfg.newton_l2)
            self.model_.append(tree,rate)
            with torch.no_grad():train_score=train_score+rate*tree(train.x);valid_score=valid_score+rate*tree(valid.x)
            # Rolling joint correction: older trees receive progressively smaller LRs.
            joint=False
            if cfg.joint_updates and (stage+1)%cfg.joint_every==0 and len(self.model_.trees)>1:
                joint=True;start=max(0,len(self.model_.trees)-cfg.active_window);active=list(self.model_.trees[start:]);active_rates=self.model_.rates[start:]
                with torch.no_grad():
                    frozen=self.model_.bias.expand(len(train.x),-1).detach().clone();frozen_v=self.model_.bias.expand(len(valid.x),-1).detach().clone()
                    for r,t in zip(self.model_.rates[:start],self.model_.trees[:start]):frozen+=r*t(train.x);frozen_v+=r*t(valid.x)
                groups=[]
                for j,t in enumerate(active):groups.append({'params':list(t.parameters()),'lr':cfg.learning_rate*(cfg.old_tree_lr_decay**(len(active)-1-j))})
                jo=torch.optim.AdamW(groups,weight_decay=cfg.weight_decay)
                for _ in range(cfg.joint_updates):
                    idx=_sample_indices(len(train.x),cfg.batch_size,cfg.row_subsample,rng);jo.zero_grad(set_to_none=True);z=frozen[idx]
                    for r,t in zip(active_rates,active):z=z+r*t(train.x[idx])
                    loss=self.objective_.weighted_loss(z,train.y[idx],train.weight[idx])
                    if active:
                        loss=loss+sum(_tree_regularization(t,cfg) for t in active)/len(active)
                    loss.backward();torch.nn.utils.clip_grad_norm_([p for t in active for p in t.parameters()],cfg.gradient_clip,error_if_nonfinite=True);jo.step()
                with torch.no_grad():
                    train_score=frozen.clone();valid_score=frozen_v.clone()
                    for r,t in zip(active_rates,active):train_score+=r*t(train.x);valid_score+=r*t(valid.x)
            with torch.no_grad():tr=float(self.objective_.weighted_loss(train_score,train.y,train.weight));va=float(self.objective_.weighted_loss(valid_score,valid.y,valid.weight))
            self.history_.append({'stage':stage+1,'train_loss':tr,'validation_loss':va,'joint_refined':joint,'active_window':min(cfg.active_window,stage+1)})
            if va<best-cfg.min_improvement:best=va;best_stage=stage+1;best_state=deepcopy(self.model_.state_dict());stale=0
            else:stale+=1
            if stale>=cfg.patience_stages:break
        self.model_.trees=nn.ModuleList(list(self.model_.trees)[:best_stage]);self.model_.rates=self.model_.rates[:best_stage].clone()
        if best_stage:self.model_.load_state_dict(best_state)
        else:self.model_.bias.data.copy_(best_state['bias'])
        self.model_.eval();self.n_estimators_=best_stage;self.best_score_=best;return self
    @torch.no_grad()
    def predict_proba(self,X):
        check_is_fitted(self,'model_');x=self.preprocessor_.transform_x(X);raw=torch.cat([self.model_(b) for b in x.split(2048)]);return self.objective_.response(raw).numpy()
    def predict(self,X):return self.classes_[self.predict_proba(X).argmax(1)]
