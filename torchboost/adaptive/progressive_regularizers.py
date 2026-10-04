"""Penalties on the native predictive decomposition, not unused config knobs."""
from dataclasses import dataclass
import math
import torch

@dataclass
class Regularizers:
    leaf_l2: float = 0.
    hierarchy: float = 0.
    depth_slope: float = .5
    allocation: str = 'fixed'
    allocation_l2: float = 1e-3
    tree_l2: float = 0.
    route_balance: float = 0.
    min_child_fraction: float = .1
    child_penalty: float = 0.
    feature_l1: float = 0.
    correction_cost: float = 0.
    linear_value_l2: float = 0.

    def __post_init__(self):
        if self.allocation not in ('fixed','learned','exponential','uniform'):raise ValueError('invalid allocation')
        for k,v in vars(self).items():
            if k=='allocation':continue
            if not math.isfinite(v) or (k!='depth_slope' and v<0):raise ValueError(f'invalid {k}')
        if self.min_child_fraction>=.5:raise ValueError('child fraction must be <.5')


def allocation(tree,cfg):
    nodes=list(tree.nodes.values());depths=sorted({n.depth for n in nodes});d=tree.depth_logits.new_tensor(depths)
    if cfg.allocation=='learned':z=tree.depth_logits[depths]*d
    elif cfg.allocation=='fixed':z=cfg.depth_slope*d
    elif cfg.allocation=='exponential':z=-d
    else:z=torch.zeros_like(d)
    answer={}
    for depth,budget in zip(depths,z.softmax(0)):
        group=[n for n in nodes if n.depth==depth]
        within=torch.stack([-n.allocation_logit for n in group]).softmax(0) if cfg.allocation=='learned' else d.new_full((len(group),),1/len(group))
        answer.update({n.node_id:budget*f for n,f in zip(group,within)})
    return answer


def canonical_leaves(tree):
    out=[];root=tree.get(tree.root_id)
    def visit(key,prefix,scale):
        n=tree.get(key);value=prefix+scale*n.value
        if n.is_leaf:out.append(value)
        else:
            for c in n.children_ids:visit(c,value,scale*n.gate())
    visit(tree.root_id,torch.zeros_like(root.value),root.value.new_tensor(1.))
    return torch.stack(out)


def penalties(model,weights,trace,cfg):
    z=model.bias.new_zeros(())
    out={k:z for k in ('leaf','hierarchy','allocation','feature','tree','balance','child')}
    for i,t in enumerate(model.trees):
        rate=model.stage_rates[i]
        if cfg.leaf_l2:out['leaf']=out['leaf']+.5*cfg.leaf_l2*(rate*canonical_leaves(t)).square().mean()
        if cfg.hierarchy:
            budget=allocation(t,cfg)
            for n in t.nodes.values():
                local=(rate*n.value).square().mean()
                if n.linear_value is not None: local=local+(rate*n.linear_value).square().mean()
                out['hierarchy']=out['hierarchy']+.5*cfg.hierarchy*budget[n.node_id]*local
        if cfg.linear_value_l2:
            terms=[n.linear_value.square().mean() for n in t.nodes.values() if n.linear_value is not None]
            if terms: out['leaf']=out['leaf']+.5*cfg.linear_value_l2*torch.stack(terms).mean()
            if cfg.allocation=='learned':out['allocation']=out['allocation']+cfg.allocation_l2*(t.depth_logits.square().mean()+torch.stack([n.allocation_logit.square() for n in t.nodes.values()]).mean())
        if cfg.feature_l1:
            terms=[((n.routing_weight-n.routing_weight.mean(0,keepdim=True))*t.feature_mask/n.temperature).abs().mean() for n in t.nodes.values() if not n.is_leaf]
            if terms:out['feature']=out['feature']+cfg.feature_l1*torch.stack(terms).mean()
    if trace is not None:
        if cfg.tree_l2:out['tree']=cfg.tree_l2*(weights[:,None]*(trace.tree_outputs*trace.coefficients).square().mean(2)).sum()/weights.sum()
        for n in model.iter_nodes():
            raw=trace.nodes.get(n.node_id)
            if raw is None or raw.probabilities is None:continue
            rw=weights*raw.reach;mass=rw.sum();usage=(rw[:,None]*raw.probabilities).sum(0)/mass.clamp_min(1e-12)
            occupancy=mass/weights.sum()
            if cfg.route_balance:out['balance']=out['balance']+cfg.route_balance*occupancy*(-usage.clamp_min(1e-8).log().mean()-math.log(len(usage)))
            if cfg.child_penalty:
                t=model.get_tree(n.tree_id);values=torch.stack([t.get(c).value for c in n.children_ids])
                out['child']=out['child']+cfg.child_penalty*occupancy*(torch.relu(cfg.min_child_fraction-usage)*values.square().mean(1)).sum()
    # SUM over members: adding a zero function must not dilute existing penalties.
    return out
