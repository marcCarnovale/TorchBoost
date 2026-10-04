"""Leaf-level reducibility evidence for structural budgeting."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass(frozen=True)
class LeafEvidence:
    effective_n: float = 0.
    mass: float = 0.
    local_loss: float = 0.
    residual_variance: float = 0.
    reducible_loss: float = 0.
    explainable_fraction: float = 0.
    noise_fraction: float = 1.
    confident_error_mass: float = 0.
    exploration_score: float = 0.
    budget_score: float = 0.

def _weighted_var(value: Tensor, weight: Tensor) -> Tensor:
    total=weight.sum().clamp_min(1e-12)
    mean=(weight[:,None]*value).sum(0)/total
    return (weight[:,None]*(value-mean).square()).sum(0)/total

@torch.no_grad()
def leaf_evidence(x: Tensor, logits: Tensor, target: Tensor, weight: Tensor, reach: Tensor,
                  task: str, ridge: float=1., max_features: int=16, min_effective_n: float=12.) -> LeafEvidence:
    rw=weight*reach;total=rw.sum()
    if total<=1e-12:return LeafEvidence()
    effective=float(total.square()/rw.square().sum().clamp_min(1e-12))
    mass=float(total/weight.sum().clamp_min(1e-12))
    if task=="binary":
        p=logits[:,0].sigmoid();y=target.float().reshape(-1)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0],y,reduction="none")
        residual=(y-p)[:,None];curvature=(p*(1-p)).clamp_min(1e-5)[:,None]
        confident=((p-y).abs()>.75).to(rw.dtype)
    elif task=="multiclass":
        p=logits.softmax(1);labels=torch.nn.functional.one_hot(target.long().reshape(-1),logits.shape[1]).to(logits)
        loss=torch.nn.functional.cross_entropy(logits,target.long().reshape(-1),reduction="none")
        residual=labels-p;curvature=(p*(1-p)).clamp_min(1e-5)
        confident=((p.max(1).values>.75)&(p.argmax(1)!=target.long().reshape(-1))).to(rw.dtype)
    else:
        y=target.reshape_as(logits);residual=y-logits;curvature=torch.ones_like(residual)
        loss=residual.square().mean(1);confident=torch.zeros_like(rw)
    local_loss=float((rw*loss).sum()/total);variance=float(_weighted_var(residual,rw).mean())
    confident_error=float((rw*confident).sum()/total)
    if effective<min_effective_n or x.shape[1]==0:
        return LeafEvidence(effective,mass,local_loss,variance,0.,0.,1.,confident_error,mass*variance,0.)
    centered=x-(rw[:,None]*x).sum(0)/total
    corr=(centered.T@(rw[:,None]*residual)).square().sum(1);k=min(max_features,x.shape[1])
    cols=torch.topk(corr,k).indices if k<x.shape[1] else torch.arange(x.shape[1],device=x.device)
    design=torch.cat((torch.ones(len(x),1,device=x.device,dtype=x.dtype),x[:,cols]),1)
    eye=torch.eye(design.shape[1],device=x.device,dtype=x.dtype);eye[0,0]=0.;gains=[]
    for q in range(residual.shape[1]):
        h=design.T@((rw*curvature[:,q])[:,None]*design)+ridge*eye
        g=design.T@(rw*residual[:,q])
        try:gains.append(.5*torch.dot(g,torch.linalg.solve(h,g))/total)
        except RuntimeError:gains.append(logits.new_zeros(()))
    gain=max(0.,float(torch.stack(gains).mean()));explain=min(1.,gain/max(local_loss,1e-12))
    reliability=effective/(effective+2*(k+1))
    return LeafEvidence(effective,mass,local_loss,variance,gain,explain,1.-explain,confident_error,
                        mass*variance*reliability,mass*gain*reliability)
