"""Native, ragged histogram-Newton trees with true loss derivatives.

The proposal is axis-aligned. Native training can subsequently soften/rotate its
gates. Full multiclass curvature is retained; the split search is greedy and
quantile-binned, not claimed equivalent to CART or XGBoost's implementation.
"""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import torch
from scipy.special import expit, softmax
from .forest import RaggedTree


def derivatives(score, target, task):
    n, k = score.shape
    if task == 'binary':
        p = expit(score)
        return p-target.reshape(n,1), (p*(1-p)).reshape(n,1,1)
    if task == 'multiclass':
        p = softmax(score,axis=1)
        h = -p[:,:,None]*p[:,None,:]
        h[:,np.arange(k),np.arange(k)] += p
        return p-np.eye(k)[target.astype(int)], h
    if task == 'regression':
        return 2*(score-target.reshape(n,k))/k, np.broadcast_to(2*np.eye(k)/k,(n,k,k)).copy()
    raise ValueError('unknown objective')


def leaf_solution(g,h,l2):
    # The fixed numerical jitter also resolves softmax's translation gauge.
    value = -np.linalg.solve(h+(l2+1e-10)*np.eye(g.shape[-1]),g[...,None])[...,0]
    return value, -(g*value).sum(-1)


@dataclass(frozen=True)
class BuilderConfig:
    depth: int = 3
    bins: int = 32
    min_samples: int = 5
    min_child_weight: float = 1.
    l2: float = 1.
    split_cost: float = 0.
    max_delta: float = 5.
    strength: float = 8.
    readout: str = 'residual'


@torch.no_grad()
def build_tree(data, scores, task, tree_id, native, cfg, feature_mask, generator):
    if cfg.depth>native.structure.max_depth:raise ValueError('proposal exceeds topology budget')
    temporary=deepcopy(native)
    temporary.structure.dynamic=True;temporary.structure.initial_depth=0;temporary.structure.arity=2
    tree=RaggedTree(tree_id,data.x.shape[1],scores.shape[1],temporary,generator)
    tree.config=native;tree.feature_mask.copy_(feature_mask)
    x=data.x.numpy().astype(np.float64);y=data.y.numpy();w=data.weight.numpy().astype(np.float64)
    g,h=derivatives(scores.numpy().astype(np.float64),y,task);g*=w[:,None];h*=w[:,None,None]
    columns=np.flatnonzero(feature_mask.numpy())
    edges={int(j):np.unique(np.quantile(x[:,j],np.linspace(0,1,cfg.bins+1)[1:-1])) for j in columns}
    bins={j:np.searchsorted(v,x[:,j],side='left') for j,v in edges.items()}
    queue=[(tree.root_id,np.flatnonzero(w>0),np.zeros(scores.shape[1]))]
    splits=[];leaves=[]
    for key,rows,parent in queue:
        node=tree.get(key);G=g[rows].sum(0);H=h[rows].sum(0)
        value,parent_score=leaf_solution(G,H,cfg.l2);value=np.clip(value,-cfg.max_delta,cfg.max_delta)
        node.value.copy_(torch.as_tensor(value-parent if cfg.readout=='residual' else value,dtype=node.value.dtype))
        best=None
        if node.depth<cfg.depth and len(rows)>=2*cfg.min_samples:
            for j,cuts in edges.items():
                if not len(cuts):continue
                b=bins[j][rows];nb=len(cuts)+1;k=g.shape[1]
                count=np.bincount(b,minlength=nb)
                gs=np.stack([np.bincount(b,weights=g[rows,q],minlength=nb) for q in range(k)],1)
                hs=np.stack([np.bincount(b,weights=h[rows,q,r],minlength=nb) for q in range(k) for r in range(k)],1).reshape(nb,k,k)
                gl,hl,nl=gs.cumsum(0)[:-1],hs.cumsum(0)[:-1],count.cumsum()[:-1]
                gr,hr,nr=G-gl,H-hl,len(rows)-nl
                valid=(nl>=cfg.min_samples)&(nr>=cfg.min_samples)&(np.trace(hl,axis1=-2,axis2=-1)>=cfg.min_child_weight)&(np.trace(hr,axis1=-2,axis2=-1)>=cfg.min_child_weight)
                if not valid.any():continue
                _,sl=leaf_solution(gl,hl,cfg.l2);_,sr=leaf_solution(gr,hr,cfg.l2)
                gain=.5*(sl+sr-parent_score)-cfg.split_cost;gain[~valid]=-np.inf
                q=int(gain.argmax())
                if gain[q]>1e-12 and (best is None or gain[q]>best[0]):
                    best=(float(gain[q]),j,q,float(cuts[q]),float(np.trace(hl[q])),float(np.trace(hr[q])))
        if best is None:
            leaves.append({'node_id':key,'rows':len(rows),'hessian_mass':float(np.trace(H)),'value':value.tolist()});continue
        gain,j,q,threshold,hl,hr=best
        children=tree.grow(key,generator=generator,arity=2)
        if not children:
            leaves.append({'node_id':key,'rows':len(rows),'hessian_mass':float(np.trace(H)),'value':value.tolist()});continue
        node.routing_weight.zero_();node.routing_bias.zero_()
        node.routing_weight[0,j]=-cfg.strength/2;node.routing_weight[1,j]=cfg.strength/2
        node.routing_bias[0]=cfg.strength*threshold/2;node.routing_bias[1]=-cfg.strength*threshold/2
        if cfg.readout=='leaf':node.value.zero_()
        prior=value if cfg.readout=='residual' else np.zeros_like(value)
        queue.extend([(children[0],rows[bins[j][rows]<=q],prior),(children[1],rows[bins[j][rows]>q],prior)])
        splits.append({'node_id':key,'feature':int(j),'threshold':threshold,'gain':gain,'left_hessian_mass':hl,'right_hessian_mass':hr})
    if cfg.readout=='leaf':
        for node in tree.nodes.values():
            if not node.is_leaf:node.value.requires_grad_(False)
    return tree,{'splits':splits,'leaves':leaves,'features':columns.tolist(),'rows':int((w>0).sum()),'criterion':'full-Hessian histogram gain'}
