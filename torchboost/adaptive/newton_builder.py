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



def linear_solution(x, g, h, l2):
    """Regularized Newton affine correction and its gain score."""
    xa=np.concatenate([np.ones((len(x),1)),x],axis=1);p=xa.shape[1];k=g.shape[1]
    matrix=np.einsum('ip,iq,icd->pcqd',xa,xa,h).reshape(p*k,p*k)
    matrix += l2*np.eye(p*k)
    rhs=-np.einsum('ip,ic->pc',xa,g).reshape(-1)
    try: beta=np.linalg.solve(matrix,rhs).reshape(p,k)
    except np.linalg.LinAlgError: beta=np.linalg.lstsq(matrix,rhs,rcond=None)[0].reshape(p,k)
    prediction=xa@beta
    score=-(g*prediction).sum()
    return beta,score


def fit_binary_linear(x, base_score, target, weight, l2, max_iter=12):
    """IRLS fit of an affine correction to a fixed base logit."""
    xa=np.concatenate([np.ones((len(x),1)),x],axis=1);beta=np.zeros(xa.shape[1])
    y=np.asarray(target,dtype=float);w=np.asarray(weight,dtype=float);base=np.asarray(base_score).reshape(-1)
    for _ in range(max_iter):
        score=base+xa@beta;p=1/(1+np.exp(-np.clip(score,-30,30)));h=np.maximum(p*(1-p),1e-5)*w;g=(p-y)*w
        pen=np.eye(xa.shape[1]);pen[0,0]=0.;H=xa.T@(h[:,None]*xa)+l2*pen;grad=xa.T@g+l2*(pen@beta)
        try:delta=np.linalg.solve(H,-grad)
        except np.linalg.LinAlgError:break
        before=np.sum(w*(np.logaddexp(0,score)-y*score))+.5*l2*(beta[1:]@beta[1:]);step=1.
        for _ in range(10):
            cand=beta+step*delta;s2=base+xa@cand;after=np.sum(w*(np.logaddexp(0,s2)-y*s2))+.5*l2*(cand[1:]@cand[1:])
            if after<=before:beta=cand;break
            step*=.5
        if np.max(np.abs(step*delta))<1e-7:break
    return beta[:,None]


def binary_linear_objective(x, base_score, target, weight, beta, l2, penalized=True):
    xa=np.concatenate([np.ones((len(x),1)),x],axis=1);score=np.asarray(base_score).reshape(-1)+xa@beta.reshape(-1)
    y=np.asarray(target,dtype=float);w=np.asarray(weight,dtype=float)
    value=np.sum(w*(np.logaddexp(0,score)-y*score))
    if penalized:value+=.5*l2*np.sum(beta.reshape(-1)[1:]**2)
    return float(value)

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
    linear_values: bool = False
    linear_l2: float = 10.


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
    zero_beta=np.zeros((x.shape[1]+1,scores.shape[1])) if cfg.linear_values else np.zeros(scores.shape[1])
    queue=[(tree.root_id,np.flatnonzero(w>0),zero_beta)]
    splits=[];leaves=[]
    for key,rows,parent in queue:
        node=tree.get(key);G=g[rows].sum(0);H=h[rows].sum(0)
        if cfg.linear_values:
            if task=='binary':
                beta=fit_binary_linear(x[rows],scores.numpy()[rows,0],y[rows],w[rows],cfg.linear_l2)
                parent_loss=binary_linear_objective(x[rows],scores.numpy()[rows,0],y[rows],w[rows],beta,cfg.linear_l2,False)
                parent_score=-parent_loss
            else:
                beta,parent_score=linear_solution(x[rows],g[rows],h[rows],cfg.linear_l2)
            value=np.clip(beta[0],-cfg.max_delta,cfg.max_delta); linear=np.clip(beta[1:],-cfg.max_delta,cfg.max_delta)
        else:
            value,parent_score=leaf_solution(G,H,cfg.l2);value=np.clip(value,-cfg.max_delta,cfg.max_delta);linear=None
        if cfg.linear_values:
            node.value.copy_(torch.as_tensor(value-parent[0] if cfg.readout=='residual' else value,dtype=node.value.dtype))
            if node.linear_value is not None:
                node.linear_value.copy_(torch.as_tensor(linear-parent[1:] if cfg.readout=='residual' else linear,dtype=node.value.dtype))
        else:
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
                if cfg.linear_values:
                    gain=np.full(len(cuts),-np.inf)
                    for qq in np.flatnonzero(valid):
                        left=rows[b<=qq];right=rows[b>qq]
                        if task=='binary':
                            bl=fit_binary_linear(x[left],scores.numpy()[left,0],y[left],w[left],cfg.linear_l2)
                            br=fit_binary_linear(x[right],scores.numpy()[right,0],y[right],w[right],cfg.linear_l2)
                            child_loss=binary_linear_objective(x[left],scores.numpy()[left,0],y[left],w[left],bl,cfg.linear_l2,False)+binary_linear_objective(x[right],scores.numpy()[right,0],y[right],w[right],br,cfg.linear_l2,False)
                            gain[qq]=-parent_score-child_loss-cfg.split_cost
                        else:
                            _,sl=linear_solution(x[left],g[left],h[left],cfg.linear_l2);_,sr=linear_solution(x[right],g[right],h[right],cfg.linear_l2)
                            gain[qq]=.5*(sl+sr-parent_score)-cfg.split_cost
                else:
                    _,sl=leaf_solution(gl,hl,cfg.l2);_,sr=leaf_solution(gr,hr,cfg.l2)
                    gain=.5*(sl+sr-parent_score)-cfg.split_cost;gain[~valid]=-np.inf
                q=int(gain.argmax())
                if gain[q]>1e-12 and (best is None or gain[q]>best[0]):
                    best=(float(gain[q]),j,q,float(cuts[q]),float(np.trace(hl[q])),float(np.trace(hr[q])))
        if best is None:
            if cfg.linear_values and task=='binary':
                exact=fit_binary_linear(x[rows],scores.numpy()[rows,0],y[rows],w[rows],cfg.linear_l2)
                exact_value=np.clip(exact[0],-cfg.max_delta,cfg.max_delta);exact_linear=np.clip(exact[1:],-cfg.max_delta,cfg.max_delta)
                node.value.copy_(torch.as_tensor(exact_value-parent[0] if cfg.readout=='residual' else exact_value,dtype=node.value.dtype))
                node.linear_value.copy_(torch.as_tensor(exact_linear-parent[1:] if cfg.readout=='residual' else exact_linear,dtype=node.value.dtype))
                value=exact_value
            leaves.append({'node_id':key,'rows':len(rows),'hessian_mass':float(np.trace(H)),'value':value.tolist()});continue
        gain,j,q,threshold,hl,hr=best
        children=tree.grow(key,generator=generator,arity=2)
        if not children:
            leaves.append({'node_id':key,'rows':len(rows),'hessian_mass':float(np.trace(H)),'value':value.tolist()});continue
        node.routing_weight.zero_();node.routing_bias.zero_()
        node.routing_weight[0,j]=-cfg.strength/2;node.routing_weight[1,j]=cfg.strength/2
        node.routing_bias[0]=cfg.strength*threshold/2;node.routing_bias[1]=-cfg.strength*threshold/2
        if cfg.readout=='leaf':node.value.zero_()
        if cfg.linear_values:
            prior=np.vstack([value,linear]) if cfg.readout=='residual' else np.zeros_like(beta)
        else:
            prior=value if cfg.readout=='residual' else np.zeros_like(value)
        queue.extend([(children[0],rows[bins[j][rows]<=q],prior),(children[1],rows[bins[j][rows]>q],prior)])
        splits.append({'node_id':key,'feature':int(j),'threshold':threshold,'gain':gain,'left_hessian_mass':hl,'right_hessian_mass':hr})
    if cfg.readout=='leaf':
        for node in tree.nodes.values():
            if not node.is_leaf:node.value.requires_grad_(False)
    return tree,{'splits':splits,'leaves':leaves,'features':columns.tolist(),'rows':int((w>0).sum()),'criterion':'full-Hessian histogram gain'}

@torch.no_grad()
def build_linear_model_tree(data, scores, task, tree_id, native, cfg, feature_mask, generator):
    """Greedy hard model-tree proposal with regularized affine logistic leaves.

    This is intentionally a first-stage initializer: sklearn's logistic solver has
    no per-example offset, so later additive stages should use ``build_tree``.
    """
    if task != 'binary' or not cfg.linear_values:
        raise ValueError('linear model-tree proposal currently requires binary linear values')
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    temporary=deepcopy(native);temporary.structure.dynamic=True;temporary.structure.initial_depth=0;temporary.structure.arity=2
    tree=RaggedTree(tree_id,data.x.shape[1],scores.shape[1],temporary,generator);tree.config=native;tree.feature_mask.copy_(feature_mask)
    x=data.x.numpy().astype(np.float64);y=data.y.numpy().astype(int);w=data.weight.numpy().astype(np.float64)
    columns=np.flatnonzero(feature_mask.numpy()); C=1/max(cfg.linear_l2,1e-8)
    base=float(np.asarray(scores.numpy())[:,0].mean())
    def fit(rows):
        if len(np.unique(y[rows]))<2:
            # Finite intercept for pure leaves, zero slopes.
            p=(w[rows]*y[rows]).sum()/max(w[rows].sum(),1e-12);p=np.clip(p,1e-5,1-1e-5)
            return np.r_[np.log(p/(1-p))-base,np.zeros(x.shape[1])]
        m=LogisticRegression(C=C,max_iter=200,solver='lbfgs').fit(x[rows],y[rows],sample_weight=w[rows])
        return np.r_[float(m.intercept_[0])-base,m.coef_[0]]
    def loss(rows,beta):
        z=base+beta[0]+x[rows]@beta[1:];p=1/(1+np.exp(-np.clip(z,-30,30)))
        return log_loss(y[rows],np.c_[1-p,p],labels=[0,1],sample_weight=w[rows],normalize=False)
    queue=[(tree.root_id,np.flatnonzero(w>0),np.zeros(x.shape[1]+1))];splits=[];leaves=[]
    while queue:
        key,rows,parent=queue.pop(0);node=tree.get(key);beta=fit(rows)
        local=beta-parent if cfg.readout=='residual' else beta
        node.value.copy_(torch.as_tensor([local[0]],dtype=node.value.dtype));node.linear_value.copy_(torch.as_tensor(local[1:,None],dtype=node.value.dtype))
        best=None;base_loss=loss(rows,beta)
        if node.depth<cfg.depth and len(rows)>=2*cfg.min_samples:
            for j in columns:
                cuts=np.unique(np.quantile(x[rows,j],np.linspace(.1,.9,max(2,min(cfg.bins,10)))))
                for th in cuts:
                    left=rows[x[rows,j]<=th];right=rows[x[rows,j]>th]
                    if len(left)<cfg.min_samples or len(right)<cfg.min_samples:continue
                    bl,br=fit(left),fit(right);gain=base_loss-loss(left,bl)-loss(right,br)-cfg.split_cost
                    if gain>1e-10 and (best is None or gain>best[0]):best=(gain,int(j),float(th),left,right,bl,br)
        if best is None:
            leaves.append({'node_id':key,'rows':len(rows),'value':[float(beta[0])],'linear_norm':float(np.linalg.norm(beta[1:]))});continue
        gain,j,threshold,left,right,bl,br=best;children=tree.grow(key,generator=generator,arity=2)
        node.routing_weight.zero_();node.routing_bias.zero_();node.routing_weight[0,j]=-cfg.strength/2;node.routing_weight[1,j]=cfg.strength/2
        node.routing_bias[0]=cfg.strength*threshold/2;node.routing_bias[1]=-cfg.strength*threshold/2
        queue.extend([(children[0],left,beta),(children[1],right,beta)])
        splits.append({'node_id':key,'feature':j,'threshold':threshold,'gain':float(gain)})
    return tree,{'splits':splits,'leaves':leaves,'features':columns.tolist(),'rows':int((w>0).sum()),'criterion':'regularized affine-logistic model-tree gain'}


def _binary_nll(y, score, weight):
    y=np.asarray(y,dtype=float);z=np.asarray(score,dtype=float).reshape(-1);w=np.asarray(weight,dtype=float)
    return float(np.sum(w*(np.logaddexp(0,z)-y*z)))

def _fit_grouped_gate(x, y, weight, base_score, rows, group, linear_l2, gate_l2,
                      starts=4, steps=40, seed=0):
    """Local two-expert soft model with a gate restricted to one semantic group.

    Experts are affine over the same group plus a bias. This is a proposal
    optimizer only; the native model subsequently refines the chosen gate.
    """
    import torch
    rr=np.asarray(rows,dtype=int);cols=np.asarray(group,dtype=int)
    xx=torch.as_tensor(x[rr][:,cols],dtype=torch.float64)
    yy=torch.as_tensor(np.asarray(y)[rr],dtype=torch.float64)
    ww=torch.as_tensor(np.asarray(weight)[rr],dtype=torch.float64)
    bb=torch.as_tensor(np.asarray(base_score)[rr].reshape(-1),dtype=torch.float64)
    best=None
    gen=torch.Generator().manual_seed(seed)
    d=len(cols)
    for start in range(starts):
        gate_w=torch.nn.Parameter(.05*torch.randn(d,dtype=torch.float64,generator=gen))
        gate_b=torch.nn.Parameter(torch.zeros((),dtype=torch.float64))
        e0=torch.nn.Parameter(torch.zeros(d+1,dtype=torch.float64))
        e1=torch.nn.Parameter(torch.zeros(d+1,dtype=torch.float64))
        opt=torch.optim.LBFGS([gate_w,gate_b,e0,e1],lr=.5,max_iter=steps,line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            q=torch.sigmoid(xx@gate_w+gate_b)
            a0=e0[0]+xx@e0[1:];a1=e1[0]+xx@e1[1:]
            score=bb+(1-q)*a0+q*a1
            loss=(ww*(torch.nn.functional.softplus(score)-yy*score)).sum()
            loss=loss+.5*linear_l2*(e0[1:].square().sum()+e1[1:].square().sum())+.5*gate_l2*gate_w.square().sum()
            loss.backward();return loss
        try: opt.step(closure)
        except RuntimeError: continue
        with torch.no_grad():
            q=torch.sigmoid(xx@gate_w+gate_b);a0=e0[0]+xx@e0[1:];a1=e1[0]+xx@e1[1:]
            score=bb+(1-q)*a0+q*a1
            obj=float((ww*(torch.nn.functional.softplus(score)-yy*score)).sum()+.5*linear_l2*(e0[1:].square().sum()+e1[1:].square().sum())+.5*gate_l2*gate_w.square().sum())
            candidate=(obj,gate_w.detach().numpy(),float(gate_b),e0.detach().numpy(),e1.detach().numpy())
            if best is None or obj<best[0]:best=candidate
    return best

@torch.no_grad()
def build_grouped_oblique_model_tree(data, scores, task, tree_id, native, cfg, feature_mask,
                                     generator, feature_groups, gate_l2=1., starts=4, steps=40):
    """Greedy semantic-group oblique model tree.

    Each node may rotate features only *within* one declared group. Groups may
    overlap. Features outside the selected group cannot enter that gate. This
    avoids indiscriminate rotations across semantically unrelated variables.
    """
    if task!='binary' or not cfg.linear_values:
        raise ValueError('grouped oblique model tree currently requires binary linear values')
    groups=[tuple(j for j in group if bool(feature_mask[j])) for group in feature_groups]
    groups=[g for g in groups if g]
    if not groups: raise ValueError('at least one nonempty feature group is required')
    temporary=deepcopy(native);temporary.structure.dynamic=True;temporary.structure.initial_depth=0;temporary.structure.arity=2
    tree=RaggedTree(tree_id,data.x.shape[1],scores.shape[1],temporary,generator);tree.config=native;tree.feature_mask.copy_(feature_mask)
    x=data.x.numpy().astype(np.float64);y=data.y.numpy().astype(int);w=data.weight.numpy().astype(np.float64);base=np.asarray(scores.numpy())[:,0]
    # Full affine node model is retained; semantic grouping constrains routing,
    # not the predictive linear packet. Feature penalties can sparsify packets.
    def fit(rows):
        return fit_binary_linear(x[rows],base[rows],y[rows],w[rows],cfg.linear_l2).reshape(-1)
    def loss(rows,beta):
        z=base[rows]+beta[0]+x[rows]@beta[1:]
        return _binary_nll(y[rows],z,w[rows])+.5*cfg.linear_l2*np.sum(beta[1:]**2)
    queue=[(tree.root_id,np.flatnonzero(w>0),np.zeros(x.shape[1]+1))];splits=[];leaves=[]
    seed=0
    while queue:
        key,rows,parent=queue.pop(0);node=tree.get(key);beta=fit(rows);local=beta-parent if cfg.readout=='residual' else beta
        node.value.copy_(torch.as_tensor([local[0]],dtype=node.value.dtype));node.linear_value.copy_(torch.as_tensor(local[1:,None],dtype=node.value.dtype))
        best=None;base_loss=loss(rows,beta)
        if node.depth<cfg.depth and len(rows)>=2*cfg.min_samples:
            for gi,group in enumerate(groups):
                cand=_fit_grouped_gate(x,y,w,base,rows,group,cfg.linear_l2,gate_l2,starts,steps,seed+gi+31*node.depth);seed+=1
                if cand is None:continue
                obj,gw,gb,e0,e1=cand;gain=base_loss-obj-cfg.split_cost
                if gain>1e-8 and (best is None or gain>best[0]):best=(gain,group,gw,gb,e0,e1)
        if best is None:
            leaves.append({'node_id':key,'rows':len(rows),'linear_norm':float(np.linalg.norm(beta[1:]))});continue
        gain,group,gw,gb,e0,e1=best
        # Hard assignment only for recursively constructing child proposals.
        q=1/(1+np.exp(-np.clip(x[rows][:,group]@gw+gb,-30,30)));left=rows[q<.5];right=rows[q>=.5]
        if len(left)<cfg.min_samples or len(right)<cfg.min_samples:
            leaves.append({'node_id':key,'rows':len(rows),'linear_norm':float(np.linalg.norm(beta[1:]))});continue
        children=tree.grow(key,generator=generator,arity=2);node.routing_weight.zero_();node.routing_bias.zero_()
        scale=cfg.strength/max(np.linalg.norm(gw),1e-8);full=np.zeros(x.shape[1]);full[list(group)]=gw*scale
        node.routing_weight[0].copy_(torch.as_tensor(-full/2,dtype=node.value.dtype));node.routing_weight[1].copy_(torch.as_tensor(full/2,dtype=node.value.dtype))
        node.routing_bias[0]=-cfg.strength*gb/(2*max(np.linalg.norm(gw),1e-8));node.routing_bias[1]=-node.routing_bias[0]
        queue.extend([(children[0],left,beta),(children[1],right,beta)])
        splits.append({'node_id':key,'group':list(group),'gain':float(gain),'gate_norm':float(np.linalg.norm(gw))})
    return tree,{'splits':splits,'leaves':leaves,'features':np.flatnonzero(feature_mask.numpy()).tolist(),'rows':int((w>0).sum()),'criterion':'semantic-group oblique two-expert gain'}