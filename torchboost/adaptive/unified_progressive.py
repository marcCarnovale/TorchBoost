"""Progressive additive learning on the native adaptive forest engine.

Reuses native observations, plasticity, circuit dynamics, structural transactions,
and optimizer migration. Frozen predictions may be cached, but never when doing
so would invalidate input-gradient constraints, dropout, or learned attention.
"""
from __future__ import annotations
from copy import deepcopy
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn, Tensor
from sklearn.base import BaseEstimator,ClassifierMixin,RegressorMixin
from sklearn.utils.validation import check_is_fitted
from .config import ForestConfig,StructureConfig,PhysicsConfig,ScheduleConfig
from .forest import AdaptiveForest,ForestTrace
from .training import JointTrainer,model_snapshot,restore_model
from .data import Preprocessor,DataSplit,sample_weights
from .objectives import Objective
from .newton_builder import BuilderConfig,build_tree,build_linear_model_tree,build_grouped_oblique_model_tree,derivatives
from .progressive_regularizers import Regularizers,penalties


def default_native():
    return ForestConfig(n_trees=1,aggregation='additive',residual_weights=False,execution='packed',
        collect_metrics=False,compact_history=True,batch_size=256,learning_rate=.01,weight_decay=1e-4,
        observation_every=2,control_sample_size=96,
        structure=StructureConfig(max_depth=4,max_nodes=511,dynamic=False,structural_gate=False,
            complexity=0.,gate_bimodality=0.,allocation_regularization=0.),
        physics=PhysicsConfig(initial_temperature=1.,ambient_temperature=1.))


@dataclass
class UnifiedConfig:
    n_trees: int = 32
    updates_per_stage: int = 16
    depth: int = 3
    bins: int = 24
    min_samples_leaf: int = 5
    min_child_weight: float = 1.
    newton_l2: float = 1.
    split_cost: float = 0.
    max_delta: float = 5.
    shrinkage: float = .2
    age_decay: float = .2
    active_window: int = 3
    row_subsample: float = 1.
    feature_subsample: float = 1.
    cart_strength: float = 8.
    warm_value_updates: int = 2
    gate_release: str = 'oblique'   # hard / threshold / oblique
    readout: str = 'residual'
    linear_values: bool = False
    linear_l2: float = 10.
    proposal_mode: str = 'hist_newton'  # hist_newton / linear_model_tree / grouped_oblique
    feature_groups: tuple[tuple[int,...], ...] = ()
    grouped_gate_l2: float = 1.
    grouped_gate_starts: int = 4
    grouped_gate_steps: int = 40
    head_mode: str = 'none'         # none / shared / specialized bounded modulation
    anchor_min_passes: float = .25
    anchor_min_updates: int = 2
    anchor_require_utility: bool = True
    anchor_at_birth: bool = False
    reopening_stages: int = 2
    refit_every: int = 0
    refit_damping: float = .02
    max_cache_bytes: int = 64*1024*1024
    checkpoint_every: int = 0  # optimizer updates; 0 keeps stage-boundary selection only
    random_state: int = 0
    regularizers: Regularizers = field(default_factory=Regularizers)
    regularizer_schedule: ScheduleConfig = field(default_factory=lambda:ScheduleConfig(low=1.,high=1.))
    native: ForestConfig = field(default_factory=default_native)

    def __post_init__(self):
        if isinstance(self.native,dict):self.native=ForestConfig(**self.native)
        if isinstance(self.regularizers,dict):self.regularizers=Regularizers(**self.regularizers)
        if isinstance(self.regularizer_schedule,dict):self.regularizer_schedule=ScheduleConfig(**self.regularizer_schedule)
        self.native.collect_metrics=bool(self.native.collect_metrics or self.native.physics.mode!='none' or self.native.plasticity.mode!='none' or self.native.online.enabled or self.native.structure.dynamic)
        self.native.__post_init__();self.regularizers.__post_init__();self.regularizer_schedule.__post_init__()
        for k in ('n_trees','updates_per_stage','bins','min_samples_leaf','active_window','reopening_stages','max_cache_bytes'):
            if not isinstance(getattr(self,k),int) or isinstance(getattr(self,k),bool) or getattr(self,k)<1:raise ValueError(f'invalid {k}')
        for k in ('depth','warm_value_updates','anchor_min_updates','refit_every','checkpoint_every'):
            if not isinstance(getattr(self,k),int) or getattr(self,k)<0:raise ValueError(f'invalid {k}')
        for k in ('min_child_weight','newton_l2','split_cost','anchor_min_passes','refit_damping'):
            if not math.isfinite(getattr(self,k)) or getattr(self,k)<0:raise ValueError(f'invalid {k}')
        for k in ('shrinkage','row_subsample','feature_subsample'):
            if not math.isfinite(getattr(self,k)) or not 0<getattr(self,k)<=1:raise ValueError(f'invalid {k}')
        for k in ('cart_strength','max_delta'):
            if not math.isfinite(getattr(self,k)) or getattr(self,k)<=0:raise ValueError(f'invalid {k}')
        if not math.isfinite(self.age_decay) or not 0<=self.age_decay<=1:raise ValueError('invalid age_decay')
        if self.depth>self.native.structure.max_depth:raise ValueError('proposal depth exceeds native budget')
        self.native.node_linear_values=self.linear_values
        if self.readout not in ('leaf','residual') or self.gate_release not in ('hard','threshold','oblique'):raise ValueError('invalid tree mode')
        if self.readout=='leaf' and self.native.structure.dynamic:raise ValueError('native dynamic growth requires residual readout')
        if self.native.aggregation!='additive' or self.native.residual_weights:raise ValueError('explicit additive scores and stage rates required')
        if self.proposal_mode not in ('hist_newton','linear_model_tree','grouped_oblique'):raise ValueError('invalid proposal_mode')
        if self.proposal_mode in ('linear_model_tree','grouped_oblique') and not self.linear_values:raise ValueError('linear_model_tree requires linear_values')
        if self.proposal_mode=='grouped_oblique' and not self.feature_groups:
            raise ValueError('grouped_oblique requires feature_groups')
        for group in self.feature_groups:
            if not group or min(group)<0 or len(set(group))!=len(group): raise ValueError('invalid feature group')
        if self.grouped_gate_l2<0 or self.grouped_gate_starts<1 or self.grouped_gate_steps<1: raise ValueError('invalid grouped gate settings')
        if self.head_mode not in ('none','shared','specialized'):raise ValueError('invalid head_mode')
        if self.head_mode!='none' and (self.native.interaction_groups or self.age_decay==0):raise ValueError('learned heads incompatible with strict frozen contributions or hard score-interaction groups')
        if self.refit_every and self.head_mode!='none':raise ValueError('conditional refit requires fixed heads')
        if self.native.device!='cpu':raise ValueError('this implementation verifies CPU, not cross-device continuation')


class ProgressiveForest(AdaptiveForest):
    def __init__(self,input_dim,output_dim,config,*,schema=None):
        empty=schema is None or not schema['trees']
        if empty:
            temp=deepcopy(config);temp.n_trees=1;temp.structure.dynamic=True;temp.structure.initial_depth=0
            super().__init__(input_dim,output_dim,temp)
            self.config=config;self.trees=nn.ModuleList()
            for name,shape in [('attention_weight',(0,1,input_dim)),('attention_bias',(0,1)),('residual_logits',(0,1))]:
                setattr(self,name,nn.Parameter(torch.empty(shape),requires_grad=False))
        else:super().__init__(input_dim,output_dim,config,schema=schema)
        self.modulation=(schema or {}).get('modulation','none')
        self.head_count=output_dim if self.modulation=='specialized' else 1
        self.attention_weight=nn.Parameter(torch.zeros(len(self.trees),self.head_count,input_dim),requires_grad=self.modulation!='none')
        self.attention_bias=nn.Parameter(torch.zeros(len(self.trees),self.head_count),requires_grad=self.modulation!='none')
        self.bias.requires_grad_(False)
        self.register_buffer('stage_rates',torch.tensor((schema or {}).get('rates',[]),dtype=self.bias.dtype))
        for t in self.trees:t.force_hard=t.tree_id in (schema or {}).get('hard_trees',[])
        if schema is not None:self.version_offset=schema['version_offset']

    def coefficient_transform(self,x,coef):
        coef=coef*self.stage_rates[None,:,None]
        if self.modulation!='none':
            a=1+torch.tanh(torch.einsum('nd,thd->nth',x,self.attention_weight)+self.attention_bias)
            coef=coef*a
        return coef

    def forward(self,x,**kwargs):
        if len(self.trees):return super().forward(x,**kwargs)
        out=self.bias.expand(len(x),-1)
        if kwargs.get('trace',False):return out,ForestTrace({},x.new_empty(len(x),0,self.output_dim),x.new_empty(len(x),0,self.output_dim),{})
        return out

    def append(self,tree,rate,optimizer):
        n=len(self.trees);self.trees.append(tree)
        self.stage_rates=torch.cat((self.stage_rates,self.bias.new_tensor([rate])))
        h=self.output_dim if self.modulation=='specialized' else 1;self.head_count=h
        for name,tail in [('attention_weight',(h,self.input_dim)),('attention_bias',(h,)),('residual_logits',(1,))]:
            old=getattr(self,name);value=self.bias.new_zeros((n+1,)+tail)
            if old.numel():value[:n]=old.detach()
            new=nn.Parameter(value,requires_grad=self.modulation!='none' and name!='residual_logits');setattr(self,name,new)
            if old in optimizer.state:
                prior=optimizer.state.pop(old);migrated={}
                for k,v in prior.items():
                    if isinstance(v,Tensor) and v.shape==old.shape:
                        q=torch.zeros_like(new);q[:n]=v;migrated[k]=q
                    else:migrated[k]=deepcopy(v)
                optimizer.state[new]=migrated
        self.version_offset+=1

    def remove_tree(self,tree_id):
        keep=[i for i,t in enumerate(self.trees) if t.tree_id!=tree_id]
        removed,migrations=super().remove_tree(tree_id)
        if removed:self.stage_rates=self.stage_rates[keep].clone()
        return removed,migrations

    def schema(self):
        return {**super().schema(),'family':'unified-progressive-v1','rates':self.stage_rates.tolist(),
            'modulation':self.modulation,'hard_trees':[t.tree_id for t in self.trees if getattr(t,'force_hard',False)]}


class UnifiedTrainer(JointTrainer):
    def __init__(self,model,objective,cfg):
        self.pcfg=cfg;self.stage=0;self.tick=0;self.age_frozen=set();self.reopened={}
        self.exposures={};self.updates={};self.admitted=set();self.proposal_history=[];self.sampling_history=[];self.term_totals={};self.refit_history=[]
        self.row_rng=torch.Generator().manual_seed(cfg.random_state+1001)
        self.feature_rng=torch.Generator().manual_seed(cfg.random_state+1013)
        self.cache=OrderedDict();self.fixed_cache=None;self.cache_hits=0
        native=deepcopy(cfg.native);native.n_trees=cfg.n_trees;native.epochs=cfg.n_trees*cfg.updates_per_stage
        native.__post_init__();model.config=native
        super().__init__(model,objective,native,torch.Generator().manual_seed(cfg.random_state+1027))

    def _synchronize(self,changed_nodes=None):
        if not self.model.trees:self.optimizer.synchronize(self.model);return
        super()._synchronize(changed_nodes)
        for key,state in self.plastic.states.items():
            if key not in self.admitted:state['admitted']=False
        self.fixed_cache=None

    def _regularization(self,x,pred,trace,w,values):
        base=super()._regularization(x,pred,trace,w,values)
        terms=penalties(self.model,w,trace,self.pcfg.regularizers)
        factor=self.pcfg.regularizer_schedule.value(self.tick/max(1,self.config.epochs-1))
        for name,value in terms.items():self.term_totals[name]=self.term_totals.get(name,0.)+float(value.detach())*factor
        self.term_totals['native']=self.term_totals.get('native',0.)+float(base.detach())
        return base+factor*sum(terms.values())

    def _set_active(self):
        ids=[t.tree_id for t in self.model.trees]
        width=1 if self.pcfg.age_decay==0 else self.pcfg.active_window
        active=set(ids[-width:]);active.update(i for i,end in self.reopened.items() if end>=self.stage)
        protected=math.ceil(self.config.n_trees*self.config.structure.protect_tree_fraction)
        active.update(i for i in ids if i<protected)
        for n in self.model.iter_nodes():
            if n.tree_id not in active and not n.frozen:n.set_frozen(True);self.age_frozen.add(n.node_id)
            elif n.tree_id in active and n.node_id in self.age_frozen:n.set_frozen(False);self.age_frozen.discard(n.node_id)
            if self.pcfg.readout=='leaf' and not n.is_leaf:n.value.requires_grad_(False)

    def _set_rates(self,values):
        rate=values.get('learning_rate',self.config.learning_rate)
        self.optimizer.set_controls(rate,self.physical.nodes)
        newest=max(t.tree_id for t in self.model.trees);nodes=self.model.node_map()
        for group in self.optimizer.optimizer.param_groups:
            node=nodes.get(group['owner'])
            if node is None:
                group['lr']=rate if self.model.modulation!='none' else 0.
                # Learned depth allocation is native global state, not the bias.
                if self.pcfg.regularizers.allocation=='learned' or self.config.structure.depth_allocation=='learned':group['lr']=rate
            else:
                factor=self.pcfg.age_decay**(newest-node.tree_id)
                if self.reopened.get(node.tree_id,-1)>=self.stage:factor=max(factor,.1)
                group['lr']*=factor if not node.frozen else 0.
        self.plastic.stiffness_multiplier=values.get('plastic_stiffness',1.)

    def _admit(self):
        if self.config.plasticity.mode=='none':return
        obs=self.tracker.latest()
        for n in self.model.iter_nodes():
            if n.node_id in self.admitted:continue
            o=obs.get(n.node_id)
            coverage=self.exposures.get(n.tree_id,0)/self.ntrain>=self.pcfg.anchor_min_passes
            moved=self.updates.get(n.tree_id,0)>=self.pcfg.anchor_min_updates
            useful=o is not None and o.utility>self.config.plasticity.minimum_utility and o.occupancy>=self.config.plasticity.minimum_occupancy
            if self.pcfg.anchor_at_birth or (coverage and moved and (useful or not self.pcfg.anchor_require_utility)):
                state=self.plastic.states[n.node_id]
                for name,p in n.parameters_for_plasticity().items():
                    state['anchors'][name]=p.detach().clone();state['candidate'][name]=p.detach().clone()
                state['admitted']=True;state['evidence']=0.;self.admitted.add(n.node_id)
                self.events.append({'event':'anchor_admitted','step':self.tick,'node_id':n.node_id,'exposures':self.exposures.get(n.tree_id,0),'explicit_import':self.pcfg.anchor_at_birth})

    def _observe_and_control(self,control,step):
        n=len(self.events);super()._observe_and_control(control,step)
        for e in self.events[n:]:
            if e['event']=='thermal_thaw':self.reopened[int(e['node_id'].split(':')[0])]=self.stage+self.pcfg.reopening_stages-1
        self._admit();self._set_active();self.fixed_cache=None

    @torch.no_grad()
    def _cache_fixed(self,train):
        c=self.config;r=self.pcfg.regularizers
        if self.model.modulation!='none' or self.model.feature_dropout or self.model.tree_dropout or c.diversity or c.monotonicity_penalty or r.tree_l2 or r.route_balance or r.child_penalty:
            self.fixed_cache=None;return
        self.model.eval();frozen=set();base=self.model.bias.expand(len(train.x),-1).clone()
        live={t.tree_id for t in self.model.trees}
        for k in list(self.cache):
            if k not in live:del self.cache[k]
        for i,t in enumerate(self.model.trees):
            if not all(n.frozen for n in t.nodes.values()):continue
            frozen.add(i)
            key=(t.topology_version,getattr(t,'force_hard',False),tuple((id(p),p._version) for p in t.parameters()),tuple((id(b),b._version) for b in t.buffers()),tuple(n.active for n in t.nodes.values()))
            previous=self.cache.get(t.tree_id)
            if previous is not None and previous[0]==key:pred=previous[1];self.cache_hits+=1
            else:
                pred=torch.cat([t(b,hard=getattr(t,'force_hard',False))[0] for b in train.x.split(2048)])
                if pred.numel()*pred.element_size()<=self.pcfg.max_cache_bytes:
                    self.cache[t.tree_id]=(key,pred)
                    while sum(v[1].numel()*v[1].element_size() for v in self.cache.values())>self.pcfg.max_cache_bytes:self.cache.popitem(last=False)
            base+=self.model.stage_rates[i]*pred
        self.fixed_cache=(base,frozen)

    def _update(self,train,pool,values,new_id,within):
        ids=pool[torch.randint(len(pool),(min(len(pool),self.config.batch_size),),generator=self.generator)]
        x,y,w=train.x[ids],train.y[ids],train.weight[ids]
        if w.sum()<=0:return
        if self.config.monotonicity_penalty:x=x.detach().requires_grad_(True)
        self.model.train();self.optimizer.zero_grad()
        r=self.pcfg.regularizers;need=bool(self.config.diversity or r.tree_l2 or r.route_balance or r.child_penalty)
        if self.fixed_cache is None:
            out=self.model(x,trace=need,generator=self.generator);z,tr=out if need else (out,None)
        else:
            base,frozen=self.fixed_cache;z=base[ids];tr=None
            for i,t in enumerate(self.model.trees):
                if i not in frozen:z=z+self.model.stage_rates[i]*t(x,hard=getattr(t,'force_hard',False))[0]
        loss=self.objective.weighted_loss(z,y,w)+self._regularization(x,z,tr,w,values)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite progressive loss')
        if not loss.requires_grad:return
        loss.backward()
        for n in self.model.iter_nodes():
            if n.routing_weight is None:continue
            hold=self.pcfg.gate_release=='hard' or (n.tree_id==new_id and within<self.pcfg.warm_value_updates)
            if hold:n.routing_weight.grad=None;n.routing_bias.grad=None
            elif self.pcfg.gate_release=='threshold':n.routing_weight.grad=None
        self.last_gradient_norm=float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config.gradient_clip,error_if_nonfinite=True))
        before=self.collector.before_step(self.model) if self.config.collect_metrics else None
        self.optimizer.step();self.model.project();self.optimizer_steps+=1;self.examples_seen+=len(ids)
        for t in self.model.trees:
            if any(g['lr']>0 and str(g['owner']).startswith(f'{t.tree_id}:') and any(p.grad is not None for p in g['params']) for g in self.optimizer.optimizer.param_groups):
                self.exposures[t.tree_id]=self.exposures.get(t.tree_id,0)+len(ids);self.updates[t.tree_id]=self.updates.get(t.tree_id,0)+1
        if self.config.collect_metrics:self.collector.after_step(self.model,before)
        self.max_optimizer_bytes=max(self.max_optimizer_bytes,self.optimizer.tensor_bytes())

    @torch.no_grad()
    def evaluate(self,train,selection,phase):
        score=self._selection_score(selection)
        if score<self.best_score:self.best_score=score;self.best_epoch=self.tick;self.best_snapshot=model_snapshot(self.model)
        self.history.append({'stage':self.stage,'step':self.tick,'phase':phase,'trees':len(self.model.trees),
            'selection_loss':score,'train_loss':self.loss(train),'nodes':len(self.model.node_map()),
            'parameters':sum(p.numel() for p in self.model.parameters()),'optimizer_updates':self.optimizer_steps,
            'examples_seen':self.examples_seen,'model_bytes':self.model.tensor_bytes(),'optimizer_bytes':self.optimizer.tensor_bytes(),
            'cache_hits':self.cache_hits,'penalties':self.term_totals.copy(),'admitted_anchors':len(self.admitted),
            'event_counts':{name:sum(e.get('event')==name for e in self.events) for name in set(e.get('event') for e in self.events)},
            'mean_temperature':float(np.mean([st['temperature'] for st in self.physical.nodes.values()])) if self.physical.nodes else float(self.config.physics.ambient_temperature),
            'mean_charge':float(np.mean([abs(st.get('charge',0.)) for st in self.physical.nodes.values()])) if self.physical.nodes else 0.,
            'cumulative_injection':float(sum(h.get('injected_charge',0.) for h in self.physical.history)),
            'max_temperature_seen':float(max([max((v.get('temperature',self.config.physics.ambient_temperature) for v in h.get('nodes',{}).values()),default=self.config.physics.ambient_temperature) for h in self.physical.history],default=self.config.physics.ambient_temperature))})
        return score

    def run(self,train,control,selection,stop_stages=None):
        fp={k:v.fingerprint() for k,v in [('train',train),('control',control),('selection',selection)]}
        if self.fingerprints and self.fingerprints!=fp:raise ValueError('resume fingerprints differ')
        self.fingerprints=fp;self.ntrain=len(train.x)
        if self.control_indices is None:
            gen=torch.Generator().manual_seed(self.pcfg.random_state+1039)
            self.control_indices=torch.randperm(len(control.x),generator=gen)[:self.config.control_sample_size]
        ci=self.control_indices;ctl=DataSplit(control.x[ci],control.y[ci],control.weight[ci])
        if self.stage==0:self.evaluate(train,selection,'intercept')
        end=self.pcfg.n_trees if stop_stages is None else min(stop_stages,self.pcfg.n_trees)
        while self.stage<end:
            pc=self.pcfg;tree_id=self.stage
            pool=torch.randperm(len(train.x),generator=self.row_rng)[:max(1,math.ceil(len(train.x)*pc.row_subsample))]
            allowed=torch.arange(self.model.input_dim)
            if self.config.interaction_groups:allowed=torch.tensor(self.config.interaction_groups[tree_id%len(self.config.interaction_groups)])
            subset=allowed[torch.randperm(len(allowed),generator=self.feature_rng)[:max(1,math.ceil(len(allowed)*pc.feature_subsample))]]
            mask=torch.zeros(self.model.input_dim);mask[subset]=1.
            self.sampling_history.append({'stage':tree_id,'rows':pool.tolist(),'features':subset.tolist()})
            ds=DataSplit(train.x[pool],train.y[pool],train.weight[pool]);scores=self.logits(ds)
            bcfg=BuilderConfig(pc.depth,pc.bins,pc.min_samples_leaf,pc.min_child_weight,pc.newton_l2,pc.split_cost,pc.max_delta,pc.cart_strength,pc.readout,pc.linear_values,pc.linear_l2)
            if pc.proposal_mode=='grouped_oblique' and tree_id==0:
                tree,record=build_grouped_oblique_model_tree(ds,scores,self.objective.task,tree_id,self.config,bcfg,mask,self.generator,pc.feature_groups,pc.grouped_gate_l2,pc.grouped_gate_starts,pc.grouped_gate_steps)
            else:
                builder=build_linear_model_tree if pc.proposal_mode=='linear_model_tree' and tree_id==0 else build_tree
                tree,record=builder(ds,scores,self.objective.task,tree_id,self.config,bcfg,mask,self.generator)
            tree.force_hard=pc.warm_value_updates>0 or pc.gate_release=='hard'
            with torch.no_grad():
                previous=self.logits(train);direction=torch.cat([tree(b,hard=tree.force_hard)[0] for b in train.x.split(2048)])
                before=float(self.objective.weighted_loss(previous,train.y,train.weight));rate=pc.shrinkage;accepted=False
                for _ in range(10):
                    after=float(self.objective.weighted_loss(previous+rate*direction,train.y,train.weight))
                    if after+pc.regularizers.correction_cost*len(tree.nodes)<before-1e-10:accepted=True;break
                    rate*=.5
            self.stage+=1;record.update(stage=tree_id,accepted=accepted,rate=rate,loss_before=before,loss_after=after);self.proposal_history.append(record)
            if not accepted:self.evaluate(train,selection,'rejected');continue
            self.model.append(tree,rate,self.optimizer.optimizer);self._synchronize();self._set_active();self._admit()
            self.evaluate(train,selection,'proposal');self._cache_fixed(train)
            for u in range(pc.updates_per_stage):
                if u==pc.warm_value_updates and pc.gate_release!='hard':tree.force_hard=False;self.fixed_cache=None
                values=self.schedule.apply(self.model,self.tick);self._apply_online(self.tick)
                self._set_active();self._set_rates(values)
                if self.fixed_cache is None:self._cache_fixed(train)
                self._update(train,pool,values,tree_id,u)
                if self.config.collect_metrics and self.tick%self.config.observation_every==0:self._observe_and_control(ctl,self.tick)
                self.tick+=1;self.epoch=self.tick
                if pc.checkpoint_every and self.tick%pc.checkpoint_every==0:
                    self.evaluate(train,selection,'within_stage')
            if pc.refit_every and self.stage%pc.refit_every==0:self.refit(train,tree_id)
            self.evaluate(train,selection,'refined')
        self.model.eval()

    @torch.no_grad()
    def refit(self,train,tree_id):
        tree=self.model.get_tree(tree_id);nodes=[n for n in tree.nodes.values() if n.value.requires_grad]
        k=self.model.output_dim
        if len(nodes)*k>1024:raise ValueError('dense conditional refit limited to 1024 coordinates')
        if not nodes:return
        slot=[t.tree_id for t in self.model.trees].index(tree_id);rate=self.model.stage_rates[slot]
        self.model.eval();score=self.logits(train);_,trace=tree(train.x,hard=getattr(tree,'force_hard',False),trace=True)
        basis=torch.stack([trace[n.node_id].reach for n in nodes],1).double()*rate
        g,h=derivatives(score.numpy().astype(float),train.y.numpy(),self.objective.task)
        w=train.weight.double()/train.weight.sum().double();gd=torch.from_numpy(g);hd=torch.from_numpy(h)
        H=torch.einsum('ia,ib,icd,i->acbd',basis,basis,hd,w).reshape(len(nodes)*k,-1)
        gradient=torch.einsum('ia,ic,i->ac',basis,gd,w).reshape(-1)
        H+=self.pcfg.refit_damping*torch.eye(len(gradient),dtype=torch.double)
        with torch.enable_grad():
            x=train.x.detach().requires_grad_(bool(self.config.monotonicity_penalty));z,tr=self.model(x,trace=True)
            reg=self._regularization(x,z,tr,train.weight,{})
            if reg.requires_grad:
                rg=torch.autograd.grad(reg,[n.value for n in nodes],allow_unused=True)
                gradient+=torch.cat([torch.zeros_like(n.value).reshape(-1) if q is None else q.detach().reshape(-1) for n,q in zip(nodes,rg)]).double()
        delta=torch.linalg.solve(H,-gradient).reshape(len(nodes),k);saved=[n.value.clone() for n in nodes]
        def value():
            with torch.enable_grad():
                x=train.x.detach().requires_grad_(bool(self.config.monotonicity_penalty));z,tr=self.model(x,trace=True)
                return float((self.objective.weighted_loss(z,train.y,train.weight)+self._regularization(x,z,tr,train.weight,{})).detach())
        before=value();step=1.;accepted=False
        for _ in range(12):
            for n,p,q in zip(nodes,saved,delta):n.value.copy_(p+step*q.to(p))
            after=value()
            if math.isfinite(after) and after<before:accepted=True;break
            step*=.5
        if not accepted:
            for n,p in zip(nodes,saved):n.value.copy_(p)
        else:
            for n in nodes:self.optimizer.optimizer.state.pop(n.value,None)
        self.refit_history.append({'step':self.tick,'accepted':accepted,'objective_before':before,'objective_after':value(),'step_size':step,'solve_residual':float((H@delta.reshape(-1)+gradient).abs().max())})
        self.fixed_cache=None

    def state_dict(self):
        out=super().state_dict()
        out['progressive']={k:deepcopy(getattr(self,k)) for k in ('stage','tick','age_frozen','reopened','exposures','updates','admitted','proposal_history','sampling_history','term_totals','refit_history','cache_hits')}
        out['progressive'].update(row_rng=self.row_rng.get_state(),feature_rng=self.feature_rng.get_state());return out

    def load_state_dict(self,state):
        super().load_state_dict(state);p=deepcopy(state['progressive'])
        self.row_rng.set_state(p.pop('row_rng'));self.feature_rng.set_state(p.pop('feature_rng'))
        for key,value in p.items():setattr(self,key,value)
        self.fixed_cache=None;self.cache.clear()


class _Estimator(BaseEstimator):
    classification=False
    def __init__(self,config=None):self.config=config
    def fit(self,X,y,sample_weight=None,*,control_set=None,eval_set=None,stop_stages=None):
        cfg=deepcopy(self.config or UnifiedConfig());cfg.__post_init__();self.config_=cfg
        if eval_set is None:raise ValueError('explicit selection set required')
        if cfg.native.collect_metrics and control_set is None:raise ValueError('adaptive training requires separate control data')
        self.preprocessor_=Preprocessor();w=sample_weights(sample_weight,len(X))
        self.preprocessor_.fit(X,y,classification=self.classification,weights=w)
        train=self.preprocessor_.split(X,y,w);selection=self.preprocessor_.split(*eval_set)
        control=selection if control_set is None else self.preprocessor_.split(*control_set)
        if cfg.native.collect_metrics and control.fingerprint()==selection.fingerprint():raise ValueError('control and selection data must differ')
        self.n_features_in_=train.x.shape[1];self.objective_=Objective(self.preprocessor_.task,self.preprocessor_.output_dim)
        if self.classification:self.classes_=self.preprocessor_.classes.copy()
        model=ProgressiveForest(self.n_features_in_,self.objective_.output_dim,deepcopy(cfg.native));model.modulation=cfg.head_mode
        with torch.no_grad():
            if self.classification:
                counts=torch.bincount(train.y,weights=train.weight,minlength=len(self.classes_));p=(counts/counts.sum()).clamp_min(1e-10)
                model.bias.copy_((p[1]/p[0]).log().reshape(1) if len(p)==2 else p.log())
            else:model.bias.copy_((train.y*train.weight[:,None]).sum(0)/train.weight.sum())
        self.trainer_=UnifiedTrainer(model,self.objective_,cfg)
        try:self.trainer_.run(train,control,selection,stop_stages)
        finally:self.trainer_.close()
        self._select();return self
    def _select(self):
        self.model_=restore_model(self.trainer_.best_snapshot,self.n_features_in_,self.objective_.output_dim,self.trainer_.config);self.model_.eval()
        self.n_estimators_=len(self.model_.trees);self.best_score_=self.trainer_.best_score;self.history_=self.trainer_.history
    def continue_fit(self,X,y,sample_weight=None,*,control_set,eval_set,stop_stages=None,allow_domain_shift=False):
        check_is_fitted(self,'trainer_')
        if allow_domain_shift:
            self.trainer_.fingerprints={}
            self.trainer_.control_indices=None
        try:self.trainer_.run(self.preprocessor_.split(X,y,sample_weight),self.preprocessor_.split(*control_set),self.preprocessor_.split(*eval_set),stop_stages)
        finally:self.trainer_.close()
        self._select();return self
    @torch.no_grad()
    def decision_function(self,X,*,last=False):
        check_is_fitted(self,'model_');model=self.trainer_.model if last else self.model_;model.eval()
        x=self.preprocessor_.transform_x(X);return torch.cat([model(b) for b in x.split(2048)]).numpy()
    def predict(self,X):
        if self.classification:return self.classes_[self.predict_proba(X).argmax(1)]
        out=self.preprocessor_.inverse_target(self.decision_function(X));return out[:,0] if out.shape[1]==1 else out
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        state={'schema':'unified-v1','classification':self.classification,'config':asdict(self.config_),
            'preprocessor':self.preprocessor_.state_dict(),'trainer':self.trainer_.state_dict()}
        tmp=path.with_suffix(path.suffix+'.tmp');torch.save(state,tmp);tmp.replace(path)
    @classmethod
    def load(cls,path):
        state=torch.load(path,map_location='cpu',weights_only=False)  # trusted research checkpoints only
        if state['schema']!='unified-v1' or state['classification']!=cls.classification:raise ValueError('checkpoint family mismatch')
        obj=cls(UnifiedConfig(**state['config']));obj.config_=obj.config
        obj.preprocessor_=Preprocessor();obj.preprocessor_.load_state_dict(state['preprocessor'])
        obj.n_features_in_=len(obj.preprocessor_.mean);obj.objective_=Objective(obj.preprocessor_.task,obj.preprocessor_.output_dim)
        if obj.classification:obj.classes_=obj.preprocessor_.classes.copy()
        model=ProgressiveForest(obj.n_features_in_,obj.objective_.output_dim,deepcopy(obj.config_.native))
        obj.trainer_=UnifiedTrainer(model,obj.objective_,obj.config_);obj.trainer_.load_state_dict(state['trainer']);obj._select();return obj

class UnifiedProgressiveClassifier(ClassifierMixin,_Estimator):
    classification=True
    def predict_proba(self,X,*,last=False):return self.objective_.response(torch.from_numpy(self.decision_function(X,last=last))).numpy()

class UnifiedProgressiveRegressor(RegressorMixin,_Estimator):
    pass