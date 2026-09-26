"""Step-budget training in the native engine, plus exact predictor migration.

Control time is measured in fixed-size optimizer-update blocks, not data epochs:
increasing N does not silently slow down/speed up the control/annealing clock.
No audit set is accepted by the training API. Data epochs and exposures are logged.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import math,time
import numpy as np
import torch
from .config import ForestConfig
from .data import DataSplit
from .objectives import Objective
from .training import JointTrainer, model_snapshot, restore_model, packet_map


class EpochSampler:
    """Without-replacement passes with state independent of model/control RNG."""
    def __init__(self,n,seed):
        if n<1: raise ValueError('n must be positive')
        self.n=n;self.generator=torch.Generator().manual_seed(seed)
        self.order=torch.empty(0,dtype=torch.long);self.position=0;self.exposures=0
    def take(self,count):
        if count<1: raise ValueError('count must be positive')
        pieces=[]
        while count:
            if self.position==len(self.order):
                self.order=torch.randperm(self.n,generator=self.generator);self.position=0
            k=min(count,self.n-self.position)
            pieces.append(self.order[self.position:self.position+k]);self.position+=k;count-=k;self.exposures+=k
        return torch.cat(pieces)
    def state_dict(self):
        return {'n':self.n,'generator':self.generator.get_state(),'order':self.order.clone(),
                'position':self.position,'exposures':self.exposures}
    def load_state_dict(self,state):
        if state['n']!=self.n:raise ValueError('sampler size mismatch')
        self.generator.set_state(state['generator']);self.order=state['order'].clone()
        self.position=state['position'];self.exposures=state['exposures']


class StepBudgetTrainer(JointTrainer):
    def __init__(self,*args,updates_per_block=32,sampler_seed=7919,warmup_blocks=0,**kwargs):
        super().__init__(*args,**kwargs)
        if updates_per_block<1:raise ValueError('updates_per_block must be positive')
        self.updates_per_block=int(updates_per_block);self.sampler_seed=sampler_seed;self.sampler=None
        self.probe_indices=None
        self.warmup_blocks=int(warmup_blocks)
        if self.warmup_blocks<0:raise ValueError("warmup_blocks must be nonnegative")
        self.memory_initialized=False
    def _summary(self):
        phy=self.physical.history
        plastic=self.plastic.events
        nodes=list(self.model.iter_nodes())
        return {'charge_events':sum(x['injected_charge']>0 for x in phy),
            'injected_charge':sum(x['injected_charge'] for x in phy),
            'heat':sum(x['resistor_heat'] for x in phy),
            'max_energy_error':max([abs(x['energy_error']) for x in phy],default=0.),
            'fresh_flow_events':sum(x.get('flow_fraction',0)>0 for x in plastic),
            'damage_events':sum(x.get('damage',0)>0 for x in plastic),
            'blocked_release':sum(x.get('blocked_release',False) for x in plastic),
            'reference_motion':sum(x.get('anchor_motion',0) for x in plastic),
            'consolidated_nodes':sum(x.get('consolidated',False) for x in self.plastic.states.values()),
            'broken_nodes':sum(x.get('integrity',1)==0 for x in self.plastic.states.values()),
            'temperature_mean':float(np.mean([float(n.temperature) for n in nodes])),
            'nodes':len(nodes),'frozen_nodes':sum(n.frozen for n in nodes),
            'parameters':sum(p.numel() for p in self.model.parameters()),
            'topology_version':self.model.topology_version,
            'cancelled_trials':len(self.tracker.cancelled)}
    def fit_steps(self,train:DataSplit,control:DataSplit,selection:DataSplit,*,blocks=None):
        cfg=self.config
        fp={k:v.fingerprint() for k,v in [('train',train),('control',control),('selection',selection)]}
        if self.fingerprints and self.fingerprints!=fp:raise ValueError('resume data changed')
        self.fingerprints=fp
        if self.sampler is None:self.sampler=EpochSampler(len(train.x),self.sampler_seed)
        if self.control_indices is None:
            # Same fixed panel regardless of architecture-dependent RNG consumption.
            gen=torch.Generator().manual_seed(self.sampler_seed+1)
            self.control_indices=torch.randperm(len(control.x),generator=gen)[:cfg.control_sample_size]
            self.probe_indices=torch.randperm(len(train.x),generator=gen)[:min(1024,len(train.x))]
        def subset(d,i):return DataSplit(d.x[i],d.y[i],d.weight[i])
        ctl=subset(control,self.control_indices);probe=subset(train,self.probe_indices)
        if self.epoch==0:
            self.best_score=self._selection_score(selection);self.best_snapshot=model_snapshot(self.model)
        end=cfg.epochs if blocks is None else min(int(blocks),cfg.epochs)
        for block in range(self.epoch,end):
            start=time.perf_counter()
            values=self.schedule.apply(self.model,block);self._apply_online(block)
            self.optimizer.set_controls(values.get('learning_rate',cfg.learning_rate),self.physical.nodes)
            self.plastic.stiffness_multiplier=(values.get('plastic_stiffness',1.) if block>=self.warmup_blocks else 0.)
            if block>=self.warmup_blocks and self.warmup_blocks and not self.memory_initialized:
                seed_learned_anchors(self)
                self.plastic.last_step=block-1
                for state in self.plastic.states.values():state['last_step']=block-1
                self.memory_initialized=True
            for _ in range(self.updates_per_block):
                idx=self.sampler.take(cfg.batch_size)
                # Reuse native objective, regularization, collector and optimizer.
                # A within-batch permutation does not change deterministic mean loss.
                self._train_epoch(subset(train,idx),values)
            if block>=self.warmup_blocks:self._observe_and_control(ctl,block)
            pred=self.logits(probe);train_loss=float(self.objective.weighted_loss(pred,probe.y,probe.weight))
            score=self._selection_score(selection)
            if score<self.best_score:
                self.best_score=score;self.best_epoch=block;self.best_snapshot=model_snapshot(self.model)
            rec={'block':block+1,'updates':self.optimizer_steps,'examples_seen':self.examples_seen,
                 'effective_epochs':self.examples_seen/len(train.x),'train_probe_loss':train_loss,
                 'selection_loss':score,'control_loss':self.loss(ctl),'best_block':self.best_epoch+1,
                 'learning_rate':self.optimizer.optimizer.param_groups[0]['lr'],**self._summary()}
            if self.objective.task!='regression':
                response=self.objective.response(pred);rec['train_probe_error']=float((response.argmax(1)!=probe.y).float().mean())
            rec['seconds']=time.perf_counter()-start
            self.history.append(rec);self.epoch=block+1
        self.model.eval();return self
    def selected_model(self):return restore_model(self.best_snapshot,self.model.input_dim,self.model.output_dim,self.config).eval()
    def state_dict(self):
        return {**super().state_dict(),'step_budget':{'sampler':None if self.sampler is None else self.sampler.state_dict(),
                'probe_indices':self.probe_indices,'updates_per_block':self.updates_per_block,'sampler_seed':self.sampler_seed,
                'warmup_blocks':self.warmup_blocks,'memory_initialized':self.memory_initialized}}
    def load_state_dict(self,state):
        extra=state['step_budget']
        if extra['updates_per_block']!=self.updates_per_block:raise ValueError('control clock changed')
        super().load_state_dict(state)
        self.sampler_seed=extra['sampler_seed'];self.probe_indices=extra['probe_indices']
        if extra['warmup_blocks']!=self.warmup_blocks:raise ValueError('warmup schedule changed')
        self.memory_initialized=extra['memory_initialized']
        if extra['sampler'] is not None:
            self.sampler=EpochSampler(extra['sampler']['n'],self.sampler_seed);self.sampler.load_state_dict(extra['sampler'])


def seed_learned_anchors(trainer:JointTrainer):
    """Explicitly import a fitted reference, WITHOUT fabricating usefulness evidence."""
    if trainer.config.plasticity.mode=='none':return
    for key,params in packet_map(trainer.model).items():
        trainer.plastic.reset_reference(key,params,to_current=True)


def migrate_single_tree(estimator,config:ForestConfig,*,updates_per_block=32):
    """Predictor-preserving bridge; fresh optimizer phase, not fake Adam-state resume.

    Returns (native trainer, the exact copied fitted preprocessor). A caller's
    preceding FeatureMap is caller-owned and must also be kept unchanged.
    """
    packed=estimator.model_
    native=packed.to_native()
    cfg=deepcopy(config)
    if cfg.n_trees!=1:raise ValueError('single-tree migration requires n_trees=1')
    if cfg.structure.structural_gate:raise ValueError('cannot invent nontrivial structural gates during import')
    cfg.structure.max_depth=max(cfg.structure.max_depth,packed.depth)
    cfg.structure.max_nodes=max(cfg.structure.max_nodes,len(native.node_map()))
    # Native aggregation is preserved, not substituted for a probability mixture.
    cfg.aggregation='mean';cfg.residual_weights=False;cfg.shrinkage=1.
    native.config=cfg
    for tree in native.trees:tree.config=cfg
    pre=deepcopy(estimator.preprocessor_)
    trainer=StepBudgetTrainer(native,Objective(pre.task,pre.output_dim),cfg,
                             torch.Generator().manual_seed(cfg.random_state),updates_per_block=updates_per_block)
    if cfg.physics.mode!='none':
        for key,node in native.node_map().items():trainer.physical.nodes[key]['temperature']=float(node.temperature)
    seed_learned_anchors(trainer)
    return trainer,pre
