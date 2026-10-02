"""Exact CPU continuation for the fixed one-tree learner, without replaying epochs.

Continue from the last optimization state, not the selected deployment weights.
Feature maps, preprocessing, topology and schedule horizon remain fixed. The
same fit and stopping records are required. Native adaptive training has its own
checkpoint protocol and is not silently routed through this helper.
"""
from copy import deepcopy
from dataclasses import dataclass
import math,time
import torch
from .data import sample_weights
from .single_tree import _fit_objective, refit_readout, routing_telemetry, model_state_digest
from .autotune import MappedTree


@dataclass(frozen=True)
class HorizonPolicy:
    maximum_epochs: int = 2048
    boundary_fraction: float = .8
    extension_factor: int = 4

    def target(self,member):
        if self.maximum_epochs<1 or not 0<self.boundary_fraction<=1 or self.extension_factor<2:
            raise ValueError('invalid horizon policy')
        previous=member.tree_.config_.epochs
        if member.tree_.best_epoch_<self.boundary_fraction*previous:return previous
        return max(previous,min(self.maximum_epochs,self.extension_factor*previous))


def extend_positive_tail(member,X,y,*,eval_set,total_epochs,sample_weight=None):
    """Return a continued independent member; never mutate the supplied predictor.

    Only CPU fixed-tree checkpoints with full optimizer/generator state qualify.
    ``total_epochs`` counts old plus additional epochs. A fitted schedule keeps
    its existing horizon and a positive final learning rate after that horizon.
    This function is a training operation: supplied stop labels select checkpoints.
    """
    if not isinstance(member,MappedTree) or not hasattr(member,'tree_'):
        raise TypeError('a fitted MappedTree is required')
    result=deepcopy(member);est=result.tree_;c=deepcopy(est.config_);previous=c.epochs
    if not isinstance(total_epochs,int) or total_epochs<=previous:raise ValueError('total_epochs must exceed the trained horizon')
    if any(p.device.type!='cpu' for p in est.model_.parameters()):raise ValueError('exact continuation is currently CPU-only')
    if c.lr_schedule!='constant' and c.final_learning_rate_ratio<=0:raise ValueError('continuation requires a positive learning-rate tail')
    if est.history_[-1]['epoch']!=previous:raise ValueError('checkpoint does not end at the declared horizon')
    weights=sample_weights(sample_weight,len(X))
    train=est.preprocessor_.split(result.encoder_.transform(X),y,weights)
    valid=est.preprocessor_.split(result.encoder_.transform(eval_set[0]),eval_set[1],*eval_set[2:])
    if train.fingerprint()!=est.data_fingerprints_['train'] or valid.fingerprint()!=est.data_fingerprints_['validation']:
        raise ValueError('continuation requires the original ordered fit/stop records and weights')
    obj=est.objective_
    with torch.no_grad():current=float(obj.weighted_loss(est.model_(valid.x),valid.y,valid.weight))
    if not math.isclose(current,est.best_score_,rel_tol=1e-6,abs_tol=1e-7):
        raise ValueError('selected model no longer matches its trainer record; joint-refined members cannot be resumed here')
    if not all(hasattr(est,name) for name in ['optimizer_state_','generator_','last_state_']):
        raise ValueError('checkpoint lacks optimizer, last weights, or shuffle-generator state')
    model=est.model_;model.load_state_dict(est.last_state_);model.train()
    boundary_digest=model_state_digest(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c.learning_rate,weight_decay=c.weight_decay)
    optimizer.load_state_dict(est.optimizer_state_)
    c.schedule_epochs=c.schedule_epochs or previous;c.epochs=total_epochs
    est.config_=c;est.config=deepcopy(c);result.candidate.tree=deepcopy(c)
    start_steps=est.optimizer_steps_;start_examples=est.examples_seen_;started=time.perf_counter()
    def evaluate(epoch,rate,grad):
        with torch.no_grad():
            tr,va=model(train.x),model(valid.x)
            tl=float(obj.weighted_loss(tr,train.y,train.weight));vl=float(obj.weighted_loss(va,valid.y,valid.weight))
            if not math.isfinite(tl+vl):raise FloatingPointError('nonfinite continuation evaluation')
            trp,vap=obj.response(tr),obj.response(va)
            row=dict(epoch=epoch,train_loss=tl,validation_loss=vl,
                train_error=float((trp.argmax(1)!=train.y).float().mean()) if est.classification else None,
                validation_error=float((vap.argmax(1)!=valid.y).float().mean()) if est.classification else None,
                learning_rate=rate,gradient_norm=grad,
                temperature=float(model.temperatures.mean()) if model.n_internal else c.temperature,
                optimizer_steps=est.optimizer_steps_,examples_seen=est.examples_seen_)
            if c.routing_diagnostics_every and (epoch==c.epochs or epoch%c.routing_diagnostics_every==0):
                row['routing']=routing_telemetry(model,train)
            est.history_.append(row)
            if vl<est.best_score_:
                est.best_score_,est.best_epoch_,est.best_state_=vl,epoch,deepcopy(model.state_dict())
    for epoch in range(previous+1,total_epochs+1):
        progress=min(1.,(epoch-1)/max(c.schedule_epochs-1,1))
        f=1. if c.lr_schedule=='constant' else c.final_learning_rate_ratio+(1-c.final_learning_rate_ratio)*.5*(1+math.cos(math.pi*progress))
        rate=c.learning_rate*f
        for group in optimizer.param_groups:group['lr']=rate
        if c.final_temperature is not None:model.temperatures.fill_(c.temperature*(c.final_temperature/c.temperature)**progress)
        order=torch.randperm(len(train.x),generator=est.generator_);gradient=0.
        for start in range(0,len(order),c.batch_size):
            idx=order[start:start+c.batch_size]
            if float(train.weight[idx].sum())==0:continue
            optimizer.zero_grad();loss=_fit_objective(model,train.x[idx],train.y[idx],train.weight[idx],obj,c)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite continuation objective')
            loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),c.gradient_clip,error_if_nonfinite=True))
            optimizer.step();est.optimizer_steps_+=1;est.examples_seen_+=len(idx)
        if c.refit_every and epoch%c.refit_every==0:
            record=refit_readout(model,train,obj,c);est.refits_.append({'epoch':epoch,**record})
            if record['accepted']:
                optimizer.state.pop(model.values,None);optimizer.state.pop(model.bias,None)
        if epoch%c.evaluate_every==0 or epoch==c.epochs:evaluate(epoch,rate,gradient)
    elapsed=time.perf_counter()-started
    est.fit_seconds_+=elapsed;est.last_state_=deepcopy(model.state_dict());est.optimizer_state_=deepcopy(optimizer.state_dict())
    model.load_state_dict(est.best_state_);model.eval()
    result.continuation_=dict(start_epoch=previous,total_epochs=total_epochs,additional_epochs=total_epochs-previous,
        schedule_horizon=c.schedule_epochs,boundary_last_state_sha256=boundary_digest,
        additional_optimizer_steps=est.optimizer_steps_-start_steps,additional_example_exposures=est.examples_seen_-start_examples,
        elapsed_seconds=elapsed,scope='CPU fixed topology; same ordered data; last optimizer state, not selected weights')
    return result
