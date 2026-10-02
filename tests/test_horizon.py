from copy import deepcopy
import numpy as np
import pytest
import torch
from torchboost.adaptive.autotune import MappedTree,TreeCandidate
from torchboost.adaptive.single_tree import SingleTreeConfig
from torchboost.adaptive.horizon import HorizonPolicy,extend_positive_tail

torch.set_num_threads(1)

@pytest.mark.parametrize('task',['binary','multiclass','regression','multitarget'])
@pytest.mark.parametrize('readout',['residual','leaf'])
def test_continuation_matches_uninterrupted_cpu(task,readout):
    rng=np.random.default_rng(61);X=rng.normal(size=(160,4)).astype('float32')
    classification=task in ['binary','multiclass']
    y=(X[:,0]>.1).astype(int) if task=='binary' else np.digitize(X[:,0],[-.4,.4]) if task=='multiclass' else X[:,0]**2+X[:,1]
    if task=='multitarget':y=np.c_[y,X[:,2]-X[:,3]]
    base=SingleTreeConfig(depth=2,epochs=8,readout=readout,random_state=17,batch_size=31,evaluate_every=2,
                          learning_rate=.01,logit_penalty=.001)
    member=MappedTree(TreeCandidate('tiny',base),classification=classification).fit(X[:100],y[:100],eval_set=(X[100:130],y[100:130]))
    old=deepcopy(member.tree_.model_.state_dict())
    continuation=extend_positive_tail(member,X[:100],y[:100],eval_set=(X[100:130],y[100:130]),total_epochs=20)
    full=deepcopy(base);full.epochs=20;full.schedule_epochs=8
    reference=MappedTree(TreeCandidate('tiny',full),classification=classification).fit(X[:100],y[:100],eval_set=(X[100:130],y[100:130]))
    for key,value in reference.tree_.last_state_.items():assert torch.equal(value,continuation.tree_.last_state_[key]),key
    for key,value in reference.tree_.best_state_.items():assert torch.equal(value,continuation.tree_.best_state_[key]),key
    for key,value in old.items():assert torch.equal(value,member.tree_.model_.state_dict()[key])
    assert continuation.tree_.optimizer_steps_==reference.tree_.optimizer_steps_
    assert continuation.tree_.examples_seen_==reference.tree_.examples_seen_
    assert continuation.continuation_['additional_epochs']==12
    np.testing.assert_array_equal(continuation.predict(X[130:]),reference.predict(X[130:]))


def test_changed_data_and_nonadvancing_horizon_rejected():
    rng=np.random.default_rng(2);X=rng.normal(size=(80,3));y=X[:,0]
    member=MappedTree(TreeCandidate('a',SingleTreeConfig(depth=1,epochs=4,random_state=3)),classification=False).fit(X[:50],y[:50],eval_set=(X[50:],y[50:]))
    with pytest.raises(ValueError):extend_positive_tail(member,X[:50],y[:50]+1,eval_set=(X[50:],y[50:]),total_epochs=8)
    with pytest.raises(ValueError):extend_positive_tail(member,X[:50],y[:50],eval_set=(X[50:],y[50:]),total_epochs=4)
    member.tree_.best_epoch_=4
    assert HorizonPolicy(maximum_epochs=20).target(member)==16
    member.tree_.best_epoch_=0
    assert HorizonPolicy().target(member)==4
