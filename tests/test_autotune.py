from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from torchboost.adaptive.autotune import (FeatureMap, SearchFold, TreeCandidate, SearchConfig,
    TreeSearch, MappedTree, default_candidates, make_search_folds, AutoTreeClassifier)
from torchboost.adaptive.single_tree import SingleTreeConfig
from torchboost.adaptive.specialist_forest import SpecialistForest

torch.set_num_threads(1)

@pytest.mark.parametrize('mode', ['raw','quantile','ple'])
def test_map_missing_unseen_train_only(mode):
    train=pd.DataFrame({'x':[1.,2.,np.nan,4.], 'empty':[np.nan]*4, 'cat':['a','b',None,'a']})
    mapper=FeatureMap(mode).fit(train)
    before=deepcopy(mapper.__dict__)
    test=pd.DataFrame({'x':[1e8,np.nan], 'empty':[1.,np.nan], 'cat':['new','MISSING:']})
    transformed=mapper.transform(test)
    assert np.isfinite(transformed).all()
    assert transformed.shape[1]==mapper.n_features_out_
    np.testing.assert_array_equal(mapper.median_,before['median_'])
    assert mapper.fit_rows_==4
    assert (transformed[:,-len(mapper.onehot_.categories_[0]):]==0).all()
    np.testing.assert_allclose(mapper.transform(train),mapper.transform(train[train.columns[::-1]]))

@pytest.mark.parametrize('mode',['raw','quantile','ple'])
def test_constant_feature_and_binary_category(mode):
    mapper=FeatureMap(mode).fit(pd.DataFrame({'x':[3.]*10,'c':['a']*10}))
    assert np.isfinite(mapper.transform(pd.DataFrame({'x':[3.],'c':['b']}))).all()

@pytest.mark.parametrize('mode',['raw','quantile','ple'])
def test_encoder_not_mutating_input(mode):
    a=pd.DataFrame({'x':[1.,2.,3.], 'c':['a','b','a']});b=a.copy(deep=True)
    FeatureMap(mode).fit_transform(a)
    pd.testing.assert_frame_equal(a,b)


def test_encoder_schema_checks():
    a=FeatureMap().fit(np.ones((10,3)))
    with pytest.raises(ValueError):a.transform(np.ones((10,2)))
    with pytest.raises(ValueError):FeatureMap().fit(np.array([[np.inf,1.]]))
    with pytest.raises(ValueError):FeatureMap().fit(pd.DataFrame([[1,2]],columns=['a','a']))


def test_ple_ramps_monotone():
    x=np.linspace(-2,2,50)[:,None]
    mapper=FeatureMap('ple',4).fit(x)
    z=mapper.transform(x)
    assert np.all(np.diff(z[:,2:],axis=0)>=-1e-7)
    assert np.all(z[:,2:]>=0) and np.all(z[:,2:]<=1)


def test_folds_independent_roles():
    x=np.arange(240).reshape(120,2);y=np.tile([0,1],60)
    folds=make_search_folds(x,y,classification=True,random_state=3)
    assert len(folds)==3
    rank=np.concatenate([f.rank for f in folds]);assert len(np.unique(rank))==120
    for f in folds:f.validate(120)


def test_group_roles_never_overlap():
    g=np.repeat(np.arange(60),3);y=np.repeat(np.arange(60)%2,3);x=np.arange(180)[:,None]
    folds=make_search_folds(x,y,classification=True,groups=g)
    for f in folds:f.validate(180,groups=g)


def test_forward_time_roles():
    t=np.repeat(np.arange(60),4);x=np.arange(240)[:,None]
    folds=make_search_folds(x,x.ravel(),classification=False,times=t)
    for f in folds:f.validate(240,times=t)

@pytest.mark.parametrize('fold',[SearchFold(np.array([0,1]),np.array([1,2]),np.array([3])),
                                  SearchFold(np.array([0,0]),np.array([1]),np.array([2])),
                                  SearchFold(np.array([0]),np.array([1]),np.array([20]))])
def test_invalid_folds_rejected(fold):
    with pytest.raises(ValueError):fold.validate(10)


def tiny_data():
    rng=np.random.default_rng(31);x=rng.normal(size=(180,4)).astype('float32')
    return x,(x[:,0]+x[:,1]>.2).astype(int)


def tiny_choice(name='a',seed=1,depth=2):
    return TreeCandidate(name,SingleTreeConfig(depth=depth,epochs=5,evaluate_every=1,
        learning_rate=.03,batch_size=64,random_state=seed))


def test_search_records_and_budget_semantics():
    x,y=tiny_data();candidates=[tiny_choice('a',depth=1),tiny_choice('b',depth=3)]
    log=[]
    search=TreeSearch(candidates=candidates,config=SearchConfig(budgets=(2,4),finalists=1)).fit(x,y,callback=log.append)
    assert search.best_candidate_.tree.epochs==4
    assert search.summary()['failed_fits']==0
    assert len(log)==len(search.records_)
    assert {r['epochs'] for r in log}=={2,4}
    assert all(r['history'][0]['epoch']==0 for r in log)
    assert len(search.ranking_)>=1
    for c in candidates:assert c.tree.epochs==5


def test_rank_labels_never_checkpoint_inputs(monkeypatch):
    x,y=tiny_data();seen=[]
    original=MappedTree.fit
    def traced(self,X,Y,**kwargs):
        seen.append((set(np.asarray(X)[:,0]),set(np.asarray(kwargs['eval_set'][0])[:,0])))
        return original(self,X,Y,**kwargs)
    monkeypatch.setattr(MappedTree,'fit',traced)
    folds=make_search_folds(x,y,classification=True)
    TreeSearch(candidates=[tiny_choice()],config=SearchConfig(budgets=(2,),finalists=1)).fit(x,y,folds=folds)
    for (fit,stop),fold in zip(seen,folds):
        rank=set(x[fold.rank,0]);assert not fit&rank and not stop&rank


def test_search_reproducible():
    x,y=tiny_data();c=[tiny_choice()];cfg=SearchConfig(budgets=(2,))
    a=TreeSearch(candidates=c,config=cfg).fit(x,y)
    b=TreeSearch(candidates=c,config=cfg).fit(x,y)
    np.testing.assert_equal(a.ranking_,b.ranking_)


def test_auto_tree_sklearn_clone():
    m=AutoTreeClassifier([tiny_choice()],SearchConfig(budgets=(2,)))
    cloned=clone(m)
    assert cloned.candidates[0].tree.depth==2


def fitted_members(classification=True):
    x,y=tiny_data()
    if not classification:y=x[:,0]**2+x[:,1]
    members=[MappedTree(tiny_choice(str(k),seed=k+3),classification=classification).fit(
        x[:100],y[:100],eval_set=(x[100:140],y[100:140])) for k in range(3)]
    return x,y,members

@pytest.mark.parametrize('classification',[True,False])
def test_forest_mean_and_blend(classification):
    x,y,members=fitted_members(classification)
    f=SpecialistForest(members)
    p=f.predict_proba(x) if classification else f.predict(x)
    np.testing.assert_allclose(p,f.member_predictions(x).mean(0),atol=1e-7)
    before=[deepcopy(m.tree_.model_.state_dict()) for m in members]
    f.fit_blend(x[140:],y[140:])
    assert np.isclose(f.weights_.sum(),1.) and (f.weights_>=0).all()
    for a,m in zip(before,members):
        for key,v in a.items():assert torch.equal(v,m.tree_.model_.state_dict()[key])
    d=f.diagnostics(x[140:],y[140:]);assert not d['oracle_is_deployable']

@pytest.mark.parametrize('classification',[True,False])
def test_joint_refinement_preserves_source_and_initial_candidate(classification):
    x,y,members=fitted_members(classification);f=SpecialistForest(members)
    before=[deepcopy(m.tree_.model_.state_dict()) for m in members]
    j=f.jointly_refine(x[:100],y[:100],eval_set=(x[100:140],y[100:140]),epochs=4,evaluate_every=1)
    for a,m in zip(before,members):
        for key,v in a.items():assert torch.equal(v,m.tree_.model_.state_dict()[key])
    assert j.joint_best_score_ <= j.joint_history_[0]['stopping_loss']+1e-9
    assert np.isfinite(j.predict(x)).all()
    if classification:np.testing.assert_allclose(j.predict_proba(x).sum(1),1.,atol=1e-6)


def test_mixture_rejects_wrong_weights():
    _,_,m=fitted_members()
    with pytest.raises(ValueError):SpecialistForest(m,weights=[1.,1.,1.])


def test_default_candidate_dials():
    c=default_candidates();assert len(c)==12
    assert {v.representation for v in c}=={'raw','ple','quantile'}
    assert {v.tree.arity for v in c}=={2,3}
    assert {v.tree.readout for v in c}=={'residual','leaf'}

@pytest.mark.parametrize('budgets',[(3,2),(0,3),(3,3)])
def test_bad_budgets_rejected(budgets):
    with pytest.raises(ValueError):SearchConfig(budgets=budgets)
