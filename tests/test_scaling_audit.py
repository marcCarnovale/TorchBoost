import numpy as np
import pytest
from experiments.scale_study import metric
from torchboost.adaptive.data import Preprocessor

def test_binary_score_agrees_with_expected_at_bayes_probability():
    p=np.array([.1,.3,.8]);z=np.log(p/(1-p))
    entropy=-np.mean(p*np.log(p)+(1-p)*np.log(1-p))
    expected=np.mean(np.logaddexp(0,z)-p*z)
    np.testing.assert_allclose(expected,entropy,atol=1e-15)

def test_rank_metric_rejects_multitarget_broadcast():
    pre=Preprocessor();pre.task='regression';pre.target_scale=np.ones(2)
    with pytest.raises(ValueError):metric(np.ones((5,1)),np.ones((5,2)),pre)

def test_audit_cannot_lock_incomplete_job_set(tmp_path,monkeypatch):
    import experiments.audit_scaling as audit
    with pytest.raises((AssertionError,FileNotFoundError)):
        monkeypatch.setattr(audit,'OUT',tmp_path)
        audit.verify_ready()
