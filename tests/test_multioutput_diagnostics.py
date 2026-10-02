import numpy as np
import pytest
from torchboost.adaptive.autotune import MappedTree,TreeCandidate
from torchboost.adaptive.specialist_forest import SpecialistForest
from experiments.reference_mixture import ReferenceMixture

class FixedMember(MappedTree):
    def __init__(self,p):
        self.p=np.asarray(p,float);self.classification=False;self.tree_=object();self.encoder_=object()
    def predict(self,X):return self.p

@pytest.mark.parametrize('kind',['trees','references'])
def test_opposite_output_errors_do_not_cancel_in_diagnostics(kind):
    y=np.zeros((4,2));members=[FixedMember([[1,-1]]*4),FixedMember([[2,-2]]*4)]
    forest=SpecialistForest(members) if kind=='trees' else ReferenceMixture(members,classification=False)
    result=forest.diagnostics(np.zeros((4,1)),y)
    assert result['loss']==pytest.approx(1.5)
    assert result['oracle_casewise_error']==pytest.approx(1.)
    assert result['error_correlation_mean']==pytest.approx(1.)


def test_public_forest_rejects_unfitted_member_cleanly():
    with pytest.raises(ValueError,match='fitted'):
        SpecialistForest([MappedTree(TreeCandidate('x'),classification=False)])
