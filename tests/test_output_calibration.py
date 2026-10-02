from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import numpy as np
import pytest
from temperature_calibration_probe import fit_temperature,probabilities,logit_nll

@pytest.mark.parametrize('classes',[2,4])
def test_temperature_fit_preserves_classes_and_reduces_control_loss(classes):
    rng=np.random.default_rng(774);z=rng.normal(size=(100,classes))*8
    logits=z[:,1]-z[:,0] if classes==2 else z
    y=np.argmax(z+rng.normal(size=z.shape)*5,axis=1)
    result=fit_temperature(logits,y)
    assert .05<=result['temperature']<=20
    assert result['control_nll_after']<=result['control_nll_before']+1e-10
    np.testing.assert_array_equal(probabilities(logits).argmax(1),probabilities(logits,result['temperature']).argmax(1))

def test_temperature_rejects_nonpositive_or_invalid_inputs():
    with pytest.raises(ValueError):logit_nll([1,2],[0,1],temperature=0)
    with pytest.raises(ValueError):fit_temperature([float('nan')],[0])
    with pytest.raises(ValueError):fit_temperature([1,2],[0,3])
