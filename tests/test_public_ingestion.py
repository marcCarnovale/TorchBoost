import io,zipfile,gzip
import numpy as np
import pytest
from experiments.public_benchmarks import parse

def archive(files):
    f=io.BytesIO()
    with zipfile.ZipFile(f,'w') as z:
        for k,v in files.items():z.writestr(k,v)
    f.seek(0);return f

def test_miniboone_header_is_not_data_and_label_order_is_preserved():
    text='2 1\n'+'\n'.join(' '.join(map(str,row)) for row in np.arange(150).reshape(3,50))
    x,y=parse('miniboone',archive({'MiniBooNE_PID.txt':text}),check_count=False)
    assert x.shape==(3,50);assert y.tolist()==[1,1,0]

def test_miniboone_wrong_header_rejected():
    text='3 1\n'+' '.join(['0']*50)
    with pytest.raises(ValueError):parse('miniboone',archive({'MiniBooNE_PID.txt':text}),check_count=False)

def test_sgemm_repeats_are_one_row_one_target():
    header=','.join(f'f{i}' for i in range(18));row=','.join(map(str,[1]*14+[1,2,4,9]))
    x,y=parse('sgemm',archive({'sgemm_product.csv':header+'\n'+row+'\n'}),check_count=False)
    assert x.shape==(1,14);assert y[0]==pytest.approx(np.log(4))

def test_covertype_features_not_labels():
    raw=(','.join(map(str,[1]*54+[7]))+'\n'+','.join(map(str,[2]*54+[2]))+'\n').encode()
    x,y=parse('covertype',archive({'covtype.data.gz':gzip.compress(raw)}),check_count=False)
    assert x.shape==(2,54);assert y.tolist()==[6,1]

def test_full_count_contract_not_silently_bypassed():
    raw=(','.join(map(str,[1]*54+[7]))+'\n')*2
    with pytest.raises(ValueError):parse('covertype',archive({'covtype.data.gz':gzip.compress(raw.encode())}))
