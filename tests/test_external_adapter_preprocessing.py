import numpy as np
import pandas as pd

from experiments.external_adapter_benchmark import fit_preprocessor


def test_external_preprocessor_preserves_categorical_signal_and_handles_unknowns():
    train = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, np.nan, 4.0],
            "category": ["a", "b", "a", None],
        }
    )
    selection = pd.DataFrame(
        {
            "numeric": [2.5, np.nan],
            "category": ["b", "unseen"],
        }
    )

    preprocessor = fit_preprocessor(train)
    train_encoded = np.asarray(preprocessor.fit_transform(train))
    selection_encoded = np.asarray(preprocessor.transform(selection))

    assert train_encoded.shape[1] >= 3
    assert selection_encoded.shape[1] == train_encoded.shape[1]
    assert np.isfinite(train_encoded).all()
    assert np.isfinite(selection_encoded).all()
    assert not np.allclose(train_encoded[0], train_encoded[1])
