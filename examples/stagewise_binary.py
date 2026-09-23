"""A complete stagewise fit, checkpoint, and hard-export smoke example."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from torchboost import StagewiseBinaryClassifier
from torchboost.export import predict_exported_proba


def main():
    torch.set_num_threads(1)
    x, y = make_classification(n_samples=500, n_features=8, n_informative=5, random_state=42)
    xt, xv, yt, yv = train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)
    model = StagewiseBinaryClassifier(n_estimators=12, epochs_per_stage=8, random_state=42)
    model.fit(xt, yt)
    p = model.predict_proba(xv)
    print(f"Held-out AUC={roc_auc_score(yv, p[:, 1]):.4f}; NLL={log_loss(yv, p):.4f}")
    with TemporaryDirectory() as folder:
        checkpoint, exported = Path(folder)/"model.pt", Path(folder)/"model.json"
        model.save(checkpoint)
        restored = StagewiseBinaryClassifier.load(checkpoint)
        np.testing.assert_allclose(p, restored.predict_proba(xv), rtol=0, atol=0)
        model.export_json(exported)
        np.testing.assert_allclose(model.predict_proba(xv, hard=True), predict_exported_proba(exported, xv), atol=1e-7)
    print(f"Checkpoint and hard export verified; {model.n_estimators_} immutable stages.")


if __name__ == "__main__":
    main()
