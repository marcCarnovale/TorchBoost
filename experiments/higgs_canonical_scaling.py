"""Canonical HIGGS low-level-feature scaling study.

The UCI HIGGS dataset has 11,000,000 rows. Per the dataset protocol, the final
500,000 rows are never used for training/model selection and are opened only
after a family/configuration is fixed. The first 21 measured kinematic features
are used; the seven physicist-engineered high-level features are excluded.

Architectures are frozen from the 98,050-row development proxy:
- CatBoost: depth 10, max 1536 trees;
- MLP: five hidden layers of width 300;
- TorchBoost: 40-tree depth-6 progressive forest with learned tree rates.
Thus this script measures scaling, not repeated hyperparameter search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiments.higgs_hybrid_benchmark import MLP, _mlp_probability
from torchboost.adaptive.progressive import ProgressiveConfig, ProgressiveTreeClassifier

TOTAL_ROWS = 11_000_000
CANONICAL_TRAIN_STOP = 10_500_000
SELECTION_ROWS = 200_000
RANKING_ROWS = 200_000
TRAIN_POOL_STOP = CANONICAL_TRAIN_STOP - SELECTION_ROWS - RANKING_ROWS
AUDIT_START = CANONICAL_TRAIN_STOP
LOW_FEATURES = 21


def metrics(y, p):
    p = np.asarray(p, dtype=float).reshape(-1)
    return {
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)),
    }


def sha256_file(path, block=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def materialize(csv_gz: Path, cache: Path, *, expected_rows=TOTAL_ROWS, chunk_rows=100_000):
    cache.mkdir(parents=True, exist_ok=True)
    x_path = cache / "higgs-low21.f32"
    y_path = cache / "higgs-label.u8"
    meta_path = cache / "higgs-low21.json"
    source_stat = csv_gz.stat()
    signature = {
        "source_bytes": source_stat.st_size,
        "expected_rows": expected_rows,
        "features": LOW_FEATURES,
    }
    if x_path.exists() and y_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if all(meta.get(k) == v for k, v in signature.items()) and meta.get("rows") == expected_rows:
            return x_path, y_path, meta
    x = np.memmap(x_path, dtype="float32", mode="w+", shape=(expected_rows, LOW_FEATURES))
    y = np.memmap(y_path, dtype="uint8", mode="w+", shape=(expected_rows,))
    row = 0
    for frame in pd.read_csv(
        csv_gz,
        header=None,
        compression="gzip",
        usecols=range(22),
        dtype=np.float32,
        chunksize=chunk_rows,
    ):
        values = frame.to_numpy(dtype=np.float32, copy=False)
        n = len(values)
        if row + n > expected_rows:
            raise ValueError("HIGGS source contains more rows than expected")
        y[row:row+n] = values[:, 0].astype("uint8")
        x[row:row+n] = values[:, 1:22]
        row += n
        print(json.dumps({"materialized_rows": row}), flush=True)
    x.flush()
    y.flush()
    if row != expected_rows:
        raise ValueError(f"expected {expected_rows} canonical rows, parsed {row}")
    meta = {**signature, "rows": row, "source_sha256": sha256_file(csv_gz)}
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return x_path, y_path, meta


def arrays(x_path, y_path):
    x = np.memmap(x_path, dtype="float32", mode="r", shape=(TOTAL_ROWS, LOW_FEATURES))
    y = np.memmap(y_path, dtype="uint8", mode="r", shape=(TOTAL_ROWS,))
    return x, y


def fixed_splits(x, y, ntrain):
    if not 1 <= ntrain <= TRAIN_POOL_STOP:
        raise ValueError(f"train rows must be in [1, {TRAIN_POOL_STOP}]")
    return {
        "train": (x[:ntrain], y[:ntrain]),
        "selection": (
            x[TRAIN_POOL_STOP:TRAIN_POOL_STOP+SELECTION_ROWS],
            y[TRAIN_POOL_STOP:TRAIN_POOL_STOP+SELECTION_ROWS],
        ),
        "ranking": (
            x[TRAIN_POOL_STOP+SELECTION_ROWS:CANONICAL_TRAIN_STOP],
            y[TRAIN_POOL_STOP+SELECTION_ROWS:CANONICAL_TRAIN_STOP],
        ),
        "audit": (x[AUDIT_START:], y[AUDIT_START:]),
    }


def fit_cat(s, seed):
    started = time.perf_counter()
    model = CatBoostClassifier(
        iterations=1536,
        depth=10,
        learning_rate=.05,
        l2_leaf_reg=20,
        loss_function="Logloss",
        eval_metric="Logloss",
        verbose=False,
        random_seed=seed,
        thread_count=4,
        od_type="Iter",
        od_wait=100,
        use_best_model=True,
        allow_writing_files=False,
    ).fit(s["train"][0], s["train"][1], eval_set=s["selection"])
    rank = metrics(s["ranking"][1], model.predict_proba(s["ranking"][0])[:, 1])
    audit = metrics(s["audit"][1], model.predict_proba(s["audit"][0])[:, 1])
    return {
        "ranking": rank,
        "audit": audit,
        "retained_trees": int(model.tree_count_),
        "seconds": time.perf_counter() - started,
    }


def fit_mlp(s, seed, epochs=20):
    started = time.perf_counter()
    scaler = StandardScaler().fit(s["train"][0])
    train_x = scaler.transform(s["train"][0]).astype("float32")
    selection_x = scaler.transform(s["selection"][0]).astype("float32")
    ranking_x = scaler.transform(s["ranking"][0]).astype("float32")
    torch.manual_seed(seed + 305)
    model = MLP(LOW_FEATURES, 300, 5, .1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed + 9001)
    batch = 4096
    best_sel = float("inf")
    best_state = None
    best_epoch = 0
    train_y = np.asarray(s["train"][1], dtype="float32")
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), batch):
            idx = order[start:start+batch].numpy()
            xb = torch.from_numpy(train_x[idx])
            yb = torch.from_numpy(train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            opt.step()
        p = _mlp_probability(model, selection_x)
        sel = metrics(s["selection"][1], p)["nll"]
        print(json.dumps({"family": "mlp", "epoch": epoch + 1, "selection_nll": sel}), flush=True)
        if sel < best_sel:
            best_sel = sel
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    rank = metrics(s["ranking"][1], _mlp_probability(model, ranking_x))
    audit_x = scaler.transform(s["audit"][0]).astype("float32")
    audit = metrics(s["audit"][1], _mlp_probability(model, audit_x))
    return {
        "ranking": rank,
        "audit": audit,
        "best_epoch": best_epoch,
        "seconds": time.perf_counter() - started,
    }


def fit_tb(s, seed):
    started = time.perf_counter()
    cfg = ProgressiveConfig(
        n_trees=40,
        depth=6,
        stage_updates=24,
        batch_size=1024,
        learning_rate=.01,
        new_tree_shrinkage=.30,
        old_tree_lr_decay=.80,
        weight_decay=1e-5,
        cart_strength=7.,
        leaf_l2=1e-6,
        depth_shrinkage=2e-4,
        readout="residual",
        learn_tree_rates=True,
        tree_rate_l2=1e-5,
        tree_count_pressure=0.,
        row_subsample=.85,
        feature_subsample=.90,
        cart_value_updates=8,
        patience_stages=10,
        min_improvement=1e-6,
        random_state=seed + 46,
    )
    model = ProgressiveTreeClassifier(cfg).fit(
        s["train"][0], s["train"][1], eval_set=s["selection"]
    )
    rank = metrics(s["ranking"][1], model.predict_proba(s["ranking"][0])[:, 1])
    audit = metrics(s["audit"][1], model.predict_proba(s["audit"][0])[:, 1])
    return {
        "ranking": rank,
        "audit": audit,
        "retained_trees": int(model.n_estimators_),
        "seconds": time.perf_counter() - started,
    }


def run(csv_gz, cache, scales, families, seed=509, mlp_epochs=20, out=None):
    torch.set_num_threads(4)
    x_path, y_path, source = materialize(Path(csv_gz), Path(cache))
    x, y = arrays(x_path, y_path)
    result = {
        "status": "running",
        "seed": seed,
        "source": source,
        "features": "first 21 low-level detector features only",
        "canonical_test": {"start": AUDIT_START, "rows": TOTAL_ROWS - AUDIT_START},
        "selection_rows": SELECTION_ROWS,
        "ranking_rows": RANKING_ROWS,
        "frozen_architectures_from_proxy": True,
        "scales": {},
    }
    path = Path(out) if out else None

    def save():
        if path:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            tmp.replace(path)

    save()
    for n in scales:
        splits = fixed_splits(x, y, int(n))
        row = {}
        for family in families:
            print(json.dumps({"starting_family": family, "ntrain": int(n)}), flush=True)
            family_seed = seed + int(n) % 10007
            if family == "catboost":
                row[family] = fit_cat(splits, family_seed)
            elif family == "mlp":
                row[family] = fit_mlp(splits, family_seed, mlp_epochs)
            elif family == "torchboost":
                row[family] = fit_tb(splits, family_seed)
            else:
                raise ValueError(f"unknown family {family}")
            result["scales"][str(int(n))] = row
            save()
        result["scales"][str(int(n))] = row
        save()
    result["status"] = "completed"
    save()
    return result


def smoke(seed=7):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=2000)
    p = np.clip(.2 + .6 * y + rng.normal(0, .1, size=2000), .001, .999)
    return metrics(y, p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-gz")
    parser.add_argument("--cache", default="/tmp/higgs-cache")
    parser.add_argument("--scales", nargs="+", type=int, default=[500_000, 1_000_000, 3_000_000])
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("catboost", "mlp", "torchboost"),
        default=["catboost", "mlp", "torchboost"],
    )
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--mlp-epochs", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.smoke:
        result = {"status": "completed", "smoke": smoke(args.seed)}
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    else:
        if not args.csv_gz:
            parser.error("--csv-gz is required unless --smoke")
        result = run(
            args.csv_gz,
            args.cache,
            args.scales,
            args.families,
            args.seed,
            args.mlp_epochs,
            args.out,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
