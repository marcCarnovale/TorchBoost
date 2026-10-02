"""Universal differentiable soft-tree deformation screen.

One model class, one optimizer, one regularization law, and one initialization
policy are used unchanged across synthetic geometries.  There is no expert
mixture and no architecture selector.

A complete potential binary tree is represented differentiably.  Each internal
node owns a soft branch-existence gate and a learned soft routing hyperplane.
Every node owns a zero-at-birth local residual that can continuously deform from
constant -> affine -> low-rank bilinear.  Complexity rent is paid directly in
the training objective for active depth/branches, dense routing, affine values,
and interaction rank.  The test asks whether gradient descent alone moves the
same learner into different structural phases.

Physical lazy allocation is intentionally NOT implemented here: the complete
small supertree is materialized so this experiment isolates whether the
differentiable structural signal works before changing the production allocator.

Synthetic only; exact Bayes probabilities are known.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import experiments.fast_semantic_mechanism_screen as base

REGIMES = ("axis", "oblique", "ridge", "interaction", "regional_mix")
N_TRAIN = 4_000
N_TEST = 12_000
FIT_FRAC = .75
PASSES = 48
BATCH = 256
LR = 1.5e-3
MAX_DEPTH = 4
RANK = 4

# Fixed across every regime.  These are structural priors, not tuned per task.
BRANCH_COST = 5.0e-3
OBLIQUE_COST = 2.0e-3
AFFINE_COST = 1.0e-3
INTERACTION_COST = 6.0e-3
VALUE_L2 = 1.0e-4


class UniversalSoftTree(nn.Module):
    """Single soft tree with differentiable existence and deformation coordinates."""

    def __init__(self, p: int, max_depth: int = MAX_DEPTH, rank: int = RANK):
        super().__init__()
        self.p = p
        self.max_depth = max_depth
        self.rank = rank
        self.n_nodes = 2 ** (max_depth + 1) - 1
        self.n_internal = 2 ** max_depth - 1

        # Routing starts weak and generic. Sparsity pressure can collapse a dense
        # hyperplane toward an axis split; nothing selects an "oblique expert".
        self.route_w = nn.Parameter(torch.empty(self.n_internal, p))
        self.route_b = nn.Parameter(torch.zeros(self.n_internal))
        self.route_log_sharp = nn.Parameter(torch.zeros(self.n_internal))
        nn.init.normal_(self.route_w, std=.04)

        # A potential branch begins partially open so descendants receive a
        # usable gradient. Complexity rent can close it; predictive gain can
        # drive it open. Deeper branches pay more.
        self.branch_logit = nn.Parameter(torch.full((self.n_internal,), -.35))

        # Local predictions are exactly zero at birth. Affine and interaction
        # capacity therefore cannot perturb the inherited simpler predictor
        # until gradient descent proves useful.
        self.value_bias = nn.Parameter(torch.zeros(self.n_nodes))
        self.affine = nn.Parameter(torch.zeros(self.n_nodes, p))

        # Factor directions are harmless until the zero interaction gains move.
        self.ia = nn.Parameter(torch.empty(self.n_nodes, rank, p))
        self.ib = nn.Parameter(torch.empty(self.n_nodes, rank, p))
        self.igain = nn.Parameter(torch.zeros(self.n_nodes, rank))
        nn.init.normal_(self.ia, std=.05)
        nn.init.normal_(self.ib, std=.05)

    def local_value(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, nodes]
        affine = x @ self.affine.T
        ua = torch.einsum("bp,nrp->bnr", x, self.ia)
        ub = torch.einsum("bp,nrp->bnr", x, self.ib)
        bilinear = (ua * ub * self.igain[None, :, :]).sum(-1) / math.sqrt(self.rank)
        return self.value_bias[None, :] + affine + bilinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local_value(x)
        values = [None] * self.n_nodes

        # Bottom-up recursive soft tree. s_v=0 means "stop here"; s_v=1 means
        # use the learned child refinement in addition to this node's residual.
        for node in range(self.n_nodes - 1, -1, -1):
            here = local[:, node]
            if node < self.n_internal:
                direction = self.route_w[node] / self.route_w[node].norm().clamp_min(1e-6)
                sharp = .5 + torch.nn.functional.softplus(self.route_log_sharp[node])
                prob = torch.sigmoid(sharp * (x @ direction + self.route_b[node]))
                s = torch.sigmoid(self.branch_logit[node])
                left = values[2 * node + 1]
                right = values[2 * node + 2]
                here = here + s * ((1.0 - prob) * left + prob * right)
            values[node] = here
        return values[0]

    def reach_weights(self) -> torch.Tensor:
        """Expected structural reach independent of left/right probability.

        This is deliberately functional rather than an indexed in-place build:
        branch reach itself is differentiable, so autograd must retain the full
        parent->child graph without tensor version mutations.
        """
        levels = [self.branch_logit.new_ones(1)]
        offset = 0
        for depth in range(self.max_depth):
            parent = levels[-1]
            count = 2 ** depth
            gate = torch.sigmoid(self.branch_logit[offset:offset + count])
            child_mass = parent * gate * .5
            levels.append(torch.stack((child_mass, child_mass), dim=1).reshape(-1))
            offset += count
        return torch.cat(levels)

    def complexity(self, progress: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reach = self.reach_weights()
        branch = self.branch_logit.new_zeros(())
        for node in range(self.n_internal):
            depth = int(math.floor(math.log2(node + 1)))
            branch = branch + reach[node] * torch.sigmoid(self.branch_logit[node]) * (1.0 + .35 * depth)

        # Weight deformation rent by structural reach: parameters in an unused
        # branch should be driven to zero, not become a hidden free model.
        r = reach[:, None]
        affine = (r * self.affine.square().sum(1, keepdim=True).add(1e-12).sqrt()).sum()
        interaction = (r * self.igain.square().sum(1, keepdim=True).add(1e-12).sqrt()).sum()

        # Normalize routing directions before charging for obliqueness.  This
        # removes the old scale degeneracy: sharpness controls how hard a split
        # is, while ||u||_1-1 measures only departure from an axis direction.
        direction = self.route_w / self.route_w.norm(dim=1, keepdim=True).clamp_min(1e-6)
        route_reach = reach[:self.n_internal]
        oblique = (route_reach * (direction.abs().sum(1) - 1.0).clamp_min(0.)).sum()

        value = (reach * self.value_bias.square()).sum()

        # Ramp structural rent in after the learner has had a chance to discover
        # useful directions; same schedule in every regime.
        ramp = min(1.0, max(0.0, (progress - .10) / .45))
        terms = {
            "branch": BRANCH_COST * branch * ramp,
            "oblique": OBLIQUE_COST * oblique * ramp,
            "affine": AFFINE_COST * affine * ramp,
            "interaction": INTERACTION_COST * interaction * ramp,
            "value_l2": VALUE_L2 * value,
        }
        return sum(terms.values()), terms

    @torch.no_grad()
    def structure_summary(self) -> dict:
        reach = self.reach_weights().detach()
        gates = torch.sigmoid(self.branch_logit).detach()
        depth_rows = []
        for depth in range(self.max_depth):
            lo, hi = 2 ** depth - 1, 2 ** (depth + 1) - 1
            g = gates[lo:hi]
            rr = reach[lo:hi]
            depth_rows.append({
                "depth": depth,
                "mean_branch_gate": float(g.mean()),
                "max_branch_gate": float(g.max()),
                "mean_reach": float(rr.mean()),
            })

        unit_route = self.route_w.detach() / self.route_w.detach().norm(dim=1, keepdim=True).clamp_min(1e-12)
        route_abs = unit_route.abs()
        denom = route_abs.sum(1).clamp_min(1e-12)
        concentration = route_abs.max(1).values / denom
        q = route_abs / denom[:, None]
        effective_features = 1.0 / q.square().sum(1).clamp_min(1e-12)

        weighted_affine = float((reach[:, None] * self.affine.detach().abs()).sum())
        weighted_interaction = float((reach[:, None] * self.igain.detach().abs()).sum())
        expected_branches = float(sum(
            reach[n] * gates[n] for n in range(self.n_internal)
        ))
        active_branch_count = int((gates > .5).sum())
        return {
            "expected_active_branches": expected_branches,
            "branch_gates_over_half": active_branch_count,
            "depths": depth_rows,
            "route_axis_concentration_mean": float(concentration.mean()),
            "route_effective_features_mean": float(effective_features.mean()),
            "route_sharpness_mean": float((.5 + torch.nn.functional.softplus(self.route_log_sharp.detach())).mean()),
            "reach_weighted_affine_l1": weighted_affine,
            "reach_weighted_interaction_gain_l1": weighted_interaction,
            "root_affine_l2": float(self.affine[0].detach().norm()),
            "root_interaction_gain_l2": float(self.igain[0].detach().norm()),
        }


def train(model, fx, fy, sx, sy, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    xt = torch.from_numpy(fx)
    yt = torch.from_numpy(fy)
    sxt = torch.from_numpy(sx)
    syt = torch.from_numpy(sy)
    gen = torch.Generator().manual_seed(seed + 19)

    best = (float("inf"), None, 0)
    history = []
    for epoch in range(PASSES):
        model.train()
        order = torch.randperm(len(fx), generator=gen)
        epoch_terms = {}
        for start in range(0, len(order), BATCH):
            idx = order[start:start+BATCH]
            progress = (epoch + start / max(1, len(order))) / PASSES
            opt.zero_grad(set_to_none=True)
            logits = model(xt[idx])
            pred_loss = nn.functional.binary_cross_entropy_with_logits(logits, yt[idx])
            reg, terms = model.complexity(progress)
            loss = pred_loss + reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            for k, v in terms.items():
                epoch_terms[k] = epoch_terms.get(k, 0.0) + float(v.detach())

        model.eval()
        with torch.no_grad():
            sel = float(nn.functional.binary_cross_entropy_with_logits(model(sxt), syt))
        if sel < best[0]:
            best = (sel, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch + 1)
        if epoch in (0, 3, 7, 15, 23, 31, 39, 47):
            history.append({
                "epoch": epoch + 1,
                "selection_nll": sel,
                "regularization_terms": {k: v / max(1, math.ceil(len(fx)/BATCH)) for k, v in epoch_terms.items()},
                "structure": model.structure_summary(),
            })

    model.load_state_dict(best[1])
    return model, best[0], best[2], history


@torch.no_grad()
def predict(model, x):
    model.eval()
    out = []
    for start in range(0, len(x), 2048):
        out.append(torch.sigmoid(model(torch.from_numpy(x[start:start+2048]))).numpy())
    return np.concatenate(out)


def run(regime, seed, out):
    torch.set_num_threads(4)
    started = time.perf_counter()
    problem = base.latent(regime, seed + 41)
    x, y, _ = base.sample(problem, N_TRAIN, seed + 1001)
    qx, qy, bayes_p = base.sample(problem, N_TEST, seed + 500001)

    fit_idx, sel_idx = train_test_split(
        np.arange(len(y)), train_size=FIT_FRAC, stratify=y, random_state=seed + 77
    )
    scaler = StandardScaler().fit(x[fit_idx])
    x = scaler.transform(x).astype("float32")
    qx = scaler.transform(qx).astype("float32")
    fx, fy = x[fit_idx], y[fit_idx]
    sx, sy = x[sel_idx], y[sel_idx]

    # Exact same initial random state in every regime.  Only the data geometry
    # changes, so structural differences cannot be attributed to initialization.
    torch.manual_seed(seed + 2000)
    model = UniversalSoftTree(base.P)
    model, best_sel, best_epoch, history = train(model, fx, fy, sx, sy, seed + 2000)
    pp = predict(model, qx)
    tm = base.metrics(qy, pp)
    bayes = base.metrics(qy, bayes_p)

    # Comparator only. It never participates in training or architecture choice.
    cat = CatBoostClassifier(
        iterations=300, depth=7, learning_rate=.06, l2_leaf_reg=8,
        loss_function="Logloss", verbose=False, random_seed=seed, thread_count=4,
        allow_writing_files=False,
    )
    cat.fit(fx, fy)
    cm = base.metrics(qy, cat.predict_proba(qx)[:, 1])
    gap = cm["nll"] - bayes["nll"]

    result = {
        "study": "universal_soft_tree_deformation_v2",
        "regime": regime,
        "seed": seed,
        "model_policy_identical_across_regimes": True,
        "train_rows": N_TRAIN,
        "fit_rows": int(len(fit_idx)),
        "selection_rows": int(len(sel_idx)),
        "test_rows": N_TEST,
        "max_depth": MAX_DEPTH,
        "interaction_rank": RANK,
        "passes": PASSES,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "regularization": {
            "branch_cost": BRANCH_COST,
            "oblique_cost": OBLIQUE_COST,
            "affine_cost": AFFINE_COST,
            "interaction_cost": INTERACTION_COST,
            "value_l2": VALUE_L2,
        },
        "best_epoch": best_epoch,
        "best_selection_nll": best_sel,
        "universal_tree": {
            "ranking": tm,
            "structure": model.structure_summary(),
            "delta_nll_vs_catboost": tm["nll"] - cm["nll"],
            "delta_auc_vs_catboost": tm["auc"] - cm["auc"],
            "catboost_to_bayes_gap_recovered": float((cm["nll"] - tm["nll"]) / gap) if gap > 1e-8 else 0.0,
        },
        "catboost_comparator": {
            "trees": int(cat.tree_count_),
            "ranking": cm,
        },
        "bayes": bayes,
        "catboost_to_bayes_nll_gap": gap,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=REGIMES, required=True)
    p.add_argument("--seed", type=int, default=733)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.regime, a.seed, a.out)
