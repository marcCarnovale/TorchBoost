"""Read-only single-tree diagnostics. Never advances optimizer or controller state."""
from __future__ import annotations
import math
import numpy as np
import torch
from .single_tree import PackedSingleTree


def _leaf_values(model):
    if model.readout == 'leaf':
        return model.values.detach()
    values = model.values.detach().clone()
    for i in range(1, model.n_nodes):
        values[i] += values[(i - 1) // model.arity]
    return values[model.n_internal:]


def _scores(logits, split, objective):
    losses = objective.loss(logits, split.y).detach()
    w = split.weight / split.weight.sum()
    result = {'loss': float((losses * w).sum())}
    if objective.task != 'regression':
        p = objective.response(logits).detach()
        pred = p.argmax(1)
        confidence = p.max(1).values
        correct = pred == split.y
        result.update(accuracy=float((correct.float() * w).sum()),
            errors=int((~correct & (split.weight > 0)).sum()),
            high_confidence_errors=int((~correct & (confidence >= .9) & (split.weight > 0)).sum()),
            brier=float(((p - torch.nn.functional.one_hot(split.y, p.shape[1])).square().sum(1) * w).sum()),
            mean_confidence=float((confidence * w).sum()))
        bins = torch.linspace(0, 1, 11)
        ece = 0.
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (confidence > low) & (confidence <= high)
            weight = w[mask].sum()
            if weight > 0:
                ece += float((w[mask] * (correct[mask].float() - confidence[mask])).sum().abs())
        result['ece_10_equal_width'] = ece
        confusion = torch.zeros((p.shape[1], p.shape[1]), dtype=torch.long)
        for actual, prediction in zip(split.y, pred):
            confusion[actual, prediction] += 1
        result['confusion'] = confusion.tolist()
    total = float((losses * split.weight).sum())
    k = min(5, len(losses))
    result['top5_weighted_loss_fraction'] = float((losses * split.weight).topk(k).values.sum()) / max(total, 1e-15)
    return result


def diagnose_single_tree(estimator, X_train, y_train, X_validation, y_validation,
                         *, sample_weight=None, validation_weight=None, row_ids=None):
    """Audit selected snapshot, including paired subtree collapses without retraining.

    Collapse means replacing a subtree's learned logit contribution by its
    training-occupancy-weighted conditional mean. Negative validation delta flags
    a harmful refinement for that fixed predictor, not its counterfactual training effect.
    """
    m, obj = estimator.model_, estimator.objective_
    train = estimator.preprocessor_.split(X_train, y_train, sample_weight)
    valid = estimator.preprocessor_.split(X_validation, y_validation, validation_weight)
    with torch.no_grad():
        mass_t, p_t = m.routing(train.x)
        mass_v, p_v = m.routing(valid.x)
        lt, lv = m(train.x), m(valid.x)
        ht, hv = m(train.x, hard=True), m(valid.x, hard=True)
        wt = train.weight / train.weight.sum()
        wv = valid.weight / valid.weight.sum()
        reach_t = (mass_t * wt[:, None]).sum(0)
        reach_v = (mass_v * wv[:, None]).sum(0)
        leaf_mass = reach_t[m.n_internal:]
        entropy = -(leaf_mass * leaf_mass.clamp_min(1e-30).log()).sum()
        effective = float(entropy.exp())
        hard_mass_t, _ = m.routing(train.x, hard=True)
        hard_mass_v, _ = m.routing(valid.x, hard=True)
        leaf_values = _leaf_values(m)
        output = dict(depth=m.depth, arity=m.arity, trees=1,
            allocated_nodes=m.n_nodes, leaves=m.n_leaves,
            parameter_count=sum(p.numel() for p in m.parameters()),
            effective_leaves_train=effective,
            soft_leaves_below_one_training_example=int((leaf_mass * len(train.x) < 1.).sum()),
            hard_visited_leaves_train=int((hard_mass_t[:, m.n_internal:].sum(0) > 0).sum()),
            train=_scores(lt, train, obj), validation=_scores(lv, valid, obj),
            hard_train=_scores(ht, train, obj), hard_validation=_scores(hv, valid, obj))
        output['hard_minus_soft_validation_loss'] = output['hard_validation']['loss'] - output['validation']['loss']
        if obj.task != 'regression':
            output['hard_soft_prediction_disagreement'] = float((obj.response(lv).argmax(1) != obj.response(hv).argmax(1)).float().mean())
        basis = mass_t[:, m.n_internal:].double()
        centered = basis - (basis * wt.double()[:, None]).sum(0, keepdim=True)
        sv = torch.linalg.svdvals(centered * wt.double().sqrt()[:, None])
        threshold = max(float(sv.max()) * 1e-6, 1e-12)
        output['centered_leaf_design_rank_rtol_1e6'] = int((sv > threshold).sum())
        output['leaf_design_singular_values'] = sv.tolist()
        output['routing_feature_rank_bound'] = m.n_leaves - 1
    # autograd.grad returns diagnostics without writing .grad or stepping a model.
    train_loss = obj.weighted_loss(m(train.x), train.y, train.weight)
    gw, gb, gv = torch.autograd.grad(train_loss, (m.routing_weight, m.routing_bias, m.values), allow_unused=True)
    depth_rows, node_rows = [], []
    start = 0
    with torch.no_grad():
        for d in range(m.depth + 1):
            width = m.arity ** d
            row = dict(depth=d, nodes=width, mean_train_reach=float(reach_t[start:start+width].mean()),
                min_train_reach=float(reach_t[start:start+width].min()),
                hard_visited_nodes=int((hard_mass_t[:, start:start+width].sum(0) > 0).sum()))
            if d < m.depth:
                p = p_t[:, start:start+width]
                reach = mass_t[:, start:start+width] * wt[:, None]
                row.update(normalized_conditional_entropy=float((-(p * p.clamp_min(1e-30).log()).sum(-1) * reach).sum() / math.log(m.arity)),
                    saturated_route_mass=float(((p.max(-1).values > .99).float() * reach).sum()),
                    routing_gradient_l2=float(gw[start:start+width].norm()) if gw is not None else 0.)
            # Per-depth truncation uses TRAIN conditional mean values, not validation labels.
            means = []
            for pos in range(width):
                node = start + pos
                span = m.arity ** (m.depth - d)
                lo, hi = pos * span, (pos + 1) * span
                train_contrib = mass_t[:, m.n_internal+lo:m.n_internal+hi] @ leaf_values[lo:hi]
                mean = (train_contrib * wt[:, None]).sum(0) / reach_t[node].clamp_min(1e-12)
                means.append(mean)
                if d < m.depth:
                    valid_contrib = mass_v[:, m.n_internal+lo:m.n_internal+hi] @ leaf_values[lo:hi]
                    collapsed = lv - valid_contrib + mass_v[:, node, None] * mean
                    delta = float(obj.weighted_loss(collapsed, valid.y, valid.weight)) - output['validation']['loss']
                    usage = (p_t[:, node] * mass_t[:, node, None] * wt[:, None]).sum(0) / reach_t[node].clamp_min(1e-12)
                    node_rows.append(dict(node=node, native_node_id=m.node_ids[node], depth=d,
                        train_reach=float(reach_t[node]), validation_reach=float(reach_v[node]),
                        hard_train_count=int(hard_mass_t[:, node].sum()), hard_validation_count=int(hard_mass_v[:, node].sum()),
                        branch_usage=usage.tolist(), collapse_validation_loss_delta=delta,
                        routing_gradient_l2=float(gw[node].norm()) if gw is not None else 0.))
            trunc = mass_v[:, start:start+width] @ torch.stack(means) + m.bias
            row['training_mean_truncated_validation_loss'] = float(obj.weighted_loss(trunc, valid.y, valid.weight))
            depth_rows.append(row); start += width
        loss_vector = obj.loss(lv, valid.y)
        probs = obj.response(lv) if obj.task != 'regression' else None
        cases = []
        for i in loss_vector.argsort(descending=True)[:min(25, len(valid.x))].tolist():
            case = dict(row_id=int(row_ids[i]) if row_ids is not None else i,
                loss=float(loss_vector[i]), soft_leaf_concentration=float(mass_v[i, m.n_internal:].max()),
                hard_leaf=int(hard_mass_v[i, m.n_internal:].argmax()))
            if probs is not None:
                case.update(actual=int(valid.y[i]), predicted=int(probs[i].argmax()),
                    confidence=float(probs[i].max()), actual_class_probability=float(probs[i, valid.y[i]]))
            cases.append(case)
        leaf_rows = []
        for j in range(m.n_leaves):
            hvj = hard_mass_v[:, m.n_internal+j]
            leaf_rows.append(dict(leaf=j, train_soft_mass=float(leaf_mass[j]),
                train_hard_count=int(hard_mass_t[:, m.n_internal+j].sum()),
                validation_hard_count=int(hvj.sum()),
                validation_loss_sum=float((loss_vector * hvj).sum())))
    output.update(depths=depth_rows, nodes=node_rows, leaves_detail=leaf_rows, hardest_validation_examples=cases)
    return output
