"""Epicycle-to-law representation competition after interpolation.

This is a mechanism experiment for the user's hypothesis, not by itself a
TorchBoost deep-double-descent result. A deliberately overparameterized,
orbit-specific Fourier representation is warm-started at (near) interpolation.
A compact shared central-force branch competes with it. Regularization penalizes
Fourier complexity but does not encode the inverse-square exponent; the data
must identify that exponent across radii/orbits.

The decisive controls are: no regularization, frozen law branch, and cold start.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _eccentric_anomaly(mean_anomaly, eccentricity):
    e = np.full_like(mean_anomaly, eccentricity)
    E = mean_anomaly.copy()
    for _ in range(20):
        E -= (E - e * np.sin(E) - mean_anomaly) / (1. - e * np.cos(E))
    return E


def orbit_samples(a, e, mu, phases):
    E = _eccentric_anomaly(phases, e)
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1. - e * e) * np.sin(E)
    pos = np.c_[x, y]
    radius = np.linalg.norm(pos, axis=1)
    accel = -mu * pos / radius[:, None] ** 3
    return pos.astype("float32"), accel.astype("float32")


def fourier_design(phases, order):
    cols = [np.ones_like(phases)]
    for k in range(1, order + 1):
        cols.extend((np.cos(k * phases), np.sin(k * phases)))
    return np.stack(cols, axis=1).astype("float32")


@dataclass
class OrbitData:
    position: torch.Tensor
    phase: torch.Tensor
    mu: torch.Tensor
    orbit: torch.Tensor
    target: torch.Tensor


def make_data(seed=17, train_orbits=8, points=112, noise=.025):
    rng = np.random.default_rng(seed)
    train_rows, hold_rows = [], []
    params = []
    for orbit in range(train_orbits):
        a = rng.uniform(.7, 1.8)
        e = rng.uniform(.08, .62)
        mu = rng.uniform(.75, 1.3)
        params.append((a, e, mu))
        phase = np.linspace(0., 2 * np.pi, points, endpoint=False)
        pos, accel = orbit_samples(a, e, mu, phase)
        noisy = accel + rng.normal(0., noise * np.std(accel, axis=0), accel.shape)
        # A contiguous held-out arc plus periodic subsampling makes memorization
        # possible without making generalization trivial.
        hold = ((phase > 2.25) & (phase < 3.15)) | ((np.arange(points) + orbit) % 9 == 0)
        for idx in np.where(~hold)[0]:
            train_rows.append((pos[idx], phase[idx], mu, orbit, noisy[idx]))
        for idx in np.where(hold)[0]:
            hold_rows.append((pos[idx], phase[idx], mu, orbit, accel[idx]))

    # Unseen orbital regimes: no orbit-specific Fourier slot is available.
    ood_rows = []
    for j, (a, e, mu) in enumerate(((2.15, .72, .9), (2.45, .18, 1.2), (.52, .48, 1.05))):
        phase = np.linspace(0., 2 * np.pi, points, endpoint=False)
        pos, accel = orbit_samples(a, e, mu, phase)
        for idx in range(points):
            ood_rows.append((pos[idx], phase[idx], mu, -1, accel[idx]))

    def pack(rows):
        return OrbitData(
            torch.tensor(np.stack([r[0] for r in rows]), dtype=torch.float32),
            torch.tensor([r[1] for r in rows], dtype=torch.float32),
            torch.tensor([r[2] for r in rows], dtype=torch.float32),
            torch.tensor([r[3] for r in rows], dtype=torch.long),
            torch.tensor(np.stack([r[4] for r in rows]), dtype=torch.float32),
        )

    return pack(train_rows), pack(hold_rows), pack(ood_rows), params


class CompetingOrbitModel(torch.nn.Module):
    def __init__(self, n_orbits, order, warm_coeff=None, law_scale=.03, alpha=.6):
        super().__init__()
        width = 1 + 2 * order
        coeff = torch.zeros(n_orbits, width, 2)
        if warm_coeff is not None:
            coeff.copy_(torch.as_tensor(warm_coeff))
        self.epicycle = torch.nn.Parameter(coeff)
        self.law_scale = torch.nn.Parameter(torch.tensor(float(law_scale)))
        self.alpha = torch.nn.Parameter(torch.tensor(float(alpha)))
        self.order = order

    def basis(self, phase):
        columns = [torch.ones_like(phase)]
        for k in range(1, self.order + 1):
            columns.extend((torch.cos(k * phase), torch.sin(k * phase)))
        return torch.stack(columns, dim=1)

    def forward(self, data):
        radius = torch.linalg.vector_norm(data.position, dim=1).clamp_min(1e-5)
        law = -self.law_scale * data.mu[:, None] * data.position / radius[:, None] ** (self.alpha + 1.)
        epi = torch.zeros_like(law)
        known = data.orbit >= 0
        if bool(known.any()):
            design = self.basis(data.phase[known])
            coeff = self.epicycle[data.orbit[known]]
            epi[known] = torch.einsum("nb,nbd->nd", design, coeff)
        return law + epi

    def complexity(self):
        weights = [0.]
        for k in range(1, self.order + 1):
            weights.extend((float(k * k), float(k * k)))
        w = self.epicycle.new_tensor(weights)[None, :, None]
        return (w * self.epicycle.square()).mean()

    def FourierDiagnostics(self):
        energy = self.epicycle.detach().square().sum((0, 2)).cpu().numpy()
        by_order = np.zeros(self.order + 1)
        by_order[0] = energy[0]
        for k in range(1, self.order + 1):
            by_order[k] = energy[2 * k - 1] + energy[2 * k]
        total = float(by_order.sum()) + 1e-30
        effective = float(np.dot(np.arange(self.order + 1), by_order) / total)
        high = float(by_order[max(1, self.order // 2):].sum() / total)
        participation = float(total * total / (np.square(by_order).sum() + 1e-30))
        return {
            "fourier_energy": total,
            "effective_order": effective,
            "high_order_fraction": high,
            "active_order_participation": participation,
        }


def warm_start_coeff(train, n_orbits, order):
    coeff = np.zeros((n_orbits, 1 + 2 * order, 2), dtype="float32")
    phase = train.phase.numpy()
    target = train.target.numpy()
    orbit = train.orbit.numpy()
    for j in range(n_orbits):
        mask = orbit == j
        design = fourier_design(phase[mask], order)
        # Minimum-norm overparameterized interpolant.
        coeff[j] = np.linalg.lstsq(design, target[mask], rcond=None)[0].astype("float32")
    return coeff


def mse(model, data):
    with torch.no_grad():
        return float((model(data) - data.target).square().mean())


def run_variant(name, train, hold, ood, n_orbits, order=48, steps=900, regularization=3e-4, seed=17):
    torch.manual_seed(seed)
    warm = name.startswith("warm")
    coeff = warm_start_coeff(train, n_orbits, order) if warm else None
    model = CompetingOrbitModel(n_orbits, order, coeff)
    if name == "warm_reg_frozen_law":
        model.law_scale.requires_grad_(False)
        model.alpha.requires_grad_(False)
    lam = 0. if name == "warm_no_reg" else regularization
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.025)
    checkpoints = []
    first_interpolation = None
    initial_data_loss = mse(model, train)
    for step in range(steps + 1):
        if step % 15 == 0 or step == steps:
            d = model.FourierDiagnostics()
            d.update({
                "step": step,
                "train_mse": mse(model, train),
                "holdout_mse": mse(model, hold),
                "ood_mse": mse(model, ood),
                "regularizer": float(model.complexity().detach()),
                "alpha": float(model.alpha.detach()),
                "law_scale": float(model.law_scale.detach()),
            })
            checkpoints.append(d)
            if first_interpolation is None and d["train_mse"] <= max(1e-8, initial_data_loss * 1.05):
                first_interpolation = step
        if step == steps:
            break
        optimizer.zero_grad()
        data_loss = (model(train) - train.target).square().mean()
        loss = data_loss + lam * model.complexity()
        loss.backward()
        optimizer.step()

    hold_curve = np.asarray([x["holdout_mse"] for x in checkpoints])
    train_curve = np.asarray([x["train_mse"] for x in checkpoints])
    # Only label a time-domain second descent if a post-interpolation peak is
    # followed by a material late improvement while train fit remains near its floor.
    start_index = 0
    if first_interpolation is not None:
        start_index = max(i for i, x in enumerate(checkpoints) if x["step"] <= first_interpolation)
    post = hold_curve[start_index:]
    peak_index = int(np.argmax(post)) + start_index
    second_descent = bool(
        peak_index < len(hold_curve) - 2
        and hold_curve[peak_index] > 1.10 * hold_curve[-1]
        and train_curve[-1] <= max(5. * train_curve.min(), 2e-4)
    )
    return {
        "name": name,
        "initial": checkpoints[0],
        "final": checkpoints[-1],
        "first_interpolation_step": first_interpolation,
        "second_descent_observed": second_descent,
        "peak_holdout_step": checkpoints[peak_index]["step"],
        "peak_holdout_mse": float(hold_curve[peak_index]),
        "trajectory": checkpoints,
    }


def run(seed=17):
    train, hold, ood, params = make_data(seed=seed)
    n_orbits = len(params)
    variants = [
        run_variant("warm_reg", train, hold, ood, n_orbits, seed=seed),
        run_variant("warm_no_reg", train, hold, ood, n_orbits, seed=seed),
        run_variant("warm_reg_frozen_law", train, hold, ood, n_orbits, seed=seed),
        run_variant("cold_reg", train, hold, ood, n_orbits, seed=seed),
    ]
    return {
        "seed": seed,
        "train_rows": len(train.target),
        "holdout_rows": len(hold.target),
        "ood_rows": len(ood.target),
        "order": 48,
        "variants": variants,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
