"""Run-owned delayed contextual experiments, optionally computed on a CPU worker.

Workers receive detached immutable observations and copied linear-bandit state.
Only the training thread may apply a versioned proposal. Rewards are retained
observational utility changes, not fabricated gradients or causal guarantees.
"""
from __future__ import annotations
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import math
import numpy as np

from .config import OnlineConfig
from .contracts import Observation, Proposal, TrialOutcome


ACTIONS = (
    {},
    {"stiffness": .5},
    {"yield_threshold": .5},
    {"mobility": 2.},
    {"consolidation_rate": 2.},
    {"stiffness": 2.},
)
FEATURES = 22


def context(observation: Observation) -> tuple[float, ...]:
    o = observation
    return (1., math.tanh(o.utility * 100), min(1., o.occupancy), math.tanh(o.entropy),
            math.tanh(o.information), math.tanh(o.gradient_norm), math.tanh(o.update_norm),
            math.tanh(o.temperature / 2), max(-1., min(1., o.direction)), math.tanh(o.depth / 3),
            math.tanh(o.charge), math.tanh(o.heat), math.tanh(o.inductive_energy),
            math.tanh(o.hardness), o.integrity, math.tanh(o.reference_path_length),
            o.phase / 3., o.training_progress, o.structural_gate, math.tanh(o.structural_gradient),
            math.tanh(o.uncertainty), math.tanh(o.utility_change * 100.))


def _propose(snapshot: dict) -> list[Proposal]:
    rng = np.random.default_rng(snapshot["seed"])
    inverse = np.linalg.inv(snapshot["a"])
    theta = np.einsum("aij,aj->ai", inverse, snapshot["b"])
    result = []
    for index, observation in enumerate(snapshot["observations"]):
        x = np.asarray(context(observation))
        scores = theta @ x + snapshot["ucb"] * np.sqrt(np.maximum(0., np.einsum("i,aij,j->a", x, inverse, x)))
        greedy = int(np.argmax(scores))
        action = int(rng.integers(len(ACTIONS))) if rng.random() < snapshot["exploration"] else greedy
        propensity = snapshot["exploration"] / len(ACTIONS) + (1 - snapshot["exploration"] if action == greedy else 0.)
        result.append(Proposal(snapshot["run_id"], snapshot["first_trial"] + index, observation.node_id,
                               snapshot["topology_version"], snapshot["versions"][observation.node_id],
                               snapshot["step"], action, float(propensity), tuple(x.tolist())))
    return result


class OnlineScheduler:
    def __init__(self, config: OnlineConfig, run_id: str, *, seed: int = 0):
        self.config, self.run_id = config, run_id
        self.rng = np.random.default_rng(seed)
        self.a = np.stack([config.ridge * np.eye(FEATURES) for _ in ACTIONS])
        self.b = np.zeros((len(ACTIONS), FEATURES))
        self.counts = np.zeros(len(ACTIONS), dtype=np.int64)
        self.next_trial = 0
        self.cooldowns: dict[str, int] = {}
        self.history: list[dict] = []
        self.no_op_mean = 0.
        self.no_op_count = 0
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future | None = None
        self._ready: list[Proposal] = []
        self.last_request = -1

    def request(self, observations: dict[str, Observation], versions: dict[str, int], step: int,
                topology_version: int, occupied: set[str]) -> bool:
        cfg = self.config
        if not cfg.enabled or step % cfg.interval or step <= self.last_request:
            return False
        if self._future is not None or self._ready:
            return False
        candidates = [o for o in observations.values() if o.node_id in versions and not o.frozen
                      and o.topology_version == topology_version
                      and o.occupancy > .01 and o.node_id not in occupied
                      and step >= self.cooldowns.get(o.node_id, -1)]
        candidates.sort(key=lambda o: (-(o.gradient_norm + max(0., -o.utility)), o.node_id))
        candidates = tuple(candidates[:max(0, cfg.max_trials - len(occupied))])
        if not candidates:
            return False
        snapshot = {"observations": candidates, "versions": versions.copy(), "step": step,
                    "topology_version": topology_version, "run_id": self.run_id,
                    "a": self.a.copy(), "b": self.b.copy(), "seed": int(self.rng.integers(2**32)),
                    "ucb": cfg.ucb, "exploration": cfg.exploration, "first_trial": self.next_trial}
        self.next_trial += len(candidates)
        self.last_request = step
        if cfg.concurrent:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="torchboost-policy")
            self._future = self._executor.submit(_propose, snapshot)
        else:
            self._ready = _propose(snapshot)
        return True

    def collect(self, *, wait: bool = True) -> list[Proposal]:
        if self._future is not None:
            if not wait and not self._future.done():
                return []
            self._ready.extend(self._future.result())
            self._future = None
        result, self._ready = self._ready, []
        return result

    def valid(self, proposal: Proposal, *, topology_version: int, versions: dict[str, int],
              step: int) -> bool:
        valid = (proposal.run_id == self.run_id and proposal.topology_version == topology_version
                 and proposal.node_id in versions and proposal.settings_version == versions[proposal.node_id]
                 and 0 <= step - proposal.step <= self.config.interval + 1)
        if not valid:
            self.history.append({"event": "stale_proposal", **asdict(proposal), "rejected_at": step})
        return valid

    def accepted(self, proposal: Proposal, step: int) -> None:
        self.cooldowns[proposal.node_id] = step + self.config.window + self.config.cooldown
        self.history.append({"event": "trial_started", **asdict(proposal), "applied_at": step,
                             "factors": deepcopy(ACTIONS[proposal.action])})

    def learn(self, outcomes: list[TrialOutcome], step: int) -> None:
        cfg = self.config
        # A no-op cohort provides a descriptive baseline for coincident training
        # progress. This is not a randomized controlled causal estimator.
        for outcome in outcomes:
            if outcome.action == 0:
                self.no_op_count += 1
                self.no_op_mean += (outcome.reward - self.no_op_mean) / self.no_op_count
        for outcome in outcomes:
            reward = float(np.clip((outcome.reward - self.no_op_mean) / .01, -1., 1.))
            x = np.asarray(outcome.context)
            arm = outcome.action
            self.a[arm] = cfg.forgetting * self.a[arm] + (1 - cfg.forgetting) * cfg.ridge * np.eye(FEATURES) + np.outer(x, x)
            self.b[arm] = cfg.forgetting * self.b[arm] + reward * x
            self.counts[arm] += 1
            self.history.append({"event": "trial_outcome", **asdict(outcome), "step": step,
                                 "normalized_reward": reward, "no_op_baseline": self.no_op_mean})

    def synchronize(self, alive: set[str]) -> None:
        self.cooldowns = {k: v for k, v in self.cooldowns.items() if k in alive}

    def state_dict(self) -> dict:
        # A checkpoint is a declared safe barrier. It waits for proposals, but
        # never applies them or changes learned policy/model state.
        if self._future is not None:
            self._ready.extend(self._future.result())
            self._future = None
        return {"run_id": self.run_id, "a": self.a.tolist(), "b": self.b.tolist(),
                "counts": self.counts.tolist(), "next_trial": self.next_trial,
                "cooldowns": self.cooldowns.copy(), "history": deepcopy(self.history),
                "no_op_mean": self.no_op_mean, "no_op_count": self.no_op_count,
                "ready": [asdict(p) for p in self._ready], "last_request": self.last_request,
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.close()
        self.run_id = state["run_id"]
        self.a, self.b, self.counts = np.asarray(state["a"]), np.asarray(state["b"]), np.asarray(state["counts"])
        self.next_trial, self.cooldowns, self.history = state["next_trial"], state["cooldowns"].copy(), deepcopy(state["history"])
        self.no_op_mean, self.no_op_count = state["no_op_mean"], state["no_op_count"]
        self._ready = [Proposal(**p) for p in state["ready"]]
        self.last_request = state["last_request"]
        self.rng.bit_generator.state = state["rng"]

    def close(self) -> None:
        if self._future is not None:
            self._ready.extend(self._future.result())
            self._future = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
