"""Chunked-policy rollout collection for Orbit Wars PPO."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Optional

import numpy as np
import torch
from kaggle_environments import make
from torch.distributions import Bernoulli, Categorical

from orbit_wars.cpu.agents import heuristic_agent_cpu
from orbit_wars.cpu.chunked_model import ChunkedEdgePolicy, decode_chunk_actions, multiplier_from_delta
from orbit_wars.cpu.harness import FEATURE_DIM, GameView_CPU


def _get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


@dataclass
class ChunkedTurnRecord:
    """One learner turn with a chunk of slot decisions."""

    edges: np.ndarray
    src_ids: np.ndarray
    tgt_ids: np.ndarray
    n_tokens: int
    pointer_idx: np.ndarray
    active: np.ndarray
    ship_delta: np.ndarray
    logprob: float
    value: float
    reward: float = 0.0
    done: bool = False
    sampled_active: int = 0
    decoded_moves: int = 0
    dropped_slots: int = 0

    def __post_init__(self):
        if self.edges.ndim != 2 or self.edges.shape[1] != FEATURE_DIM:
            raise ValueError(f"bad edges shape: {self.edges.shape}")
        if self.src_ids.shape != (self.n_tokens,):
            raise ValueError(f"bad src_ids shape: {self.src_ids.shape}")
        if self.tgt_ids.shape != (self.n_tokens,):
            raise ValueError(f"bad tgt_ids shape: {self.tgt_ids.shape}")
        if self.pointer_idx.ndim != 1 or self.active.shape != self.pointer_idx.shape:
            raise ValueError("pointer_idx and active must be 1D with same shape")
        if self.ship_delta.shape != self.pointer_idx.shape:
            raise ValueError("ship_delta shape must match pointer_idx")
        if not ((0 <= self.pointer_idx) & (self.pointer_idx <= self.n_tokens)).all():
            raise ValueError("pointer_idx outside [0, n_tokens]")


@dataclass
class ChunkedGameTrajectory:
    records: list[ChunkedTurnRecord]
    learner_seat: int
    final_margin: float
    turns: int
    opponent_name: str


def reset_agent(agent):
    if hasattr(agent, "reset"):
        agent.reset()


def _count_ships(obs, seat: int) -> int:
    total = 0
    for p in _get(obs, "planets", []) or []:
        if int(p[1]) == seat:
            total += int(p[5])
    for f in _get(obs, "fleets", []) or []:
        if int(f[1]) == seat:
            total += int(f[6])
    return total


def _compute_phi(obs, learner_seat: int, opp_seat: int) -> float:
    my_ships = _count_ships(obs, learner_seat)
    opp_ships = _count_ships(obs, opp_seat)
    total = my_ships + opp_ships
    if total <= 0:
        return 0.0
    return (my_ships - opp_ships) / total


def _bundle_to_tensors(bundle, device: torch.device):
    valid_mask = torch.ones(1, bundle.n, dtype=torch.bool, device=device)
    return (
        torch.from_numpy(bundle.edges.astype(np.float32, copy=False)).unsqueeze(0).to(device),
        torch.from_numpy(bundle.src_ids.astype(np.int64, copy=False)).unsqueeze(0).to(device),
        torch.from_numpy(bundle.tgt_ids.astype(np.int64, copy=False)).unsqueeze(0).to(device),
        valid_mask,
    )


def sample_chunked_turn(
    model: ChunkedEdgePolicy,
    view: GameView_CPU,
    device: torch.device,
    deterministic: bool = False,
    max_slots: int | None = None,
) -> tuple[list[list], ChunkedTurnRecord | None]:
    """Sample one full-turn chunk and return ``(env_actions, record)``."""
    bundle = view.tokens()
    n = int(bundle.n)
    if n == 0:
        return [], None

    edges_np = np.asarray(bundle.edges, dtype=np.float32).copy()
    src_np = np.asarray(bundle.src_ids, dtype=np.int64).copy()
    tgt_np = np.asarray(bundle.tgt_ids, dtype=np.int64).copy()

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            edges, src_ids, tgt_ids, valid_mask = _bundle_to_tensors(bundle, device)
            out = model(edges, src_ids, tgt_ids, valid_mask=valid_mask, compute_value=True)
    finally:
        model.train(was_training)

    k_total = int(out.pointer_logits.shape[1])
    k = min(k_total, int(max_slots) if max_slots is not None else k_total)
    pointer_logits = out.pointer_logits[0, :k]
    active_logits = out.active_logits[0, :k]
    ship_delta_mu = out.ship_delta_mu[0, :k]
    value = float(out.value[0].item())

    pointer_dist = Categorical(logits=pointer_logits)
    active_dist = Bernoulli(logits=active_logits)
    if deterministic:
        pointer_t = torch.argmax(pointer_logits, dim=-1)
        active_t = active_logits > 0.0
    else:
        pointer_t = pointer_dist.sample()
        active_t = active_dist.sample().bool()

    pointer_logprob = pointer_dist.log_prob(pointer_t)
    active_logprob = active_dist.log_prob(active_t.to(active_logits.dtype))
    pointer_for_active = pointer_logprob * active_t.to(pointer_logprob.dtype)
    logprob = float((active_logprob + pointer_for_active).sum().item())

    pointer_idx = pointer_t.detach().cpu().numpy().astype(np.int64)
    active = active_t.detach().cpu().numpy().astype(np.bool_)
    ship_delta = ship_delta_mu.detach().cpu().numpy().astype(np.float32)
    multipliers = [multiplier_from_delta(float(delta)) for delta in ship_delta]
    env_actions = decode_chunk_actions(view, pointer_idx.tolist(), multipliers, active=active.tolist())

    sampled_active = int(active.sum())
    record = ChunkedTurnRecord(
        edges=edges_np,
        src_ids=src_np,
        tgt_ids=tgt_np,
        n_tokens=n,
        pointer_idx=pointer_idx,
        active=active,
        ship_delta=ship_delta,
        logprob=logprob,
        value=value,
        sampled_active=sampled_active,
        decoded_moves=len(env_actions),
        dropped_slots=max(0, sampled_active - len(env_actions)),
    )
    return env_actions, record


def play_one_game_chunked(
    learner_model: ChunkedEdgePolicy,
    opponent_fn: Callable = heuristic_agent_cpu,
    opponent_name: str = "heuristic",
    learner_seat: Optional[int] = None,
    device: torch.device | str = "cpu",
    deterministic: bool = False,
    max_turns: int = 500,
    max_slots: int | None = None,
) -> ChunkedGameTrajectory:
    if not isinstance(device, torch.device):
        device = torch.device(device)

    reset_agent(opponent_fn)
    env = make("orbit_wars", debug=False)
    env.reset()
    env.step([[], []])

    if learner_seat is None:
        learner_seat = random.randint(0, 1)
    opp_seat = 1 - learner_seat

    learner_view: Optional[GameView_CPU] = None
    records: list[ChunkedTurnRecord] = []

    initial_obs = env.state[learner_seat].observation
    phi_old = _compute_phi(initial_obs, learner_seat, opp_seat)

    while not env.done:
        obs_pair = [env.state[0].observation, env.state[1].observation]
        learner_obs = obs_pair[learner_seat]
        opp_obs = obs_pair[opp_seat]
        step = int(_get(learner_obs, "step", 0))
        if step >= max_turns:
            break

        if learner_view is None:
            learner_view = GameView_CPU(learner_obs)
        else:
            learner_view.update_from_obs(learner_obs)

        learner_action, record = sample_chunked_turn(
            learner_model,
            learner_view,
            device=device,
            deterministic=deterministic,
            max_slots=max_slots,
        )
        opp_action = opponent_fn(opp_obs)

        actions = [[], []]
        actions[learner_seat] = learner_action
        actions[opp_seat] = opp_action
        env.step(actions)

        if record is not None:
            post_obs = env.state[learner_seat].observation
            phi_new = _compute_phi(post_obs, learner_seat, opp_seat)
            record.reward += phi_new - phi_old
            phi_old = phi_new
            records.append(record)

    final_obs = env.state[learner_seat].observation
    my_ships = _count_ships(final_obs, learner_seat)
    opp_ships = _count_ships(final_obs, opp_seat)
    total_ships = my_ships + opp_ships
    margin = (my_ships - opp_ships) / max(1.0, total_ships)
    if records and env.done:
        records[-1].done = True

    final_step = (
        _get(env.state[0].observation, "step")
        or _get(env.state[1].observation, "step")
        or len(env.steps)
    )
    return ChunkedGameTrajectory(
        records=records,
        learner_seat=learner_seat,
        final_margin=margin,
        turns=int(final_step),
        opponent_name=opponent_name,
    )


def play_one_game_chunked_worker(task: dict) -> ChunkedGameTrajectory:
    import random as _rnd

    torch.set_num_threads(1)
    seed = int(task["seed"])
    _rnd.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    model = ChunkedEdgePolicy(n_slots=int(task.get("n_slots", 10))).to(device)
    model.load_state_dict(task["model_state"])
    model.eval()
    return play_one_game_chunked(
        model,
        heuristic_agent_cpu,
        "heuristic",
        device=device,
        deterministic=bool(task["deterministic"]),
        max_turns=int(task["max_turns"]),
        max_slots=int(task.get("max_slots", task.get("n_slots", 10))),
    )
