"""Self-play rollout collection for the CPU dynamic-edge stack."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Optional

import numpy as np
import torch
from kaggle_environments import make
from torch.distributions import Categorical

from orbit_wars.core.action_space import MAX_MODEL_MOVES
from orbit_wars.cpu.harness import FEATURE_DIM, GameView_CPU
from orbit_wars.cpu.model import OrbitWarsEdgeTransformer


def _get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


@dataclass
class SubmoveRecord:
    """Single learner decision in ragged CPU token space."""

    edges: np.ndarray
    src_ids: np.ndarray
    tgt_ids: np.ndarray
    n_tokens: int
    action_idx: int
    logprob: float
    value: float
    reward: float = 0.0
    done: bool = False

    def __post_init__(self):
        if self.edges.ndim != 2 or self.edges.shape[1] != FEATURE_DIM:
            raise ValueError(f"bad edges shape: {self.edges.shape}")
        if self.src_ids.shape != (self.n_tokens,):
            raise ValueError(f"bad src_ids shape: {self.src_ids.shape}")
        if self.tgt_ids.shape != (self.n_tokens,):
            raise ValueError(f"bad tgt_ids shape: {self.tgt_ids.shape}")
        if not (0 <= self.action_idx <= self.n_tokens):
            raise ValueError(
                f"action_idx {self.action_idx} outside [0, {self.n_tokens}]"
            )


@dataclass
class GameTrajectory:
    records: list[SubmoveRecord]
    learner_seat: int
    final_margin: float
    turns: int
    opponent_name: str


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
    return (
        torch.from_numpy(bundle.edges.astype(np.float32, copy=False))
        .unsqueeze(0)
        .to(device),
        torch.from_numpy(bundle.src_ids.astype(np.int64, copy=False))
        .unsqueeze(0)
        .to(device),
        torch.from_numpy(bundle.tgt_ids.astype(np.int64, copy=False))
        .unsqueeze(0)
        .to(device),
    )


def rollout_submoves(
    model: OrbitWarsEdgeTransformer,
    view: GameView_CPU,
    records: list[SubmoveRecord],
    deterministic: bool,
    device: torch.device,
    max_moves: int = MAX_MODEL_MOVES,
) -> list[list]:
    """Generate learner moves for one env turn and append PPO records."""

    was_training = model.training
    model.eval()
    env_actions: list[list] = []

    try:
        for _ in range(max_moves):
            bundle = view.tokens()
            n = int(bundle.n)
            if n == 0:
                break

            edges_np = np.asarray(bundle.edges, dtype=np.float32).copy()
            src_np = np.asarray(bundle.src_ids, dtype=np.int64).copy()
            tgt_np = np.asarray(bundle.tgt_ids, dtype=np.int64).copy()

            with torch.no_grad():
                logits, value = model(
                    *_bundle_to_tensors(bundle, device),
                    compute_value=True,
                )
            logits_1 = logits[0]
            value_1 = float(value[0].item())

            dist = Categorical(logits=logits_1)
            if deterministic:
                action_idx = int(logits_1.argmax().item())
            else:
                action_idx = int(dist.sample().item())
            logprob = float(
                dist.log_prob(
                    torch.tensor(action_idx, device=device, dtype=torch.long)
                ).item()
            )

            records.append(
                SubmoveRecord(
                    edges=edges_np,
                    src_ids=src_np,
                    tgt_ids=tgt_np,
                    n_tokens=n,
                    action_idx=action_idx,
                    logprob=logprob,
                    value=value_1,
                )
            )

            if action_idx == n:
                break

            action = view.apply_planned_move(action_idx)
            if action is None:
                break
            env_actions.append(action)
    finally:
        model.train(was_training)

    return env_actions


def reset_agent(agent):
    if hasattr(agent, "reset"):
        agent.reset()


def play_one_game(
    learner_model: OrbitWarsEdgeTransformer,
    opponent_fn: Callable,
    opponent_name: str,
    learner_seat: Optional[int] = None,
    device: torch.device | str = "cpu",
    deterministic: bool = False,
    max_turns: int = 500,
) -> GameTrajectory:
    """Play one game and return learner submove records for PPO."""

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
    records: list[SubmoveRecord] = []

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

        n_records_before = len(records)
        learner_action = rollout_submoves(
            learner_model,
            learner_view,
            records,
            deterministic=deterministic,
            device=device,
        )
        opp_action = opponent_fn(opp_obs)

        actions = [[], []]
        actions[learner_seat] = learner_action
        actions[opp_seat] = opp_action
        env.step(actions)

        if len(records) > n_records_before:
            post_obs = env.state[learner_seat].observation
            phi_new = _compute_phi(post_obs, learner_seat, opp_seat)
            records[-1].reward += phi_new - phi_old
            phi_old = phi_new

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

    return GameTrajectory(
        records=records,
        learner_seat=learner_seat,
        final_margin=margin,
        turns=int(final_step),
        opponent_name=opponent_name,
    )


def play_one_game_worker(task: dict) -> GameTrajectory:
    """Picklable entrypoint: build model + opponent locally and play one game.

    Intended for use via ProcessPoolExecutor. Each worker forces CPU with a
    single BLAS thread to avoid oversubscription when N workers run in parallel.
    """
    import random as _rnd

    torch.set_num_threads(1)

    seed = int(task["seed"])
    _rnd.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    model = OrbitWarsEdgeTransformer().to(device)
    model.load_state_dict(task["model_state"])
    model.eval()

    from orbit_wars.cpu.rl_opponent_pool import OpponentPool

    pool = OpponentPool(
        heuristic_weight=task["heuristic_weight"],
        max_snapshots=task["max_snapshots"],
    )
    pool_state = task.get("pool_state")
    if pool_state is not None:
        pool.load_state_dict(pool_state)
    opp_fn, opp_name = pool.sample(device=device)

    return play_one_game(
        model,
        opp_fn,
        opp_name,
        device=device,
        deterministic=bool(task["deterministic"]),
        max_turns=int(task["max_turns"]),
    )
