"""Self-play rollout collection for PPO training.

Plays one game between a learner model and an opponent (heuristic or
model snapshot). Logs per-sub-move (edge_features, action_mask, action_idx,
logprob, value) tuples for the learner's seat.

MDP step = one sub-move. Episode reward is terminal-only: normalized score
margin (my_ships - opp_ships) / (my_ships + opp_ships).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import random

import numpy as np
import torch
from torch.distributions import Categorical
from kaggle_environments import make

from action_space import MAX_MODEL_MOVES
from agents import _mutable_obs
from harness import GameView, FEATURE_ETA, FEATURE_SHIPS_NEEDED, FEATURE_TGT_PRODUCTION
from model import OrbitWarsTransformer


SAFETY_MARGIN = 1


@dataclass
class SubmoveRecord:
    """Single sub-move trajectory tuple for PPO."""
    edge_features: np.ndarray       # (50, 50, 11) float32
    action_mask: np.ndarray         # (50, 50) bool
    legal_mask: np.ndarray          # (50, 50) bool
    action_idx: int                 # in [0, 2500], 2500 = stop
    logprob: float
    value: float
    reward: float = 0.0
    done: bool = False


@dataclass
class GameTrajectory:
    """Per-game trajectory for learner's seat."""
    records: list[SubmoveRecord]
    learner_seat: int
    final_margin: float             # (my_ships - opp_ships) / total_ships
    turns: int
    opponent_name: str


def _count_ships(obs, seat: int) -> int:
    """Total ships owned by a player (planets + fleets)."""
    total = 0
    planets = obs.get("planets") or []
    fleets = obs.get("fleets") or []
    for p in planets:
        if int(p[1]) == seat:
            total += int(p[5])
    for f in fleets:
        if int(f[1]) == seat:
            total += int(f[6])
    return total


def _rollout_submoves(
    model: OrbitWarsTransformer,
    view: GameView,
    records: list[SubmoveRecord],
    deterministic: bool = False,
    device: str = "cpu",
) -> list[list]:
    """Generate learner's sub-moves, logging each for PPO.

    Returns list of env actions [src_planet_id, angle, ships].
    """
    was_training = model.training
    model.eval()
    env_actions = []
    try:
        for _ in range(MAX_MODEL_MOVES):
            action_mask = view.action_mask(SAFETY_MARGIN)
            if not action_mask.any():
                break

            # Snapshot state before this sub-move.
            ef_np = view.edge_features.copy()
            lm_np = view.legal_mask.copy()
            am_np = action_mask.copy()

            ef = torch.as_tensor(ef_np, dtype=torch.float32, device=device).unsqueeze(0)
            lm = torch.as_tensor(lm_np, dtype=torch.bool, device=device).unsqueeze(0)
            am = torch.as_tensor(am_np, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                out = model(ef, lm, am)
            logits = out["action_logits"][0]
            value = float(out["value"][0].item())

            # Sample action from masked distribution.
            dist = Categorical(logits=logits)
            if deterministic:
                action_idx = int(logits.argmax().item())
            else:
                action_idx = int(dist.sample().item())
            logprob = float(dist.log_prob(torch.tensor(action_idx, device=device, dtype=torch.long)).item())

            records.append(SubmoveRecord(
                edge_features=ef_np,
                action_mask=am_np,
                legal_mask=lm_np,
                action_idx=action_idx,
                logprob=logprob,
                value=value,
            ))

            # Check for stop action.
            stop_idx = view.n_max * view.n_max
            if action_idx == stop_idx:
                break

            # Decode to env action and apply to view.
            src_slot = action_idx // view.n_max
            tgt_slot = action_idx % view.n_max
            ships = view.deterministic_ship_count(src_slot, tgt_slot, SAFETY_MARGIN)
            action = view.apply_planned_move(src_slot, tgt_slot, ships)
            if action is None:
                break
            env_actions.append(action)
    finally:
        model.train(was_training)
    return env_actions


def play_one_game(
    learner_model: OrbitWarsTransformer,
    opponent_fn: Callable,
    opponent_name: str,
    learner_seat: Optional[int] = None,
    device: str = "cpu",
    deterministic: bool = False,
    max_turns: int = 500,
) -> GameTrajectory:
    """Play one game, logging learner's trajectory.

    Args:
        learner_model: The model being trained.
        opponent_fn: Callable(obs) -> [[src, angle, ships], ...].
        opponent_name: For logging (e.g. "heuristic", "snapshot_3").
        learner_seat: Player 0 or 1; randomized if None.
        device: CPU or MPS.
        deterministic: If True, argmax policy; else sample.
        max_turns: Episode length cap.

    Returns:
        GameTrajectory with records list, final margin, etc.
    """
    env = make("orbit_wars", debug=False)
    env.reset()
    env.step([[], []])  # Populate initial state.

    if learner_seat is None:
        learner_seat = random.randint(0, 1)
    opp_seat = 1 - learner_seat

    learner_view: Optional[GameView] = None
    records: list[SubmoveRecord] = []

    while not env.done:
        obs_pair = [env.state[0].observation, env.state[1].observation]
        learner_obs = obs_pair[learner_seat]
        opp_obs = obs_pair[opp_seat]

        step = int(learner_obs.get("step", 0))
        if step >= max_turns:
            break

        # Update learner view.
        learner_mut = _mutable_obs(learner_obs)
        if learner_view is None:
            learner_view = GameView(learner_mut)
        else:
            learner_view.update_from_obs(learner_mut)

        # Learner's sub-moves.
        learner_action = _rollout_submoves(
            learner_model, learner_view, records, deterministic, device
        )

        # Opponent's action.
        opp_action = opponent_fn(opp_obs)

        # Step environment.
        actions = [None, None]
        actions[learner_seat] = learner_action
        actions[opp_seat] = opp_action
        env.step(actions)

    # Terminal reward: normalized score margin.
    final_obs = env.state[learner_seat].observation
    my_ships = _count_ships(final_obs, learner_seat)
    opp_ships = _count_ships(final_obs, opp_seat)
    total_ships = my_ships + opp_ships
    margin = (my_ships - opp_ships) / max(1.0, total_ships) if total_ships > 0 else 0.0

    # Assign terminal reward to last sub-move.
    if records:
        records[-1].reward = margin
        records[-1].done = True

    return GameTrajectory(
        records=records,
        learner_seat=learner_seat,
        final_margin=margin,
        turns=int(final_obs.get("step", 0)),
        opponent_name=opponent_name,
    )
