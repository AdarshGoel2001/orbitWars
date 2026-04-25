"""CPU-token agents for the dynamic-edge Orbit Wars stack."""

from __future__ import annotations

from pathlib import Path

import torch

from action_space import MAX_MODEL_MOVES
from harness_cpu import (
    FEATURE_ETA,
    FEATURE_KIND_ATTACK_ENEMY,
    FEATURE_KIND_ATTACK_NEUTRAL,
    FEATURE_KIND_REINFORCE,
    FEATURE_SHIPS_NEEDED,
    FEATURE_SRC_CAN_FUND,
    FEATURE_TGT_PRODUCTION,
    FEATURE_TGT_WILL_FALL,
    FEATURE_TURNS_LEFT,
    GameView_CPU,
)
from model_cpu import OrbitWarsEdgeTransformer


HOARD_WINDOW = 20


def choose_heuristic_token_cpu(view: GameView_CPU) -> int | None:
    """Pick one token from a planned CPU view, or None for stop.

    Priority mirrors the old teacher at a coarser level:
      1. reinforce owned planets projected to fall
      2. stop offense near the end
      3. expand by simple ROI, preferring currently fundable attacks
    """
    bundle = view.tokens()
    if bundle.n == 0:
        return None

    edges = bundle.edges

    best_defense = None
    for i, e in enumerate(edges):
        if e[FEATURE_KIND_REINFORCE] < 0.5 or e[FEATURE_TGT_WILL_FALL] < 0.5:
            continue
        eta = float(e[FEATURE_ETA])
        can_fund = float(e[FEATURE_SRC_CAN_FUND])
        # Arrival speed matters most for saves; fundability breaks ties.
        score = can_fund * 1000.0 - eta
        if best_defense is None or score > best_defense[0]:
            best_defense = (score, i)
    if best_defense is not None:
        return int(best_defense[1])

    turns_left = float(edges[0, FEATURE_TURNS_LEFT])
    if turns_left < HOARD_WINDOW:
        return None

    best_attack = None
    for i, e in enumerate(edges):
        is_attack = (
            e[FEATURE_KIND_ATTACK_NEUTRAL] > 0.5
            or e[FEATURE_KIND_ATTACK_ENEMY] > 0.5
        )
        if not is_attack:
            continue
        eta = float(e[FEATURE_ETA])
        if eta <= 0.0 or turns_left <= eta:
            continue

        can_fund = float(e[FEATURE_SRC_CAN_FUND])
        ships_needed = max(1.0, float(e[FEATURE_SHIPS_NEEDED]))
        production = float(e[FEATURE_TGT_PRODUCTION])
        enemy_bonus = 1.15 if e[FEATURE_KIND_ATTACK_ENEMY] > 0.5 else 1.0
        hold_time = turns_left - eta
        roi = enemy_bonus * production * hold_time / (ships_needed + eta + 1.0)

        # Underfunded attacks remain visible to the model, but the teacher
        # should usually label immediately capturable expansion moves.
        score = roi + can_fund * 10000.0
        if best_attack is None or score > best_attack[0]:
            best_attack = (score, i)

    if best_attack is None:
        return None
    return int(best_attack[1])


def heuristic_agent_cpu(obs, max_moves: int = MAX_MODEL_MOVES):
    """Rule-based teacher acting directly in CPU token space."""
    view = GameView_CPU(obs)
    moves = []
    for _ in range(max_moves):
        token_idx = choose_heuristic_token_cpu(view)
        if token_idx is None:
            break
        action = view.apply_planned_move(token_idx)
        if action is None:
            break
        moves.append(action)
    return moves


def _bundle_to_tensors(bundle, device: torch.device):
    return (
        torch.from_numpy(bundle.edges).unsqueeze(0).to(device),
        torch.from_numpy(bundle.src_ids).long().unsqueeze(0).to(device),
        torch.from_numpy(bundle.tgt_ids).long().unsqueeze(0).to(device),
    )


def model_agent_actions_cpu(model, obs, max_moves: int = MAX_MODEL_MOVES,
                            deterministic: bool = True,
                            view: GameView_CPU | None = None,
                            device: torch.device | str = "cpu"):
    """Decode a CPU edge model into up to ``max_moves`` env actions."""
    device = torch.device(device)
    if view is None:
        view = GameView_CPU(obs)

    moves = []
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_moves):
            bundle = view.tokens()
            if bundle.n == 0:
                break
            with torch.no_grad():
                logits, _ = model(
                    *_bundle_to_tensors(bundle, device),
                    compute_value=False,
                )
            logits = logits[0]
            if deterministic:
                action_idx = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                action_idx = int(torch.multinomial(probs, 1).item())
            if action_idx == bundle.n:
                break
            action = view.apply_planned_move(action_idx)
            if action is None:
                break
            moves.append(action)
    finally:
        model.train(was_training)
    return moves


class StatefulCpuModelAgent:
    """Persistent ``GameView_CPU`` wrapper for Kaggle/server agent calls."""

    def __init__(self, model, max_moves: int = MAX_MODEL_MOVES,
                 deterministic: bool = True, device: torch.device | str = "cpu"):
        self.model = model
        self.max_moves = int(max_moves)
        self.deterministic = bool(deterministic)
        self.device = torch.device(device)
        self._view: GameView_CPU | None = None

    def __call__(self, obs):
        step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
        if self._view is None or int(step) <= int(self._view.step):
            self._view = GameView_CPU(obs)
        else:
            self._view.update_from_obs(obs)
        return model_agent_actions_cpu(
            self.model,
            obs,
            max_moves=self.max_moves,
            deterministic=self.deterministic,
            view=self._view,
            device=self.device,
        )

    def reset(self):
        self._view = None


def load_cpu_model_agent(checkpoint_path: str | Path,
                         device: torch.device | str = "cpu",
                         deterministic: bool = True) -> StatefulCpuModelAgent:
    """Load a BC/RL CPU-model checkpoint as a callable agent."""
    device = torch.device(device)
    model = OrbitWarsEdgeTransformer().to(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return StatefulCpuModelAgent(
        model, deterministic=deterministic, device=device,
    )
