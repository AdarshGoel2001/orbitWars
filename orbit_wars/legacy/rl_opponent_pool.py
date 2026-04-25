"""Opponent pool for PPO self-play.

Samples opponents between fixed heuristic and model snapshots (past training
checkpoints). Starts with heuristic-only; add_snapshot() populates the pool
for more interesting training dynamics.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

import torch

from agents import heuristic_agent, StatefulModelAgent
from model import OrbitWarsTransformer


class OpponentPool:
    """Manages opponent sampling: heuristic + past model snapshots."""

    def __init__(self, heuristic_weight: float = 0.5, max_snapshots: int = 8):
        """
        Args:
            heuristic_weight: Relative weight of heuristic in pool.
            max_snapshots: Max number of past checkpoints to keep (FIFO eviction).
        """
        self.heuristic_weight = heuristic_weight
        self.snapshot_weight = 1.0  # Keep equal to heuristic initially.
        self.max_snapshots = max_snapshots
        self._snapshots: list[tuple[dict, str]] = []

    def add_snapshot(self, model: OrbitWarsTransformer, name: str):
        """Store a copy of the current model as an opponent snapshot.

        Args:
            model: Model to snapshot.
            name: Label for logging (e.g., "step_5000", "epoch_3").
        """
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self._snapshots.append((state, name))
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)

    def sample(self, device: str = "cpu") -> tuple[Callable, str]:
        """Sample an opponent (heuristic or snapshot).

        Returns:
            (opponent_fn: Callable(obs) -> actions, opponent_name: str)
        """
        # If no snapshots, always use heuristic.
        if not self._snapshots:
            return heuristic_agent, "heuristic"

        # Weighted sample between heuristic and snapshots.
        use_snapshot = random.random() < (
            self.snapshot_weight / (self.heuristic_weight + self.snapshot_weight)
        )

        if use_snapshot:
            state, name = random.choice(self._snapshots)
            opp_model = OrbitWarsTransformer().to(device)
            opp_model.load_state_dict(state)
            opp_model.eval()
            return StatefulModelAgent(opp_model, deterministic=False), f"snapshot_{name}"

        return heuristic_agent, "heuristic"
