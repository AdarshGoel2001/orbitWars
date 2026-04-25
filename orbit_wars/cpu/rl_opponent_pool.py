"""Opponent pool for CPU-stack PPO self-play."""

from __future__ import annotations

import random
from typing import Callable

import torch

from orbit_wars.cpu.agents import StatefulCpuModelAgent, heuristic_agent_cpu
from orbit_wars.cpu.model import OrbitWarsEdgeTransformer


class OpponentPool:
    """Samples between the CPU heuristic and past learner snapshots."""

    def __init__(self, heuristic_weight: float = 0.5, max_snapshots: int = 8):
        self.heuristic_weight = float(heuristic_weight)
        self.snapshot_weight = 1.0
        self.max_snapshots = int(max_snapshots)
        self._snapshots: list[tuple[dict, str]] = []

    def add_snapshot(self, model: OrbitWarsEdgeTransformer, name: str):
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self._snapshots.append((state, str(name)))
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)

    def state_dict(self) -> dict:
        return {
            "heuristic_weight": self.heuristic_weight,
            "snapshot_weight": self.snapshot_weight,
            "max_snapshots": self.max_snapshots,
            "snapshots": self._snapshots,
        }

    def load_state_dict(self, state: dict):
        self.heuristic_weight = float(state["heuristic_weight"])
        self.snapshot_weight = float(state["snapshot_weight"])
        self.max_snapshots = int(state["max_snapshots"])
        self._snapshots = [(snapshot, str(name)) for snapshot, name in state["snapshots"]]

    def sample(self, device: torch.device | str = "cpu") -> tuple[Callable, str]:
        if not self._snapshots:
            return heuristic_agent_cpu, "heuristic"

        total_weight = self.heuristic_weight + self.snapshot_weight
        use_snapshot = random.random() < (self.snapshot_weight / total_weight)
        if not use_snapshot:
            return heuristic_agent_cpu, "heuristic"

        if not isinstance(device, torch.device):
            device = torch.device(device)

        state, name = random.choice(self._snapshots)
        model = OrbitWarsEdgeTransformer().to(device)
        model.load_state_dict(state)
        model.eval()
        return (
            StatefulCpuModelAgent(model, deterministic=False, device=device),
            f"snapshot_{name}",
        )
