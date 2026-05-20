"""Tests for chunked CPU model agent decoding."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaggle_environments import make  # noqa: E402

from orbit_wars.cpu.agents import model_agent_actions_chunked  # noqa: E402
from orbit_wars.cpu.chunked_model import ChunkedEdgePolicy  # noqa: E402
from orbit_wars.cpu.harness import GameView_CPU  # noqa: E402


def _obs():
    env = make("orbit_wars", debug=False)
    env.reset()
    env.step([[], []])
    return env.state[0].observation


def test_chunked_model_agent_returns_legal_move_list():
    torch.manual_seed(0)
    obs = _obs()
    model = ChunkedEdgePolicy(d_model=32, d_ff=64, n_slots=3, encoder_layers=1, decoder_layers=1)

    moves = model_agent_actions_chunked(
        model,
        obs,
        max_moves=3,
        active_threshold=-1.0e9,
        deterministic=True,
        device="cpu",
    )

    assert isinstance(moves, list)
    assert len(moves) <= 3
    owned = {int(p[0]): int(p[5]) for p in obs["planets"] if int(p[1]) == int(obs["player"])}
    spent = {}
    for src_pid, angle, ships in moves:
        assert src_pid in owned
        assert isinstance(angle, float)
        assert ships >= 1
        spent[src_pid] = spent.get(src_pid, 0) + int(ships)
    assert all(spent[src] <= owned[src] for src in spent)


def test_stateful_chunked_agent_reuses_view_update():
    torch.manual_seed(0)
    obs = _obs()
    model = ChunkedEdgePolicy(d_model=32, d_ff=64, n_slots=2, encoder_layers=1, decoder_layers=1)

    from orbit_wars.cpu.agents import StatefulChunkedModelAgent  # noqa: PLC0415

    agent = StatefulChunkedModelAgent(
        model,
        max_moves=2,
        active_threshold=-1.0e9,
        deterministic=True,
        device="cpu",
    )
    moves0 = agent(obs)
    first_view = agent._view
    assert isinstance(first_view, GameView_CPU)

    obs2 = dict(obs)
    obs2["step"] = int(obs["step"]) + 1
    moves1 = agent(obs2)
    assert agent._view is first_view
    assert len(moves0) <= 2
    assert len(moves1) <= 2


if __name__ == "__main__":
    print("test_chunked_model_agent_returns_legal_move_list")
    test_chunked_model_agent_returns_legal_move_list()
    print("test_stateful_chunked_agent_reuses_view_update")
    test_stateful_chunked_agent_reuses_view_update()
    print("\nALL CHUNKED AGENT TESTS PASSED")
