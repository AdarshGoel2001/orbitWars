"""Tests for chunked behavior-cloning targets and loss."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaggle_environments import make  # noqa: E402

from orbit_wars.cpu.chunked_bc import (  # noqa: E402
    build_teacher_chunk,
    chunked_bc_loss,
    collate_chunk_examples,
)
from orbit_wars.cpu.chunked_model import ChunkedEdgePolicy  # noqa: E402


def _obs():
    env = make("orbit_wars", debug=False)
    env.reset()
    env.step([[], []])
    return env.state[0].observation


def test_build_teacher_chunk_uses_initial_candidate_indices_and_empty_slots():
    example = build_teacher_chunk(_obs(), max_slots=6)

    assert example.n_tokens > 0
    assert example.pointer_idx.shape == (6,)
    assert example.active.shape == (6,)
    assert example.ship_delta.shape == (6,)
    assert (example.pointer_idx >= 0).all()
    assert (example.pointer_idx <= example.n_tokens).all()
    assert (example.pointer_idx[~example.active] == example.n_tokens).all()
    assert torch.isfinite(torch.from_numpy(example.ship_delta)).all()


def test_chunked_bc_loss_runs_backward_on_teacher_chunk():
    torch.manual_seed(0)
    examples = [
        build_teacher_chunk(_obs(), max_slots=4),
        build_teacher_chunk(_obs(), max_slots=4),
    ]
    batch = collate_chunk_examples(examples)
    model = ChunkedEdgePolicy(d_model=32, d_ff=64, n_slots=4)

    loss, metrics = chunked_bc_loss(model, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["examples"] == 2
    assert metrics["slots"] == 8
    assert metrics["active_slots"] >= 0
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.parameters()
    )


if __name__ == "__main__":
    print("test_build_teacher_chunk_uses_initial_candidate_indices_and_empty_slots")
    test_build_teacher_chunk_uses_initial_candidate_indices_and_empty_slots()
    print("test_chunked_bc_loss_runs_backward_on_teacher_chunk")
    test_chunked_bc_loss_runs_backward_on_teacher_chunk()
    print("\nALL CHUNKED BC TESTS PASSED")
