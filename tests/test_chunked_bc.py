"""Tests for chunked behavior-cloning targets and loss."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaggle_environments import make  # noqa: E402

from orbit_wars.cpu.chunked_bc import (  # noqa: E402
    build_teacher_chunk,
    chunked_bc_loss,
    collate_chunk_examples,
    load_chunk_examples_from_cpu_shards,
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


def test_load_chunk_examples_from_cpu_shards_groups_turn_moves(tmp_path):
    path = tmp_path / "cpu_shard.npz"
    edges0 = np.array([[1.0] * 11, [2.0] * 11, [3.0] * 11], dtype=np.float32)
    edges0[0, 1] = 8.0
    edges0[0, 5] = 30.0
    edges0[1, 1] = 8.0
    edges0[1, 5] = 9.0
    edges1 = np.array([[4.0] * 11, [5.0] * 11], dtype=np.float32)
    edges2 = np.array([[6.0] * 11], dtype=np.float32)
    packed = np.concatenate([edges0, edges1, edges2], axis=0)
    np.savez_compressed(
        path,
        edges_packed=packed,
        src_ids_packed=np.array([0, 0, 1, 0, 1, 0], dtype=np.int32),
        tgt_ids_packed=np.array([1, 2, 2, 2, 1, 1], dtype=np.int32),
        offsets=np.array([0, 3, 5, 6], dtype=np.int64),
        n_tokens=np.array([3, 2, 1], dtype=np.int32),
        action_idx=np.array([1, 0, 0], dtype=np.int64),
        game=np.array([7, 7, 7], dtype=np.int32),
        step=np.array([42, 42, 43], dtype=np.int32),
        player=np.array([1, 1, 1], dtype=np.int32),
        submove=np.array([0, 1, 0], dtype=np.int32),
        src_slot=np.array([0, 0, 0], dtype=np.int32),
        tgt_slot=np.array([2, 1, 1], dtype=np.int32),
        src_pid=np.array([10, 10, 20], dtype=np.int32),
        tgt_pid=np.array([12, 11, 21], dtype=np.int32),
        ships=np.array([9, 18, 5], dtype=np.int32),
    )

    examples = load_chunk_examples_from_cpu_shards([path], max_slots=4)

    assert len(examples) == 2
    first = examples[0]
    assert first.n_tokens == 3
    assert first.active.tolist() == [True, True, False, False]
    assert first.pointer_idx.tolist() == [1, 0, 3, 3]
    assert np.isclose(first.ship_delta[0], 0.0)
    assert first.ship_delta[1] > 0.0


if __name__ == "__main__":
    print("test_build_teacher_chunk_uses_initial_candidate_indices_and_empty_slots")
    test_build_teacher_chunk_uses_initial_candidate_indices_and_empty_slots()
    print("test_chunked_bc_loss_runs_backward_on_teacher_chunk")
    test_chunked_bc_loss_runs_backward_on_teacher_chunk()
    print("test_load_chunk_examples_from_cpu_shards_groups_turn_moves")
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        test_load_chunk_examples_from_cpu_shards_groups_turn_moves(Path(tmp))
    print("\nALL CHUNKED BC TESTS PASSED")
