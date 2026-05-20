"""Tests for chunked PPO rollout/update utilities."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbit_wars.cpu.chunked_model import ChunkedEdgePolicy  # noqa: E402
from orbit_wars.cpu.rl_chunked_ppo import pad_chunk_records, ppo_update_step_chunked  # noqa: E402
from orbit_wars.cpu.rl_chunked_rollout import ChunkedGameTrajectory, ChunkedTurnRecord  # noqa: E402


def _record(n_tokens: int, n_slots: int, reward: float = 0.1) -> ChunkedTurnRecord:
    edges = np.random.default_rng(0).normal(size=(n_tokens, 11)).astype(np.float32)
    src_ids = np.arange(n_tokens, dtype=np.int64) % 3
    tgt_ids = (np.arange(n_tokens, dtype=np.int64) + 1) % 3
    pointer_idx = np.full(n_slots, n_tokens, dtype=np.int64)
    active = np.zeros(n_slots, dtype=np.bool_)
    if n_tokens:
        pointer_idx[0] = 0
        active[0] = True
    return ChunkedTurnRecord(
        edges=edges,
        src_ids=src_ids,
        tgt_ids=tgt_ids,
        n_tokens=n_tokens,
        pointer_idx=pointer_idx,
        active=active,
        ship_delta=np.zeros(n_slots, dtype=np.float32),
        logprob=-1.0,
        value=0.0,
        reward=reward,
        done=False,
        sampled_active=int(active.sum()),
        decoded_moves=int(active.sum()),
        dropped_slots=0,
    )


def test_pad_chunk_records_remaps_empty_to_batch_stop():
    records = [_record(3, 4), _record(5, 4)]
    batch = pad_chunk_records(records, device=torch.device("cpu"))

    assert batch["edges"].shape == (2, 5, 11)
    assert batch["pointer_idx"].shape == (2, 4)
    assert batch["pointer_idx"][0, 1:].tolist() == [5, 5, 5]
    assert batch["pointer_idx"][1, 1:].tolist() == [5, 5, 5]
    assert batch["valid_mask"][0, :3].all()
    assert not batch["valid_mask"][0, 3:].any()


def test_chunked_ppo_update_changes_parameters_and_returns_finite_metrics():
    torch.manual_seed(0)
    model = ChunkedEdgePolicy(d_model=32, d_ff=64, n_slots=4, encoder_layers=1, decoder_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    records = [_record(6, 4, reward=0.1), _record(4, 4, reward=-0.05)]
    records[-1].done = True
    traj = ChunkedGameTrajectory(
        records=records,
        learner_seat=0,
        final_margin=0.1,
        turns=2,
        opponent_name="test",
    )
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    metrics = ppo_update_step_chunked(
        model,
        [traj],
        optimizer,
        ppo_epochs=1,
        ppo_batch_size=2,
        target_kl=None,
    )

    assert metrics["updates"] == 1.0
    assert all(np.isfinite(float(v)) for v in metrics.values())
    changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert changed


if __name__ == "__main__":
    print("test_pad_chunk_records_remaps_empty_to_batch_stop")
    test_pad_chunk_records_remaps_empty_to_batch_stop()
    print("test_chunked_ppo_update_changes_parameters_and_returns_finite_metrics")
    test_chunked_ppo_update_changes_parameters_and_returns_finite_metrics()
    print("\nALL CHUNKED RL TESTS PASSED")
