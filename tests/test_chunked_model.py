"""Tests for the experimental chunked CPU edge policy."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaggle_environments import make  # noqa: E402

from harness_cpu import FEATURE_DIM, GameView_CPU  # noqa: E402
from orbit_wars.cpu.chunked_model import (  # noqa: E402
    ChunkedEdgePolicy,
    decode_chunk_actions,
)


def _real_bundle():
    env = make("orbit_wars", debug=False)
    env.reset()
    env.step([[], []])
    view = GameView_CPU(env.state[0].observation)
    bundle = view.tokens()
    assert bundle.n > 0
    return view, bundle


def test_chunked_policy_forward_shapes_and_masks_padded_candidates():
    torch.manual_seed(0)
    model = ChunkedEdgePolicy(feature_dim=FEATURE_DIM, d_model=32, d_ff=64, n_slots=4)
    edges = torch.randn(2, 7, FEATURE_DIM)
    src_ids = torch.tensor([
        [0, 0, 1, 1, 2, 2, 0],
        [0, 1, 1, 2, 0, 0, 0],
    ])
    tgt_ids = torch.tensor([
        [1, 2, 0, 2, 0, 1, 0],
        [1, 0, 2, 0, 0, 0, 0],
    ])
    valid_mask = torch.tensor([
        [True, True, True, True, True, True, True],
        [True, True, True, True, False, False, False],
    ])

    out = model(edges, src_ids, tgt_ids, valid_mask=valid_mask)

    assert out.pointer_logits.shape == (2, 4, 8)
    assert out.ship_delta_mu.shape == (2, 4)
    assert out.ship_delta_log_std.shape == (2, 4)
    assert out.active_logits.shape == (2, 4)
    assert out.value.shape == (2,)
    assert torch.isfinite(out.pointer_logits[:, :, -1]).all()
    assert (out.pointer_logits[1, :, 4:7] < -1.0e8).all()


def test_decode_chunk_actions_recomputes_legal_move_from_candidate_and_multiplier():
    view, bundle = _real_bundle()
    token_idx = 0
    base_move = view.decode_action(token_idx)
    assert base_move is not None

    decoded = decode_chunk_actions(
        view,
        token_indices=[token_idx],
        ship_multipliers=[1.0],
        active=[True],
    )

    assert len(decoded) == 1
    assert decoded[0][0] == base_move[0]
    assert decoded[0][2] == base_move[2]


def test_decode_chunk_actions_enforces_source_budget_across_slots():
    view, bundle = _real_bundle()
    first_src = int(bundle.src_ids[0])
    same_source = [
        i for i, src in enumerate(bundle.src_ids.tolist())
        if int(src) == first_src
    ]
    assert len(same_source) >= 1
    repeated = same_source[:1] * 3

    decoded = decode_chunk_actions(
        view,
        token_indices=repeated,
        ship_multipliers=[10.0] * len(repeated),
        active=[True] * len(repeated),
    )

    src_pid = int(bundle.planet_ids[first_src])
    src_initial = int(view.planets_by_id[src_pid][5])
    spent = sum(move[2] for move in decoded if move[0] == src_pid)
    assert spent <= src_initial


if __name__ == "__main__":
    print("test_chunked_policy_forward_shapes_and_masks_padded_candidates")
    test_chunked_policy_forward_shapes_and_masks_padded_candidates()
    print("test_decode_chunk_actions_recomputes_legal_move_from_candidate_and_multiplier")
    test_decode_chunk_actions_recomputes_legal_move_from_candidate_and_multiplier()
    print("test_decode_chunk_actions_enforces_source_budget_across_slots")
    test_decode_chunk_actions_enforces_source_budget_across_slots()
    print("\nALL CHUNKED MODEL TESTS PASSED")
