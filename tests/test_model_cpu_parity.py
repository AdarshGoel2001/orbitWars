"""Padded-batch vs unpadded-single forward parity for model_cpu.

Goal: for every valid (non-pad) position in a padded batch, the model's
output must match what that sample would produce if run alone with no
padding. This is a mask-correctness test — it catches bugs where pad
tokens leak into attention, where the stop/value pool attends to pad, or
where the axial masks are built wrong.

Two regimes tested:
  1. Full-attention mode — use two real game observations (N < threshold)
  2. Axial-attention mode — lower the threshold and use synthetic tokens
     that exceed it, forcing axial masks in the padded forward

Both tests use a fresh model with deterministic init and run forward with
grad disabled; any divergence above 1e-5 fails the test.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaggle_environments import make  # noqa: E402

from harness_cpu import GameView_CPU, FEATURE_DIM  # noqa: E402
from model_cpu import OrbitWarsEdgeTransformer  # noqa: E402


TOL = 1e-5


def _make_later_obs(turns: int):
    """Play `turns` turns of no-op so both players start producing ships."""
    env = make("orbit_wars", debug=False)
    env.reset()
    env.step([[], []])
    for _ in range(turns):
        env.step([[], []])
    return env.state[0].observation, env.state[1].observation


def _to_tensors(bundle):
    return (
        torch.from_numpy(bundle.edges).unsqueeze(0),
        torch.from_numpy(bundle.src_ids).long().unsqueeze(0),
        torch.from_numpy(bundle.tgt_ids).long().unsqueeze(0),
    )


def _pad_batch(bundles, n_max):
    B = len(bundles)
    edges = torch.zeros(B, n_max, FEATURE_DIM, dtype=torch.float32)
    src = torch.zeros(B, n_max, dtype=torch.long)
    tgt = torch.zeros(B, n_max, dtype=torch.long)
    valid = torch.zeros(B, n_max, dtype=torch.bool)
    for i, b in enumerate(bundles):
        n = b.n
        edges[i, :n] = torch.from_numpy(b.edges)
        src[i, :n] = torch.from_numpy(b.src_ids).long()
        tgt[i, :n] = torch.from_numpy(b.tgt_ids).long()
        valid[i, :n] = True
    return edges, src, tgt, valid


def _compare_sample(sample_idx, bundle, batch_logits, batch_value,
                    solo_logits, solo_value, n_max):
    """Return (edge_diff, stop_diff, value_diff) for one sample."""
    n = bundle.n
    # Edge logits: batch[:n] vs solo[:n]
    edge_diff = (batch_logits[sample_idx, :n] - solo_logits[0, :n]).abs().max().item()
    # Stop logit: at index n_max in batch, at index n in solo
    stop_diff = (batch_logits[sample_idx, n_max] - solo_logits[0, n]).abs().item()
    value_diff = (batch_value[sample_idx] - solo_value[0]).abs().item()
    return edge_diff, stop_diff, value_diff


def test_full_attention_parity():
    """Two real-game samples with N below threshold → full attention."""
    torch.manual_seed(0)
    model = OrbitWarsEdgeTransformer(axial_threshold=256)
    model.eval()

    obs_a, _ = _make_later_obs(turns=0)
    obs_b, _ = _make_later_obs(turns=30)
    view_a = GameView_CPU(obs_a)
    view_b = GameView_CPU(obs_b)
    bundle_a = view_a.tokens()
    bundle_b = view_b.tokens()
    assert bundle_a.n > 0 and bundle_b.n > 0, "both views must have tokens"
    assert bundle_a.n <= 256 and bundle_b.n <= 256, \
        "full-attn test needs N below threshold"

    with torch.no_grad():
        logits_a, value_a = model(*_to_tensors(bundle_a))
        logits_b, value_b = model(*_to_tensors(bundle_b))

    n_max = max(bundle_a.n, bundle_b.n)
    edges, src, tgt, valid = _pad_batch([bundle_a, bundle_b], n_max)
    with torch.no_grad():
        logits_batch, value_batch = model(edges, src, tgt, valid_mask=valid)

    assert logits_batch.shape == (2, n_max + 1)

    ea, sa, va = _compare_sample(0, bundle_a, logits_batch, value_batch,
                                 logits_a, value_a, n_max)
    eb, sb, vb = _compare_sample(1, bundle_b, logits_batch, value_batch,
                                 logits_b, value_b, n_max)
    print(f"  [full attn]  N_a={bundle_a.n:>3}  edge={ea:.2e}  stop={sa:.2e}  val={va:.2e}")
    print(f"  [full attn]  N_b={bundle_b.n:>3}  edge={eb:.2e}  stop={sb:.2e}  val={vb:.2e}")
    assert max(ea, sa, va, eb, sb, vb) < TOL


def test_axial_attention_parity():
    """Synthetic tokens with N above a low threshold → axial masks fire."""
    torch.manual_seed(1)
    model = OrbitWarsEdgeTransformer(axial_threshold=10)
    model.eval()

    # Sample A: N=5 (full, below threshold 10)
    # Sample B: N=25 (axial, above threshold 10)
    class _FakeBundle:
        def __init__(self, edges, src_ids, tgt_ids):
            self.edges = edges
            self.src_ids = src_ids
            self.tgt_ids = tgt_ids
            self.n = edges.shape[0]

    rng = np.random.default_rng(42)
    edges_a = rng.standard_normal((5, FEATURE_DIM)).astype(np.float32) * 2
    src_a = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    tgt_a = np.array([1, 2, 0, 2, 0], dtype=np.int32)
    bundle_a = _FakeBundle(edges_a, src_a, tgt_a)

    edges_b = rng.standard_normal((25, FEATURE_DIM)).astype(np.float32) * 2
    # Make src/tgt have real group structure so axial masks differ from full.
    src_b = rng.integers(0, 5, size=25).astype(np.int32)
    tgt_b = rng.integers(0, 8, size=25).astype(np.int32)
    bundle_b = _FakeBundle(edges_b, src_b, tgt_b)

    with torch.no_grad():
        logits_a, value_a = model(*_to_tensors(bundle_a))
        logits_b, value_b = model(*_to_tensors(bundle_b))

    n_max = max(bundle_a.n, bundle_b.n)
    edges, src, tgt, valid = _pad_batch([bundle_a, bundle_b], n_max)
    with torch.no_grad():
        logits_batch, value_batch = model(edges, src, tgt, valid_mask=valid)

    ea, sa, va = _compare_sample(0, bundle_a, logits_batch, value_batch,
                                 logits_a, value_a, n_max)
    eb, sb, vb = _compare_sample(1, bundle_b, logits_batch, value_batch,
                                 logits_b, value_b, n_max)
    print(f"  [mixed]      N_a={bundle_a.n:>3}  edge={ea:.2e}  stop={sa:.2e}  val={va:.2e}  (full)")
    print(f"  [mixed]      N_b={bundle_b.n:>3}  edge={eb:.2e}  stop={sb:.2e}  val={vb:.2e}  (axial)")
    assert max(ea, sa, va, eb, sb, vb) < TOL


def test_inference_matches_training_path():
    """compute_value=False should give identical logits to compute_value=True."""
    torch.manual_seed(2)
    model = OrbitWarsEdgeTransformer()
    model.eval()

    obs, _ = _make_later_obs(turns=20)
    view = GameView_CPU(obs)
    bundle = view.tokens()
    assert bundle.n > 0

    with torch.no_grad():
        logits_train, _ = model(*_to_tensors(bundle), compute_value=True)
        logits_infer, value_infer = model(*_to_tensors(bundle), compute_value=False)

    diff = (logits_train - logits_infer).abs().max().item()
    print(f"  [val gate]   N={bundle.n:>3}  logit diff={diff:.2e}  value_is_None={value_infer is None}")
    assert diff < TOL
    assert value_infer is None


if __name__ == "__main__":
    print("test_full_attention_parity")
    test_full_attention_parity()
    print("test_axial_attention_parity")
    test_axial_attention_parity()
    print("test_inference_matches_training_path")
    test_inference_matches_training_path()
    print("\nALL PARITY TESTS PASSED")
