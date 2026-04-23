"""Benchmark numpy CPU inference for model_cpu.

Ports the torch forward pass to pure numpy, runs it across a real
heuristic-vs-heuristic game, and reports per-call latency. This is the
Kaggle-feasibility check: if numpy forward p95 × ~10 (Kaggle slowdown)
fits in 1 s per agent call, we're shippable.

This is a *benchmark*, not a submission artifact. Step 8 will
re-implement the numpy forward in a submission-ready form.
"""

from __future__ import annotations

import math
import time
from typing import Dict

import numpy as np
import torch

from kaggle_environments import make
from agents import heuristic_agent
from harness_cpu import GameView_CPU
from model_cpu import OrbitWarsEdgeTransformer


# ----------------------------------------------------------------------
# numpy primitives
# ----------------------------------------------------------------------


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    # tanh approximation — matches torch's approximate='tanh' to ~1e-4
    return 0.5 * x * (1.0 + np.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))


def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    # w is (out, in) per torch convention
    return x @ w.T + b


def _masked_attention(x, W_qkv, b_qkv, W_o, b_o, mask, scale):
    """Single-head self-attention. x: (N, d); mask: (N, N) bool."""
    qkv = _linear(x, W_qkv, b_qkv)        # (N, 3d)
    d = x.shape[-1]
    q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
    scores = q @ k.T * scale              # (N, N)
    scores = np.where(mask, scores, -1e9)
    attn = _softmax(scores, axis=-1)
    row_has_any = mask.any(axis=-1, keepdims=True)
    attn = np.where(row_has_any, attn, 0.0)
    out = attn @ v                         # (N, d)
    return _linear(out, W_o, b_o)


def _encoder_block(x, sd, prefix, mask, scale):
    x_norm = _layer_norm(x, sd[f"{prefix}.ln1.weight"], sd[f"{prefix}.ln1.bias"])
    x = x + _masked_attention(
        x_norm,
        sd[f"{prefix}.attn.qkv.weight"], sd[f"{prefix}.attn.qkv.bias"],
        sd[f"{prefix}.attn.out.weight"], sd[f"{prefix}.attn.out.bias"],
        mask, scale,
    )
    x_norm = _layer_norm(x, sd[f"{prefix}.ln2.weight"], sd[f"{prefix}.ln2.bias"])
    ff = _gelu_tanh(_linear(
        x_norm, sd[f"{prefix}.ffn.fc1.weight"], sd[f"{prefix}.ffn.fc1.bias"],
    ))
    ff = _linear(ff, sd[f"{prefix}.ffn.fc2.weight"], sd[f"{prefix}.ffn.fc2.bias"])
    return x + ff


def _attention_pool(x, query, scale, valid_mask):
    """x: (N, d); query: (d,); valid_mask: (N,) bool → (d,)."""
    N, d = x.shape
    if N == 0:
        return np.zeros(d, dtype=x.dtype)
    scores = (query[None, :] @ x.T) * scale        # (1, N)
    scores = np.where(valid_mask[None, :], scores, -1e9)
    attn = _softmax(scores, axis=-1)
    if not valid_mask.any():
        attn = np.zeros_like(attn)
    return (attn @ x).squeeze(0)                    # (d,)


# ----------------------------------------------------------------------
# full forward
# ----------------------------------------------------------------------


def extract_numpy_weights(model: OrbitWarsEdgeTransformer) -> Dict[str, np.ndarray]:
    sd = {}
    for name, tensor in model.state_dict().items():
        sd[name] = tensor.detach().cpu().numpy().astype(np.float32)
    return sd


def forward_numpy(edges: np.ndarray, src_ids: np.ndarray, tgt_ids: np.ndarray,
                  sd: Dict[str, np.ndarray], d_model: int,
                  axial_threshold: int) -> np.ndarray:
    """Single-sample numpy forward. Returns logits of shape (N+1,).

    Skips the value head — inference doesn't need it.
    """
    N = edges.shape[0]
    if N == 0:
        # Only stop is available. Logit is stop_head(stop_pool(no_tokens)).
        # With zero pool, stop_logit = bias of stop_head.
        stop_bias = sd["stop_head.bias"][0]
        return np.array([stop_bias], dtype=np.float32)

    valid_mask = np.ones(N, dtype=bool)
    scale = 1.0 / math.sqrt(d_model)

    x = edges.astype(np.float32) / sd["feature_scales"]
    x = _linear(x, sd["input_proj.weight"], sd["input_proj.bias"])
    x = _layer_norm(x, sd["input_ln.weight"], sd["input_ln.bias"])

    if N > axial_threshold:
        mask1 = (src_ids[:, None] == src_ids[None, :])
        mask2 = (tgt_ids[:, None] == tgt_ids[None, :])
    else:
        mask1 = np.ones((N, N), dtype=bool)
        mask2 = mask1

    x = _encoder_block(x, sd, "block1", mask1, scale)
    x = _encoder_block(x, sd, "block2", mask2, scale)

    edge_logits = _linear(x, sd["edge_head.weight"], sd["edge_head.bias"]).squeeze(-1)

    stop_pooled = _attention_pool(x, sd["stop_pool.query"], scale, valid_mask)
    stop_logit = _linear(
        stop_pooled[None, :], sd["stop_head.weight"], sd["stop_head.bias"],
    )[0, 0]

    return np.concatenate([edge_logits, [stop_logit]]).astype(np.float32)


# ----------------------------------------------------------------------
# benchmark driver
# ----------------------------------------------------------------------


def _play_game_and_time(sd, d_model, axial_threshold, max_turns=200):
    env = make("orbit_wars", debug=False)
    env.reset()
    env.step([[], []])
    view = GameView_CPU(env.state[0].observation)

    times_ms = []
    n_tokens = []
    for turn in range(max_turns):
        obs = env.state[0].observation
        if turn > 0:
            view.update_from_obs(obs)
        b = view.tokens()
        if b.n > 0:
            t0 = time.perf_counter()
            _ = forward_numpy(b.edges, b.src_ids, b.tgt_ids,
                              sd, d_model, axial_threshold)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
            n_tokens.append(b.n)

        a0 = heuristic_agent(env.state[0].observation)
        a1 = heuristic_agent(env.state[1].observation)
        env.step([a0, a1])
        if env.state[0].status != "ACTIVE":
            break

    return np.array(times_ms), np.array(n_tokens)


def _verify_numpy_matches_torch(model, sd):
    """Sanity-check the numpy port matches torch on a sample obs."""
    env = make("orbit_wars", debug=False)
    env.reset()
    env.step([[], []])
    for _ in range(10):
        env.step([[], []])
    obs = env.state[0].observation
    view = GameView_CPU(obs)
    b = view.tokens()

    edges = torch.from_numpy(b.edges).unsqueeze(0)
    src = torch.from_numpy(b.src_ids).long().unsqueeze(0)
    tgt = torch.from_numpy(b.tgt_ids).long().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        torch_logits, _ = model(edges, src, tgt, compute_value=False)
    torch_logits = torch_logits[0].cpu().numpy()

    numpy_logits = forward_numpy(
        b.edges, b.src_ids, b.tgt_ids, sd,
        model.d_model, model.axial_threshold,
    )

    diff = np.abs(torch_logits - numpy_logits).max()
    return diff, b.n


def main():
    torch.manual_seed(0)
    model = OrbitWarsEdgeTransformer()
    sd = extract_numpy_weights(model)

    diff, n = _verify_numpy_matches_torch(model, sd)
    print(f"numpy vs torch logits max-diff (N={n}): {diff:.2e}")
    print("  (tanh GELU approx causes small divergence; acceptable for bench)")
    print()

    # Warmup
    env = make("orbit_wars", debug=False)
    env.reset()
    env.step([[], []])
    warmup_view = GameView_CPU(env.state[0].observation)
    wb = warmup_view.tokens()
    for _ in range(10):
        _ = forward_numpy(wb.edges, wb.src_ids, wb.tgt_ids,
                          sd, model.d_model, model.axial_threshold)

    # Benchmark
    times, ns = _play_game_and_time(
        sd, model.d_model, model.axial_threshold, max_turns=200,
    )

    print(f"samples: {len(times)}")
    print(f"N tokens: mean={ns.mean():5.1f}  p50={np.percentile(ns,50):5.0f}  "
          f"p95={np.percentile(ns,95):5.0f}  max={ns.max()}")
    print()
    print("numpy forward_numpy() latency (ms):")
    for label, pct in [("mean", None), ("p50", 50), ("p95", 95), ("p99", 99), ("max", 100)]:
        if pct is None:
            v = times.mean()
        elif pct == 100:
            v = times.max()
        else:
            v = np.percentile(times, pct)
        print(f"  {label:5s}  {v:7.2f}")

    p95 = np.percentile(times, 95)
    mx = times.max()
    print()
    print("Projected Kaggle (×10 slowdown):")
    print(f"  p95 ≈ {p95 * 10:7.0f} ms  (budget per forward ≈ 333 ms if MAX_MODEL_MOVES=3)")
    print(f"  max ≈ {mx * 10:7.0f} ms")
    print(f"Status: {'PASS' if p95 * 10 < 333 else 'TIGHT' if p95 * 10 < 700 else 'FAIL'}")


if __name__ == "__main__":
    main()
