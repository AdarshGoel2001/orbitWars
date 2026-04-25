"""Build a self-contained NumPy submission for the CPU dynamic-edge model."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parent


def _strip_local_imports(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("from __future__ import "):
            continue
        if stripped in {
            "from radar import Radar",
            "from radar import Radar, RadarHit",
            "import targeting as T",
        }:
            continue
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def _npz_from_checkpoint(checkpoint: Path) -> tuple[str, str, int, int]:
    import numpy as np
    import torch

    from orbit_wars.cpu.model import OrbitWarsEdgeTransformer

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    arrays = {
        key.replace(".", "__"): value.detach().cpu().numpy().astype(np.float32)
        for key, value in state.items()
        if not key.startswith("value_head.")
    }
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    raw = buf.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    axial_threshold = int(OrbitWarsEdgeTransformer().axial_threshold)
    return base64.b64encode(raw).decode("ascii"), digest, len(raw), axial_threshold


def _literal(encoded: str) -> str:
    lines = textwrap.wrap(encoded, width=88)
    body = "\n".join(f"    {line!r}" for line in lines)
    return "(\n" + body + "\n)"


def _append_sources(parts: list[str]):
    sources = [
        ("action_space.py", ROOT / "orbit_wars/core/action_space.py"),
        ("radar.py", ROOT / "orbit_wars/core/radar.py"),
        ("targeting.py", ROOT / "orbit_wars/core/targeting.py"),
        ("harness_cpu.py", ROOT / "orbit_wars/cpu/harness.py"),
    ]
    for label, path in sources:
        parts.append(f"\n# --- {label} ---\n")
        parts.append(_strip_local_imports(path.read_text()))
        if label == "targeting.py":
            parts.append(
                "\n"
                "class _TargetingNamespace:\n"
                "    pass\n\n"
                "T = _TargetingNamespace()\n"
                "T.fleet_speed = fleet_speed\n"
                "T.future_position = future_position\n"
            )


def _runtime(encoded: str, max_moves: int, axial_threshold: int) -> str:
    return f"""

# --- CPU edge-model NumPy runtime ---
import base64 as _base64
import io as _io
import os as _os

_WEIGHTS_B64 = {_literal(encoded)}
_NP_WEIGHTS = None
_STATEFUL_AGENT = None
_STATEFUL_PLAYER = None
_LAST_STEP = None
_LN_EPS = 1e-5


def _load_np_weights():
    global _NP_WEIGHTS
    if _NP_WEIGHTS is None:
        payload = _base64.b64decode(_WEIGHTS_B64.encode("ascii"))
        z = np.load(_io.BytesIO(payload))
        _NP_WEIGHTS = {{k.replace("__", "."): z[k].astype(np.float32) for k in z.files}}
    return _NP_WEIGHTS


def _linear(x, weight, bias):
    return x @ weight.T + bias


def _layer_norm(x, weight, bias):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + _LN_EPS)) * weight + bias


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _masked_attention(x, w, prefix, mask, scale):
    qkv = _linear(x, w[prefix + ".attn.qkv.weight"], w[prefix + ".attn.qkv.bias"])
    d = x.shape[-1]
    q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
    scores = (q @ k.T) * scale
    scores = np.where(mask, scores, -1.0e9)
    attn = _softmax(scores, axis=-1)
    row_has_any = mask.any(axis=-1, keepdims=True)
    attn = np.where(row_has_any, attn, 0.0)
    return _linear(attn @ v, w[prefix + ".attn.out.weight"], w[prefix + ".attn.out.bias"])


def _encoder_block(x, w, prefix, mask, scale):
    x_norm = _layer_norm(x, w[prefix + ".ln1.weight"], w[prefix + ".ln1.bias"])
    x = x + _masked_attention(x_norm, w, prefix, mask, scale)
    x_norm = _layer_norm(x, w[prefix + ".ln2.weight"], w[prefix + ".ln2.bias"])
    ff = _gelu(_linear(x_norm, w[prefix + ".ffn.fc1.weight"], w[prefix + ".ffn.fc1.bias"]))
    ff = _linear(ff, w[prefix + ".ffn.fc2.weight"], w[prefix + ".ffn.fc2.bias"])
    return x + ff


def _attention_pool(x, query, scale, valid_mask):
    n, d = x.shape
    if n == 0:
        return np.zeros(d, dtype=np.float32)
    scores = (query.reshape(1, d) @ x.T) * scale
    scores = np.where(valid_mask.reshape(1, n), scores, -1.0e9)
    attn = _softmax(scores, axis=-1)
    if not valid_mask.any():
        attn = np.zeros_like(attn)
    return (attn @ x).reshape(d)


def _forward_numpy(bundle):
    w = _load_np_weights()
    edges = bundle.edges
    n = edges.shape[0]
    if n == 0:
        return np.asarray([w["stop_head.bias"][0]], dtype=np.float32)

    d_model = int(w["feature_scales"].shape[0])
    # d_model is feature scale width here; actual hidden width comes from input proj.
    hidden = int(w["input_proj.weight"].shape[0])
    scale = 1.0 / math.sqrt(hidden)
    valid_mask = np.ones(n, dtype=bool)

    x = edges.astype(np.float32) / w["feature_scales"]
    x = _linear(x, w["input_proj.weight"], w["input_proj.bias"])
    x = _layer_norm(x, w["input_ln.weight"], w["input_ln.bias"])

    axial_threshold = {axial_threshold}
    if n > axial_threshold:
        mask1 = bundle.src_ids[:, None] == bundle.src_ids[None, :]
        mask2 = bundle.tgt_ids[:, None] == bundle.tgt_ids[None, :]
    else:
        mask1 = np.ones((n, n), dtype=bool)
        mask2 = mask1

    x = _encoder_block(x, w, "block1", mask1, scale)
    x = _encoder_block(x, w, "block2", mask2, scale)

    edge_logits = _linear(x, w["edge_head.weight"], w["edge_head.bias"]).reshape(-1)
    pooled = _attention_pool(x, w["stop_pool.query"], scale, valid_mask)
    stop_logit = float(_linear(pooled.reshape(1, -1), w["stop_head.weight"], w["stop_head.bias"])[0, 0])
    return np.concatenate([edge_logits, np.asarray([stop_logit], dtype=np.float32)])


class StatefulCpuNumpyAgent:
    def __init__(self, max_moves={max_moves}):
        self.max_moves = int(max_moves)
        self._view = None

    def __call__(self, obs):
        step = int(_get(obs, "step", 0) or 0)
        if self._view is None or step <= int(self._view.step):
            self._view = GameView_CPU(obs)
        else:
            self._view.update_from_obs(obs)

        moves = []
        for _ in range(self.max_moves):
            bundle = self._view.tokens()
            if bundle.n == 0:
                break
            logits = _forward_numpy(bundle)
            action_idx = int(np.argmax(logits))
            if action_idx == bundle.n:
                break
            action = self._view.apply_planned_move(action_idx)
            if action is None:
                break
            moves.append(action)
        return moves


def agent(obs, config=None):
    global _STATEFUL_AGENT, _STATEFUL_PLAYER, _LAST_STEP
    player = _get(obs, "player", 0)
    step = int(_get(obs, "step", 0) or 0)
    if (_STATEFUL_AGENT is None or _STATEFUL_PLAYER != player
            or (_LAST_STEP is not None and step <= _LAST_STEP)):
        _STATEFUL_AGENT = StatefulCpuNumpyAgent(max_moves={max_moves})
        _STATEFUL_PLAYER = player
    _LAST_STEP = step
    return _STATEFUL_AGENT(obs)
"""


def build(checkpoint: Path, max_moves: int) -> str:
    encoded, digest, size, axial_threshold = _npz_from_checkpoint(checkpoint)
    rel = checkpoint.relative_to(ROOT) if checkpoint.is_relative_to(ROOT) else checkpoint
    parts = [
        "# Generated by make_cpu_submission.py. Do not edit by hand.\n",
        f"# checkpoint: {rel}\n",
        f"# npz_sha256: {digest}\n",
        f"# embedded_npz_bytes: {size}\n",
        f"# max_moves: {max_moves}\n",
        f"# axial_threshold: {axial_threshold}\n",
        "from __future__ import annotations\n\n",
    ]
    _append_sources(parts)
    parts.append(_runtime(encoded, max_moves, axial_threshold))
    return "".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/bc_cpu_model.pt"))
    parser.add_argument("--out", type=Path, default=Path("submission_cpu.py"))
    parser.add_argument("--max-moves", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    out = args.out
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(build(checkpoint, args.max_moves))
    print(f"wrote {out}")
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
