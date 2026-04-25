"""Build a self-contained neural-network Kaggle Orbit Wars submission.

This emits a single Python file containing the game harness, agent decoder,
model architecture, and embedded checkpoint weights. The generated artifact is
intended for Kaggle upload; source-of-truth code stays in the normal modules.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
from pathlib import Path
import tarfile
import textwrap
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "submission.py"
DEFAULT_ARCHIVE_OUT = ROOT / "submission_nn.tar.gz"
WEIGHTS_FILENAME = "weights.npz"
ARCHIVE_WEIGHTS_PATH = Path(WEIGHTS_FILENAME)

CORE_SOURCES = [
    ("action_space.py", ROOT / "orbit_wars/core/action_space.py"),
    ("radar.py", ROOT / "orbit_wars/core/radar.py"),
    ("targeting.py", ROOT / "orbit_wars/core/targeting.py"),
    ("harness.py", ROOT / "orbit_wars/legacy/harness.py"),
    ("agents.py", ROOT / "orbit_wars/legacy/agents.py"),
]


def _strip_local_imports(text: str) -> str:
    out: list[str] = []
    skipping_multiline_local_import = False

    for line in text.splitlines():
        stripped = line.strip()

        if skipping_multiline_local_import:
            if stripped == ")":
                skipping_multiline_local_import = False
            continue

        if stripped.startswith("from __future__ import "):
            continue
        if stripped in {
            "from radar import Radar",
            "from radar import Radar, RadarHit",
            "import targeting as T",
            "from action_space import MAX_MODEL_MOVES",
            "from harness import FEATURE_DIM, FEATURE_SCALES, N_MAX_DEFAULT",
        }:
            continue
        if stripped == "from harness import (":
            skipping_multiline_local_import = True
            continue

        out.append(line)

    return "\n".join(out).rstrip() + "\n"


def _state_dict_from_checkpoint(checkpoint: Path) -> bytes:
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def _npz_from_checkpoint(checkpoint: Path) -> bytes:
    import numpy as np
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    arrays = {
        key.replace(".", "__"): value.detach().cpu().numpy().astype(np.float32)
        for key, value in state.items()
    }
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _compressed_b64_checkpoint(checkpoint: Path) -> tuple[str, str, int, int]:
    raw = _state_dict_from_checkpoint(checkpoint)
    compressed = gzip.compress(raw, compresslevel=9)
    digest = hashlib.sha256(raw).hexdigest()
    return base64.b64encode(compressed).decode("ascii"), digest, len(raw), len(compressed)


def _numpy_checkpoint_bytes(checkpoint: Path) -> tuple[bytes, str, int]:
    raw = _npz_from_checkpoint(checkpoint)
    digest = hashlib.sha256(raw).hexdigest()
    return raw, digest, len(raw)


def _b64_numpy_checkpoint(checkpoint: Path) -> tuple[str, str, int]:
    raw, digest, size = _numpy_checkpoint_bytes(checkpoint)
    return base64.b64encode(raw).decode("ascii"), digest, size


def _latest_checkpoint(patterns: list[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(ROOT.glob(pattern))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    non_last = [p for p in candidates if not p.name.endswith(".last.pt")]
    pool = non_last or candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def resolve_checkpoint(explicit: str | None, allow_bc_fallback: bool) -> Path:
    if explicit:
        checkpoint = Path(explicit).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint not found: {checkpoint}")
        return checkpoint

    checkpoint = _latest_checkpoint(["checkpoints/rl*.pt", "checkpoints/*rl*.pt"])
    if checkpoint is not None:
        return checkpoint

    if allow_bc_fallback:
        checkpoint = _latest_checkpoint(["checkpoints/*.pt"])
        if checkpoint is not None:
            return checkpoint

    raise SystemExit(
        "No RL checkpoint found under checkpoints/. Pass --checkpoint PATH, "
        "or pass --allow-bc-fallback to build from the newest non-RL checkpoint."
    )


def _checkpoint_literal(encoded: str) -> str:
    lines = textwrap.wrap(encoded, width=88)
    body = "\n".join(f"    {line!r}" for line in lines)
    return "(\n" + body + "\n)"


def _checkpoint_assignment(encoded: str | None) -> str:
    if encoded is None:
        return "_MODEL_STATE_B64 = None\n"
    return f"_MODEL_STATE_B64 = {_checkpoint_literal(encoded)}\n"


def _append_core_sources(parts: list[str]) -> None:
    for name, path in CORE_SOURCES:
        source = path.read_text()
        parts.append(f"\n# --- {name} ---\n")
        parts.append(_strip_local_imports(source))

        if name == "targeting.py":
            parts.append(
                "\n"
                "class _TargetingNamespace:\n"
                "    pass\n\n"
                "T = _TargetingNamespace()\n"
                "T.fleet_speed = fleet_speed\n"
                "T.future_position = future_position\n"
            )


def _numpy_runtime(
    encoded: str | None,
    deterministic: bool,
    max_moves: int | None,
    weights_mode: str,
    candidate_limit: int | None,
) -> str:
    max_moves_expr = str(max_moves) if max_moves is not None else "MAX_MODEL_MOVES"
    deterministic_expr = "True" if deterministic else "False"
    candidate_limit_expr = "None" if candidate_limit is None else str(candidate_limit)
    file_loader = (
        "        for path in ('/kaggle-environments/agent/weights.npz', 'agent/weights.npz', 'weights.npz'):\n"
        "            if os.path.exists(path):\n"
        "                z = np.load(path)\n"
        "                _NP_WEIGHTS = {k.replace('__', '.'): z[k].astype(np.float32) for k in z.files}\n"
        "                return _NP_WEIGHTS\n"
    ) if weights_mode == "file" else ""
    return (
        "\n"
        "# --- NumPy model runtime ---\n"
        "import base64 as _base64\n"
        "import io as _io\n"
        "import os\n\n"
        f"{_checkpoint_assignment(encoded)}"
        "_NP_WEIGHTS = None\n"
        "_STATEFUL_NP_AGENT = None\n"
        "_STATEFUL_NP_PLAYER = None\n"
        "_LAST_STEP = None\n"
        "_NHEAD = 4\n"
        "_DMODEL = 64\n"
        "_HEAD_DIM = _DMODEL // _NHEAD\n"
        "_LN_EPS = 1e-5\n\n"
        "\n"
        "def _load_np_weights():\n"
        "    global _NP_WEIGHTS\n"
        "    if _NP_WEIGHTS is None:\n"
        f"{file_loader}"
        "        if _MODEL_STATE_B64 is not None:\n"
        "            payload = _base64.b64decode(_MODEL_STATE_B64.encode('ascii'))\n"
        "            z = np.load(_io.BytesIO(payload))\n"
        "            _NP_WEIGHTS = {k.replace('__', '.'): z[k].astype(np.float32) for k in z.files}\n"
        "            return _NP_WEIGHTS\n"
        "        raise FileNotFoundError('could not load weights.npz')\n"
        "    return _NP_WEIGHTS\n\n"
        "\n"
        "def _linear(x, weight, bias):\n"
        "    return x @ weight.T + bias\n\n"
        "\n"
        "def _layer_norm(x, weight, bias):\n"
        "    mean = x.mean(axis=-1, keepdims=True)\n"
        "    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)\n"
        "    return ((x - mean) / np.sqrt(var + _LN_EPS)) * weight + bias\n\n"
        "\n"
        "def _gelu(x):\n"
        "    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))\n\n"
        "\n"
        "def _softmax(x, axis=-1):\n"
        "    x = x - np.max(x, axis=axis, keepdims=True)\n"
        "    ex = np.exp(x)\n"
        "    return ex / np.sum(ex, axis=axis, keepdims=True)\n\n"
        "\n"
        "def _self_attn(x, w, prefix):\n"
        "    qkv = _linear(\n"
        "        x,\n"
        "        w[prefix + '.self_attn.in_proj_weight'],\n"
        "        w[prefix + '.self_attn.in_proj_bias'],\n"
        "    )\n"
        "    q, k, v = np.split(qkv, 3, axis=-1)\n"
        "    seq = x.shape[0]\n"
        "    q = q.reshape(seq, _NHEAD, _HEAD_DIM).transpose(1, 0, 2)\n"
        "    k = k.reshape(seq, _NHEAD, _HEAD_DIM).transpose(1, 0, 2)\n"
        "    v = v.reshape(seq, _NHEAD, _HEAD_DIM).transpose(1, 0, 2)\n"
        "    scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(_HEAD_DIM)\n"
        "    attn = _softmax(scores, axis=-1)\n"
        "    out = (attn @ v).transpose(1, 0, 2).reshape(seq, _DMODEL)\n"
        "    return _linear(\n"
        "        out,\n"
        "        w[prefix + '.self_attn.out_proj.weight'],\n"
        "        w[prefix + '.self_attn.out_proj.bias'],\n"
        "    )\n\n"
        "\n"
        "def _encoder_layer(x, w, layer_idx):\n"
        "    prefix = f'encoder.layers.{layer_idx}'\n"
        "    x = _layer_norm(\n"
        "        x + _self_attn(x, w, prefix),\n"
        "        w[prefix + '.norm1.weight'],\n"
        "        w[prefix + '.norm1.bias'],\n"
        "    )\n"
        "    ff = _linear(x, w[prefix + '.linear1.weight'], w[prefix + '.linear1.bias'])\n"
        "    ff = _gelu(ff)\n"
        "    ff = _linear(ff, w[prefix + '.linear2.weight'], w[prefix + '.linear2.bias'])\n"
        "    return _layer_norm(\n"
        "        x + ff,\n"
        "        w[prefix + '.norm2.weight'],\n"
        "        w[prefix + '.norm2.bias'],\n"
        "    )\n\n"
        "\n"
        "def _np_action_policy(edge_features, action_mask):\n"
        "    w = _load_np_weights()\n"
        f"    candidate_limit = {candidate_limit_expr}\n"
        "    flat_mask = action_mask.reshape(-1).astype(bool)\n"
        "    candidate_idx = np.nonzero(flat_mask)[0]\n"
        "    if candidate_limit is not None and candidate_idx.size > candidate_limit:\n"
        "        flat_features = edge_features.reshape(N_MAX_DEFAULT * N_MAX_DEFAULT, FEATURE_DIM)\n"
        "        cand = flat_features[candidate_idx]\n"
        "        # Fast local prior: favor high-value low-cost edges before the neural rerank.\n"
        "        eta = cand[:, FEATURE_ETA]\n"
        "        ships_needed = cand[:, FEATURE_SHIPS_NEEDED]\n"
        "        prod = cand[:, FEATURE_TGT_PRODUCTION]\n"
        "        kind_attack = cand[:, FEATURE_KIND_ATTACK_ENEMY] + cand[:, FEATURE_KIND_ATTACK_NEUTRAL]\n"
        "        kind_reinforce = cand[:, FEATURE_KIND_REINFORCE]\n"
        "        src_ships = cand[:, FEATURE_SRC_SHIPS]\n"
        "        score = (\n"
        "            2.0 * kind_attack * prod / (ships_needed + eta + 1.0)\n"
        "            + 0.25 * kind_reinforce\n"
        "            + 0.02 * src_ships\n"
        "            - 0.01 * eta\n"
        "        )\n"
        "        keep = np.argpartition(score, -candidate_limit)[-candidate_limit:]\n"
        "        candidate_idx = candidate_idx[keep[np.argsort(score[keep])[::-1]]]\n"
        "    else:\n"
        "        flat_features = edge_features.reshape(N_MAX_DEFAULT * N_MAX_DEFAULT, FEATURE_DIM)\n"
        "    tokens = (flat_features[candidate_idx].astype(np.float32) / FEATURE_SCALES.reshape(1, -1))\n"
        "    x = _linear(tokens, w['edge_embed.weight'], w['edge_embed.bias'])\n"
        "    for layer_idx in range(3):\n"
        "        x = _encoder_layer(x, w, layer_idx)\n"
        "    move_logits = _linear(x, w['policy_head.weight'], w['policy_head.bias']).reshape(-1)\n"
        "    pooled = x.mean(axis=0, keepdims=True)\n"
        "    stop_logit = float(_linear(pooled, w['stop_head.weight'], w['stop_head.bias'])[0, 0])\n"
        "    masked = np.full(N_MAX_DEFAULT * N_MAX_DEFAULT, -1.0e9, dtype=np.float32)\n"
        "    masked[candidate_idx] = move_logits.astype(np.float32)\n"
        "    logits = np.concatenate([masked, np.asarray([stop_logit], dtype=np.float32)])\n"
        "    return _softmax(logits, axis=0)\n\n"
        "\n"
        "def np_model_agent_actions(obs, max_moves=MAX_MODEL_MOVES, deterministic=True, view=None):\n"
        "    if view is None:\n"
        "        view = GameView(_mutable_obs(obs))\n"
        "    moves = []\n"
        "    for _ in range(max_moves):\n"
        "        action_mask = view.action_mask(SAFETY_MARGIN)\n"
        "        if not action_mask.any():\n"
        "            break\n"
        "        policy = _np_action_policy(view.edge_features, action_mask)\n"
        "        if deterministic:\n"
        "            action_idx = int(np.argmax(policy))\n"
        "        else:\n"
        "            action_idx = int(np.random.choice(policy.shape[0], p=policy))\n"
        "        stop_idx = view.n_max * view.n_max\n"
        "        if action_idx == stop_idx:\n"
        "            break\n"
        "        src_slot = action_idx // view.n_max\n"
        "        tgt_slot = action_idx % view.n_max\n"
        "        ships = view.deterministic_ship_count(src_slot, tgt_slot, SAFETY_MARGIN)\n"
        "        action = view.apply_planned_move(src_slot, tgt_slot, ships)\n"
        "        if action is None:\n"
        "            break\n"
        "        moves.append(action)\n"
        "    return moves\n\n"
        "\n"
        "class StatefulNumpyModelAgent:\n"
        "    def __init__(self, max_moves=MAX_MODEL_MOVES, deterministic=True):\n"
        "        self.max_moves = max_moves\n"
        "        self.deterministic = deterministic\n"
        "        self._view = None\n\n"
        "    def __call__(self, obs):\n"
        "        mut = _mutable_obs(obs)\n"
        "        if self._view is None:\n"
        "            self._view = GameView(mut)\n"
        "        else:\n"
        "            self._view.update_from_obs(mut)\n"
        "        return np_model_agent_actions(\n"
        "            mut, max_moves=self.max_moves, deterministic=self.deterministic, view=self._view\n"
        "        )\n\n"
        "\n"
        "# --- Kaggle entrypoint ---\n"
        "def agent(obs, config=None):\n"
        "    global _STATEFUL_NP_AGENT, _STATEFUL_NP_PLAYER, _LAST_STEP\n"
        "    player = _get(obs, 'player', 0)\n"
        "    step = int(_get(obs, 'step', 0) or 0)\n"
        "    if (_STATEFUL_NP_AGENT is None or _STATEFUL_NP_PLAYER != player\n"
        "            or (_LAST_STEP is not None and step <= _LAST_STEP)):\n"
        f"        _STATEFUL_NP_AGENT = StatefulNumpyModelAgent(max_moves={max_moves_expr}, deterministic={deterministic_expr})\n"
        "        _STATEFUL_NP_PLAYER = player\n"
        "    _LAST_STEP = step\n"
        "    return _STATEFUL_NP_AGENT(obs)\n"
    )


def _torch_model_source() -> str:
    return "\n# --- model.py ---\n" + _strip_local_imports((ROOT / "model.py").read_text())


def _torch_runtime(encoded: str, deterministic: bool, max_moves: int | None) -> str:
    max_moves_expr = str(max_moves) if max_moves is not None else "MAX_MODEL_MOVES"
    deterministic_expr = "True" if deterministic else "False"
    return (
        "\n"
        "# --- Embedded checkpoint ---\n"
        "import base64 as _base64\n"
        "import gzip as _gzip\n"
        "import io as _io\n\n"
        f"_MODEL_STATE_B64 = {_checkpoint_literal(encoded)}\n"
        "_MODEL = None\n"
        "_STATEFUL_AGENT = None\n"
        "_STATEFUL_PLAYER = None\n"
        "_LAST_STEP = None\n\n"
        "\n"
        "def _load_embedded_state_dict():\n"
        "    payload = _base64.b64decode(_MODEL_STATE_B64.encode('ascii'))\n"
        "    raw = _gzip.decompress(payload)\n"
        "    return torch.load(_io.BytesIO(raw), map_location='cpu', weights_only=False)\n\n"
        "\n"
        "def _get_model():\n"
        "    global _MODEL\n"
        "    if _MODEL is None:\n"
        "        try:\n"
        "            torch.set_num_threads(1)\n"
        "        except Exception:\n"
        "            pass\n"
        "        model = OrbitWarsTransformer()\n"
        "        model.load_state_dict(_load_embedded_state_dict())\n"
        "        model.eval()\n"
        "        _MODEL = model\n"
        "    return _MODEL\n\n"
        "\n"
        "# --- Kaggle entrypoint ---\n"
        "def agent(obs, config=None):\n"
        "    global _STATEFUL_AGENT, _STATEFUL_PLAYER, _LAST_STEP\n"
        "    player = _get(obs, 'player', 0)\n"
        "    step = int(_get(obs, 'step', 0) or 0)\n"
        "    if (_STATEFUL_AGENT is None or _STATEFUL_PLAYER != player\n"
        "            or (_LAST_STEP is not None and step <= _LAST_STEP)):\n"
        "        _STATEFUL_AGENT = StatefulModelAgent(\n"
        f"            _get_model(), max_moves={max_moves_expr}, deterministic={deterministic_expr}\n"
        "        )\n"
        "        _STATEFUL_PLAYER = player\n"
        "    _LAST_STEP = step\n"
        "    return _STATEFUL_AGENT(obs)\n"
    )


def build(
    checkpoint: Path,
    deterministic: bool = True,
    max_moves: int | None = None,
    backend: str = "numpy",
    weights_mode: str = "embedded",
    candidate_limit: int | None = None,
) -> str:
    ck_rel = checkpoint.relative_to(ROOT) if checkpoint.is_relative_to(ROOT) else checkpoint

    parts = [
        "# Generated by make_nn_submission.py. Do not edit by hand.\n",
        f"# checkpoint: {ck_rel}\n",
        f"# backend: {backend}\n",
        f"# weights_mode: {weights_mode}\n",
        "from __future__ import annotations\n\n",
    ]

    _append_core_sources(parts)

    if backend == "numpy":
        if weights_mode == "embedded":
            encoded, digest, npz_size = _b64_numpy_checkpoint(checkpoint)
        elif weights_mode == "file":
            _raw, digest, npz_size = _numpy_checkpoint_bytes(checkpoint)
            encoded = None
        else:
            raise ValueError(f"unknown weights_mode: {weights_mode}")
        parts.insert(3, f"# npz_sha256: {digest}\n")
        parts.insert(4, f"# embedded_npz_bytes: {npz_size}\n")
        parts.append(_numpy_runtime(encoded, deterministic, max_moves, weights_mode, candidate_limit))
    elif backend == "torch":
        encoded, digest, raw_size, compressed_size = _compressed_b64_checkpoint(checkpoint)
        parts.insert(3, f"# state_dict_sha256: {digest}\n")
        parts.insert(4, f"# state_dict_bytes: {raw_size}\n")
        parts.insert(5, f"# embedded_gzip_bytes: {compressed_size}\n")
        parts.append(_torch_model_source())
        parts.append(_torch_runtime(encoded, deterministic, max_moves))
    else:
        raise ValueError(f"unknown backend: {backend}")

    return "".join(parts)


def build_archive(
    checkpoint: Path,
    archive_out: Path,
    deterministic: bool,
    max_moves: int | None,
    backend: str,
    candidate_limit: int | None,
) -> None:
    if backend != "numpy":
        raise SystemExit("archive output currently supports --backend numpy only")

    source = build(
        checkpoint=checkpoint,
        deterministic=deterministic,
        max_moves=max_moves,
        backend=backend,
        weights_mode="file",
        candidate_limit=candidate_limit,
    )
    weights, _digest, _size = _numpy_checkpoint_bytes(checkpoint)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "submission.py").write_text(source)
        weights_path = root / ARCHIVE_WEIGHTS_PATH
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_bytes(weights)

        if archive_out.suffix == ".zip":
            with zipfile.ZipFile(archive_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(root / "submission.py", arcname="submission.py")
                zf.write(weights_path, arcname=str(ARCHIVE_WEIGHTS_PATH))
        else:
            with tarfile.open(archive_out, "w:gz") as tar:
                tar.add(root / "submission.py", arcname="submission.py")
                tar.add(weights_path, arcname=str(ARCHIVE_WEIGHTS_PATH))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        help="checkpoint to embed; defaults to newest RL checkpoint",
    )
    parser.add_argument(
        "--allow-bc-fallback",
        action="store_true",
        help="if no RL checkpoint exists, embed the newest non-RL .pt checkpoint",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="generated submission path",
    )
    parser.add_argument(
        "--archive-out",
        type=Path,
        default=None,
        help="write a .tar.gz/.zip containing submission.py and weights.npz",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="sample from the policy instead of deterministic argmax",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="override MAX_MODEL_MOVES in the generated entrypoint",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="rerank only the top K candidate edges with the NN; default uses all edges",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "torch"),
        default="numpy",
        help="runtime backend for generated submission; numpy avoids Kaggle torch import timeouts",
    )
    parser.add_argument(
        "--weights-mode",
        choices=("embedded", "file"),
        default="embedded",
        help="embed weights in submission.py or load agent/weights.npz from an archive",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint, args.allow_bc_fallback)
    if args.archive_out is not None:
        archive_out = args.archive_out.expanduser()
        if not archive_out.is_absolute():
            archive_out = ROOT / archive_out
        build_archive(
            checkpoint=checkpoint,
            archive_out=archive_out,
            deterministic=not args.sample,
            max_moves=args.max_moves,
            backend=args.backend,
            candidate_limit=args.candidate_limit,
        )
        print(f"wrote {archive_out}")
    else:
        out = args.out.expanduser()
        if not out.is_absolute():
            out = ROOT / out
        out.write_text(
            build(
                checkpoint=checkpoint,
                deterministic=not args.sample,
                max_moves=args.max_moves,
                backend=args.backend,
                weights_mode=args.weights_mode,
                candidate_limit=args.candidate_limit,
            )
        )
        print(f"wrote {out}")
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
