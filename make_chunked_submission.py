"""Build a self-contained Torch submission for the chunked CPU policy."""

from __future__ import annotations

import argparse
import base64
import hashlib
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
            "from harness_cpu import FEATURE_DIM, FEATURE_SCALES, GameView_CPU",
        }:
            continue
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def _literal(encoded: str) -> str:
    lines = textwrap.wrap(encoded, width=88)
    body = "\n".join(f"    {line!r}" for line in lines)
    return "(\n" + body + "\n)"


def _checkpoint_b64(checkpoint: Path) -> tuple[str, str, int]:
    raw = checkpoint.read_bytes()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest(), len(raw)


def _append_sources(parts: list[str]):
    sources = [
        ("radar.py", ROOT / "orbit_wars/core/radar.py"),
        ("targeting.py", ROOT / "orbit_wars/core/targeting.py"),
        ("harness_cpu.py", ROOT / "orbit_wars/cpu/harness.py"),
        ("chunked_model.py", ROOT / "orbit_wars/cpu/chunked_model.py"),
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


def _runtime(encoded: str, max_moves: int, active_threshold: float) -> str:
    return f"""

# --- Chunked Torch runtime ---
import base64 as _base64
import io as _io

_CHECKPOINT_B64 = {_literal(encoded)}
_CHUNKED_MODEL = None
_STATEFUL_AGENT = None
_STATEFUL_PLAYER = None
_LAST_STEP = None
_DEVICE = torch.device("cpu")


def _load_chunked_model():
    global _CHUNKED_MODEL
    if _CHUNKED_MODEL is None:
        raw = _base64.b64decode(_CHECKPOINT_B64.encode("ascii"))
        checkpoint = torch.load(_io.BytesIO(raw), map_location=_DEVICE, weights_only=False)
        args = checkpoint.get("args", {{}})
        model = ChunkedEdgePolicy(n_slots=int(args.get("max_slots", {max_moves}))).to(_DEVICE)
        model.load_state_dict(checkpoint.get("model_state", checkpoint))
        model.eval()
        _CHUNKED_MODEL = model
    return _CHUNKED_MODEL


def _bundle_to_tensors(bundle):
    valid_mask = torch.ones(1, bundle.n, dtype=torch.bool, device=_DEVICE)
    return (
        torch.from_numpy(bundle.edges).unsqueeze(0).to(_DEVICE),
        torch.from_numpy(bundle.src_ids).long().unsqueeze(0).to(_DEVICE),
        torch.from_numpy(bundle.tgt_ids).long().unsqueeze(0).to(_DEVICE),
        valid_mask,
    )


def _model_moves_chunked(model, view, max_moves={max_moves}, active_threshold={active_threshold!r}):
    bundle = view.tokens()
    if bundle.n == 0 or max_moves <= 0:
        return []
    with torch.no_grad():
        edges, src_ids, tgt_ids, valid_mask = _bundle_to_tensors(bundle)
        out = model(
            edges,
            src_ids,
            tgt_ids,
            valid_mask=valid_mask,
            compute_value=False,
        )
    pointer_logits = out.pointer_logits[0, :max_moves]
    active_logits = out.active_logits[0, :max_moves]
    ship_delta = out.ship_delta_mu[0, :max_moves]
    token_indices = torch.argmax(pointer_logits, dim=-1).cpu().tolist()
    active = (active_logits > float(active_threshold)).cpu().tolist()
    multipliers = [multiplier_from_delta(float(delta)) for delta in ship_delta.cpu().tolist()]
    return decode_chunk_actions(view, token_indices, multipliers, active=active)


class StatefulChunkedSubmissionAgent:
    def __init__(self, max_moves={max_moves}):
        self.max_moves = int(max_moves)
        self._view = None
        self.model = _load_chunked_model()

    def __call__(self, obs):
        step = int(_get(obs, "step", 0) or 0)
        if self._view is None or step <= int(self._view.step):
            self._view = GameView_CPU(obs)
        else:
            self._view.update_from_obs(obs)
        return _model_moves_chunked(self.model, self._view, max_moves=self.max_moves)


def agent(obs, config=None):
    global _STATEFUL_AGENT, _STATEFUL_PLAYER, _LAST_STEP
    player = _get(obs, "player", 0)
    step = int(_get(obs, "step", 0) or 0)
    if (_STATEFUL_AGENT is None or _STATEFUL_PLAYER != player
            or (_LAST_STEP is not None and step <= _LAST_STEP)):
        _STATEFUL_AGENT = StatefulChunkedSubmissionAgent(max_moves={max_moves})
        _STATEFUL_PLAYER = player
    _LAST_STEP = step
    return _STATEFUL_AGENT(obs)
"""


def build(checkpoint: Path, max_moves: int, active_threshold: float) -> str:
    encoded, digest, size = _checkpoint_b64(checkpoint)
    rel = checkpoint.relative_to(ROOT) if checkpoint.is_relative_to(ROOT) else checkpoint
    parts = [
        "# Generated by make_chunked_submission.py. Do not edit by hand.\n",
        f"# checkpoint: {rel}\n",
        f"# checkpoint_sha256: {digest}\n",
        f"# embedded_checkpoint_bytes: {size}\n",
        f"# max_moves: {max_moves}\n",
        f"# active_threshold: {active_threshold}\n",
        "from __future__ import annotations\n\n",
    ]
    _append_sources(parts)
    parts.append(_runtime(encoded, max_moves, active_threshold))
    return "".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/chunked_bc_model.pt"))
    parser.add_argument("--out", type=Path, default=Path("submission_chunked.py"))
    parser.add_argument("--max-moves", type=int, default=10)
    parser.add_argument("--active-threshold", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    out = args.out
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(build(checkpoint, args.max_moves, args.active_threshold))
    print(f"wrote {out}")
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
