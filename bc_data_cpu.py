"""Behavior-cloning data capture for the CPU dynamic-edge stack.

Captures heuristic-vs-heuristic games using ``GameView_CPU`` and the CPU-token
teacher. Each supervised example is one planned sub-move:

    edges/src_ids/tgt_ids for the current TokenBundle -> action_idx

``action_idx == n_tokens`` is the stop action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from kaggle_environments import make

from action_space import MAX_MODEL_MOVES
from agents_cpu import choose_heuristic_token_cpu
from harness_cpu import FEATURE_DIM, GameView_CPU


def append_example(examples, view: GameView_CPU, action_idx: int, game_idx: int,
                   player: int, submove: int, src_slot: int = -1,
                   tgt_slot: int = -1, ships: int = 0):
    bundle = view.tokens()
    if action_idx < 0 or action_idx > bundle.n:
        raise ValueError(f"action_idx {action_idx} outside [0, {bundle.n}]")

    src_pid = -1
    tgt_pid = -1
    if 0 <= src_slot < bundle.num_planets:
        src_pid = int(bundle.planet_ids[src_slot])
    if 0 <= tgt_slot < bundle.num_planets:
        tgt_pid = int(bundle.planet_ids[tgt_slot])

    examples.append({
        "edges": bundle.edges.copy(),
        "src_ids": bundle.src_ids.copy(),
        "tgt_ids": bundle.tgt_ids.copy(),
        "action_idx": int(action_idx),
        "n_tokens": int(bundle.n),
        "game": int(game_idx),
        "step": int(view.step),
        "player": int(player),
        "submove": int(submove),
        "src_slot": int(src_slot),
        "tgt_slot": int(tgt_slot),
        "src_pid": int(src_pid),
        "tgt_pid": int(tgt_pid),
        "ships": int(ships),
    })


def capture_turn(obs, game_idx: int, max_moves: int):
    """Capture one player's turn and return ``(moves, examples, stats)``."""
    view = GameView_CPU(obs)
    player = int(view.player)
    moves = []
    examples = []
    stats = {"move_examples": 0, "stop_examples": 0}

    for submove in range(max_moves):
        bundle = view.tokens()
        token_idx = choose_heuristic_token_cpu(view)
        if token_idx is None:
            append_example(examples, view, bundle.n, game_idx, player, submove)
            stats["stop_examples"] += 1
            break

        if not (0 <= token_idx < bundle.n):
            raise ValueError(
                f"teacher chose token {token_idx} outside bundle size {bundle.n}"
            )

        src_slot = int(bundle.src_ids[token_idx])
        tgt_slot = int(bundle.tgt_ids[token_idx])
        ships = int(bundle.ships[token_idx])
        append_example(
            examples, view, token_idx, game_idx, player, submove,
            src_slot=src_slot, tgt_slot=tgt_slot, ships=ships,
        )
        stats["move_examples"] += 1

        action = view.apply_planned_move(token_idx)
        if action is None:
            raise ValueError(
                "teacher token failed apply_planned_move: "
                f"game={game_idx} step={view.step} player={player} "
                f"submove={submove} token_idx={token_idx}"
            )
        moves.append(action)

    return moves, examples, stats


def flush_shard(examples, out_dir: Path, shard_idx: int):
    if not examples:
        return None

    path = out_dir / f"bc_cpu_shard_{shard_idx:05d}.npz"
    n_examples = len(examples)
    n_tokens = np.asarray([e["n_tokens"] for e in examples], dtype=np.int32)
    offsets = np.zeros(n_examples + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(n_tokens, dtype=np.int64)
    total_n = int(offsets[-1])

    if total_n:
        edges_packed = np.concatenate([e["edges"] for e in examples], axis=0)
        src_ids_packed = np.concatenate([e["src_ids"] for e in examples], axis=0)
        tgt_ids_packed = np.concatenate([e["tgt_ids"] for e in examples], axis=0)
    else:
        edges_packed = np.zeros((0, FEATURE_DIM), dtype=np.float32)
        src_ids_packed = np.zeros((0,), dtype=np.int32)
        tgt_ids_packed = np.zeros((0,), dtype=np.int32)

    arrays = {
        "edges_packed": edges_packed.astype(np.float32, copy=False),
        "src_ids_packed": src_ids_packed.astype(np.int32, copy=False),
        "tgt_ids_packed": tgt_ids_packed.astype(np.int32, copy=False),
        "offsets": offsets,
        "n_tokens": n_tokens,
        "action_idx": np.asarray([e["action_idx"] for e in examples], dtype=np.int64),
        "game": np.asarray([e["game"] for e in examples], dtype=np.int32),
        "step": np.asarray([e["step"] for e in examples], dtype=np.int32),
        "player": np.asarray([e["player"] for e in examples], dtype=np.int8),
        "submove": np.asarray([e["submove"] for e in examples], dtype=np.int8),
        "src_slot": np.asarray([e["src_slot"] for e in examples], dtype=np.int16),
        "tgt_slot": np.asarray([e["tgt_slot"] for e in examples], dtype=np.int16),
        "src_pid": np.asarray([e["src_pid"] for e in examples], dtype=np.int16),
        "tgt_pid": np.asarray([e["tgt_pid"] for e in examples], dtype=np.int16),
        "ships": np.asarray([e["ships"] for e in examples], dtype=np.int16),
    }
    np.savez_compressed(path, **arrays)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10,
                        help="number of CPU-heuristic games to capture")
    parser.add_argument("--out", type=Path, default=Path("data/bc_cpu"),
                        help="output directory for .npz shards and manifest.json")
    parser.add_argument("--shard-size", type=int, default=512,
                        help="examples per compressed shard")
    parser.add_argument("--max-moves", type=int, default=MAX_MODEL_MOVES,
                        help="max sequential teacher moves per turn")
    parser.add_argument("--max-turns", type=int, default=0,
                        help="optional per-game turn cap; 0 means full game")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.games < 1:
        raise SystemExit("--games must be >= 1")
    if args.shard_size < 1:
        raise SystemExit("--shard-size must be >= 1")
    if args.max_moves < 1:
        raise SystemExit("--max-moves must be >= 1")

    args.out.mkdir(parents=True, exist_ok=True)

    pending = []
    shard_paths = []
    shard_idx = 0
    totals = {
        "games": args.games,
        "turns": 0,
        "examples": 0,
        "move_examples": 0,
        "stop_examples": 0,
        "moves": 0,
        "feature_dim": FEATURE_DIM,
        "max_moves": args.max_moves,
        "stop_action": "n_tokens",
    }

    t0 = time.perf_counter()
    env = make("orbit_wars", debug=False)

    for game_idx in range(args.games):
        env.reset()
        env.step([[], []])
        turns = 0

        while not env.done:
            if args.max_turns and turns >= args.max_turns:
                break

            obs0 = dict(env.state[0].observation)
            obs1 = dict(env.state[1].observation)
            moves0, examples0, stats0 = capture_turn(obs0, game_idx, args.max_moves)
            moves1, examples1, stats1 = capture_turn(obs1, game_idx, args.max_moves)

            pending.extend(examples0)
            pending.extend(examples1)
            totals["examples"] += len(examples0) + len(examples1)
            totals["move_examples"] += stats0["move_examples"] + stats1["move_examples"]
            totals["stop_examples"] += stats0["stop_examples"] + stats1["stop_examples"]
            totals["moves"] += len(moves0) + len(moves1)

            while len(pending) >= args.shard_size:
                shard = pending[:args.shard_size]
                del pending[:args.shard_size]
                path = flush_shard(shard, args.out, shard_idx)
                shard_paths.append(path.name)
                shard_idx += 1

            env.step([moves0, moves1])
            turns += 1
            totals["turns"] += 1

        print(f"game {game_idx + 1}/{args.games}: turns={turns} examples={totals['examples']}")

    if pending:
        path = flush_shard(pending, args.out, shard_idx)
        shard_paths.append(path.name)

    totals["seconds"] = round(time.perf_counter() - t0, 3)
    totals["shards"] = shard_paths

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(totals, indent=2, sort_keys=True) + "\n")

    print(json.dumps(totals, indent=2, sort_keys=True))
    print(f"wrote {len(shard_paths)} shard(s) to {args.out}")


if __name__ == "__main__":
    main()
