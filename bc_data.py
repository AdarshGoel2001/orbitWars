"""Behavior-cloning data capture for Orbit Wars.

Runs heuristic-vs-heuristic games and records model-shaped supervised
examples:

    edge_features, legal_mask, action_mask -> chosen action index

The important detail is that examples are captured in the same sequential
shape used by `model_agent_actions`: choose one edge, apply the planned move
to the GameView, then choose again from the updated planned state. If the
teacher stops before MAX_MODEL_MOVES, we record a stop-action example.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from kaggle_environments import make

from action_space import MAX_MODEL_MOVES
from agents import SAFETY_MARGIN, heuristic_agent
from harness import GameView


def action_to_edge(view: GameView, action):
    """Map `[src_pid, angle, ships]` back to `(src_slot, tgt_slot)`.

    The heuristic emits env actions, not target ids. Radar gives us the real
    first-hit planet, which must be the target represented in the model label.
    """
    try:
        src_pid = int(action[0])
        angle = float(action[1])
        ships = int(action[2])
    except Exception as exc:
        raise ValueError(f"bad action shape: {action!r}") from exc

    src_slot = view.slot_of.get(src_pid)
    src = view.planets_by_id.get(src_pid)
    if src_slot is None or src is None:
        raise ValueError(f"unknown source planet in action: {action!r}")

    hit = view._get_radar().simulate_launch(src, angle, ships)
    if not hit.hit_planet or hit.target_id is None:
        raise ValueError(f"teacher action does not hit a planet first: {action!r}, hit={hit}")

    tgt_slot = view.slot_of.get(int(hit.target_id))
    if tgt_slot is None:
        raise ValueError(f"teacher action hit unknown target id {hit.target_id}: {action!r}")

    return int(src_slot), int(tgt_slot), int(ships), hit


def append_example(examples, view: GameView, action_idx: int, game_idx: int,
                   player: int, submove: int, src_slot: int = -1,
                   tgt_slot: int = -1, ships: int = 0):
    examples.append({
        "edge_features": view.edge_features.copy(),
        "legal_mask": view.legal_mask.copy(),
        "action_mask": view.action_mask(SAFETY_MARGIN).copy(),
        "planet_ids": view.planet_ids.copy(),
        "action_idx": int(action_idx),
        "game": int(game_idx),
        "step": int(view.step),
        "player": int(player),
        "submove": int(submove),
        "src_slot": int(src_slot),
        "tgt_slot": int(tgt_slot),
        "ships": int(ships),
    })


def capture_turn(obs, game_idx: int, max_moves: int):
    """Capture one player's turn and return `(moves, examples, stats)`."""
    view = GameView(obs)
    player = int(view.player)
    stop_idx = view.n_max * view.n_max
    moves = []
    examples = []
    stats = {"move_examples": 0, "stop_examples": 0}

    for submove in range(max_moves):
        action_mask = view.action_mask(SAFETY_MARGIN)

        # Query the teacher for exactly one decision from the current planned
        # state. This mirrors the model loop and lets planned fleets affect
        # the next label.
        teacher_obs = view._reconstructed_obs()
        teacher_moves = heuristic_agent(teacher_obs, max_moves=1)
        if not teacher_moves:
            append_example(examples, view, stop_idx, game_idx, player, submove)
            stats["stop_examples"] += 1
            break

        action = teacher_moves[0]
        src_slot, tgt_slot, ships, hit = action_to_edge(view, action)
        if not action_mask[src_slot, tgt_slot]:
            raise ValueError(
                "teacher chose an edge outside action_mask: "
                f"game={game_idx} step={view.step} player={player} "
                f"submove={submove} src_slot={src_slot} tgt_slot={tgt_slot} hit={hit}"
            )

        action_idx = src_slot * view.n_max + tgt_slot
        append_example(
            examples, view, action_idx, game_idx, player, submove,
            src_slot=src_slot, tgt_slot=tgt_slot, ships=ships,
        )
        stats["move_examples"] += 1

        applied = view.apply_planned_move(src_slot, tgt_slot, ships)
        if applied is None:
            raise ValueError(
                "teacher action failed apply_planned_move despite action_mask: "
                f"game={game_idx} step={view.step} player={player} "
                f"submove={submove} action={action!r}"
            )
        moves.append(applied)

    return moves, examples, stats


def flush_shard(examples, out_dir: Path, shard_idx: int):
    if not examples:
        return None

    path = out_dir / f"bc_shard_{shard_idx:05d}.npz"
    arrays = {
        "edge_features": np.stack([e["edge_features"] for e in examples]).astype(np.float32),
        "legal_mask": np.stack([e["legal_mask"] for e in examples]).astype(bool),
        "action_mask": np.stack([e["action_mask"] for e in examples]).astype(bool),
        "planet_ids": np.stack([e["planet_ids"] for e in examples]).astype(np.int32),
        "action_idx": np.asarray([e["action_idx"] for e in examples], dtype=np.int64),
        "game": np.asarray([e["game"] for e in examples], dtype=np.int32),
        "step": np.asarray([e["step"] for e in examples], dtype=np.int32),
        "player": np.asarray([e["player"] for e in examples], dtype=np.int8),
        "submove": np.asarray([e["submove"] for e in examples], dtype=np.int8),
        "src_slot": np.asarray([e["src_slot"] for e in examples], dtype=np.int16),
        "tgt_slot": np.asarray([e["tgt_slot"] for e in examples], dtype=np.int16),
        "ships": np.asarray([e["ships"] for e in examples], dtype=np.int16),
    }
    np.savez_compressed(path, **arrays)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10,
                        help="number of heuristic-vs-heuristic games to capture")
    parser.add_argument("--out", type=Path, default=Path("data/bc"),
                        help="output directory for .npz shards and manifest.json")
    parser.add_argument("--shard-size", type=int, default=512,
                        help="examples per compressed shard")
    parser.add_argument("--max-moves", type=int, default=MAX_MODEL_MOVES,
                        help="max sequential teacher moves per turn")
    parser.add_argument("--max-turns", type=int, default=0,
                        help="optional per-game turn cap for smoke tests; 0 means full game")
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
    totals["max_moves"] = args.max_moves
    totals["safety_margin"] = SAFETY_MARGIN
    totals["stop_action_idx"] = GameView({"planets": [], "fleets": []}).n_max ** 2

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(totals, indent=2, sort_keys=True) + "\n")

    print(json.dumps(totals, indent=2, sort_keys=True))
    print(f"wrote {len(shard_paths)} shard(s) to {args.out}")


if __name__ == "__main__":
    main()
