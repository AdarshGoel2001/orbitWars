"""Behavior-cloning trainer for the experimental chunked edge policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from kaggle_environments import make

from agents_cpu import heuristic_agent_cpu
from chunked_bc_cpu import (
    build_teacher_chunk,
    chunked_bc_loss,
    collate_chunk_examples,
    load_chunk_examples_from_cpu_shards,
)
from chunked_model_cpu import ChunkedEdgePolicy


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def capture_examples(games: int, max_turns: int, max_slots: int):
    """Capture chunk examples from CPU heuristic-vs-heuristic play."""
    env = make("orbit_wars", debug=False)
    examples = []
    t0 = time.perf_counter()
    turns = 0
    for game_idx in range(games):
        env.reset()
        env.step([[], []])
        game_turns = 0
        while not env.done:
            if max_turns and game_turns >= max_turns:
                break
            obs0 = dict(env.state[0].observation)
            obs1 = dict(env.state[1].observation)
            examples.append(build_teacher_chunk(obs0, max_slots=max_slots))
            examples.append(build_teacher_chunk(obs1, max_slots=max_slots))
            moves0 = heuristic_agent_cpu(obs0, max_moves=max_slots)
            moves1 = heuristic_agent_cpu(obs1, max_moves=max_slots)
            env.step([moves0, moves1])
            turns += 1
            game_turns += 1
        print(
            f"capture game {game_idx + 1}/{games}: turns={game_turns} "
            f"examples={len(examples)}",
            flush=True,
        )
    elapsed = time.perf_counter() - t0
    return examples, {"games": games, "turns": turns, "examples": len(examples), "seconds": elapsed}


def split_examples(examples, val_fraction: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    if val_fraction <= 0.0 or len(shuffled) < 2:
        return shuffled, []
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    n_val = min(n_val, len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


def iter_batches(examples, batch_size: int, shuffle: bool, seed: int):
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        yield collate_chunk_examples([examples[i] for i in idxs])


def move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model, examples, batch_size, device, optimizer=None, seed: int = 0):
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "pointer_loss": 0.0,
        "active_loss": 0.0,
        "ship_loss": 0.0,
        "pointer_accuracy": 0.0,
        "examples": 0,
        "slots": 0,
        "active_slots": 0,
    }
    t0 = time.perf_counter()
    for batch in iter_batches(examples, batch_size, shuffle=training, seed=seed):
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, metrics = chunked_bc_loss(model, batch)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        n = int(metrics["examples"])
        totals["examples"] += n
        totals["slots"] += int(metrics["slots"])
        totals["active_slots"] += int(metrics["active_slots"])
        for key in ("loss", "pointer_loss", "active_loss", "ship_loss", "pointer_accuracy"):
            totals[key] += float(metrics[key]) * n
    denom = max(1, totals["examples"])
    for key in ("loss", "pointer_loss", "active_loss", "ship_loss", "pointer_accuracy"):
        totals[key] /= denom
    totals["seconds"] = time.perf_counter() - t0
    return totals


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, args, capture: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "capture": capture,
        "args": vars(args),
    }, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument(
        "--data",
        nargs="*",
        type=Path,
        default=[],
        help="Existing CPU BC shard files/directories. If set, skip fresh capture.",
    )
    parser.add_argument("--max-slots", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/chunked_bc_model.pt"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data and args.games < 1:
        raise SystemExit("--games must be >= 1")
    if args.max_slots < 1:
        raise SystemExit("--max-slots must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    if args.data:
        examples = load_chunk_examples_from_cpu_shards(args.data, args.max_slots)
        if not examples:
            raise SystemExit(f"no CPU BC examples found under: {args.data}")
        capture = {
            "source": "cpu_shards",
            "paths": [str(path) for path in args.data],
            "examples": len(examples),
        }
        print(f"loaded chunk examples from CPU shards: examples={len(examples)}", flush=True)
    else:
        examples, capture = capture_examples(args.games, args.max_turns, args.max_slots)
    train_examples, val_examples = split_examples(examples, args.val_fraction, args.seed)
    print(
        f"device={device} train_examples={len(train_examples)} "
        f"val_examples={len(val_examples)}",
        flush=True,
    )

    model = ChunkedEdgePolicy(n_slots=args.max_slots).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model, train_examples, args.batch_size, device, optimizer=optimizer,
            seed=args.seed + epoch,
        )
        val = run_epoch(model, val_examples or train_examples, args.batch_size, device)
        metrics = {"epoch": epoch, "train": train, "val": val}
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        save_checkpoint(args.out.with_suffix(".last.pt"), model, optimizer, epoch, metrics, args, capture)
        if val["loss"] < best_val:
            best_val = val["loss"]
            save_checkpoint(args.out, model, optimizer, epoch, metrics, args, capture)

    args.out.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
