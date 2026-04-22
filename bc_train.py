"""Behavior-cloning trainer for Orbit Wars.

Consumes `.npz` shards produced by `bc_data.py` and trains
`OrbitWarsTransformer` with cross-entropy over the flattened edge+stop action
space. This is intentionally plain PyTorch: no optimizer tricks, no PPO, no
value loss yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from model import OrbitWarsTransformer, count_parameters


REQUIRED_ARRAYS = ("edge_features", "legal_mask", "action_mask", "action_idx")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def find_shards(data_dirs: list[Path], limit: int = 0) -> list[Path]:
    """Find BC shards under one or more worker output directories."""
    found = []
    for data_dir in data_dirs:
        if data_dir.is_file() and data_dir.name.startswith("bc_shard_") and data_dir.suffix == ".npz":
            found.append(data_dir)
        elif data_dir.is_dir():
            found.extend(data_dir.rglob("bc_shard_*.npz"))
    shards = sorted(set(found))
    if limit:
        shards = shards[:limit]
    if not shards:
        roots = ", ".join(str(p) for p in data_dirs)
        raise FileNotFoundError(f"no bc_shard_*.npz files found under: {roots}")
    return shards


def split_shards(shards: list[Path], val_fraction: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(shards)
    rng.shuffle(shuffled)
    if val_fraction <= 0 or len(shuffled) == 1:
        return shuffled, []
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    n_val = min(n_val, len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


def count_examples(shards: list[Path]) -> int:
    total = 0
    for path in shards:
        with np.load(path) as z:
            total += int(z["action_idx"].shape[0])
    return total


class BcShardDataset(IterableDataset):
    """Shard-streaming dataset.

    Each worker/process loads one compressed shard at a time and yields rows.
    This avoids holding a large BC corpus in memory. With `shuffle=True`, shard
    order and row order inside each shard change every epoch.
    """

    def __init__(self, shards: list[Path], shuffle: bool = True, seed: int = 0):
        super().__init__()
        self.shards = list(shards)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shards = list(self.shards)
        if self.shuffle:
            rng.shuffle(shards)

        for path in shards:
            with np.load(path) as z:
                for key in REQUIRED_ARRAYS:
                    if key not in z:
                        raise KeyError(f"{path} missing required array {key!r}")

                n = int(z["action_idx"].shape[0])
                order = np.arange(n)
                if self.shuffle:
                    rng.shuffle(order)

                edge_features = z["edge_features"]
                legal_mask = z["legal_mask"]
                action_mask = z["action_mask"]
                action_idx = z["action_idx"]

                for i in order:
                    yield (
                        torch.from_numpy(edge_features[i].astype(np.float32, copy=False)),
                        torch.from_numpy(legal_mask[i].astype(bool, copy=False)),
                        torch.from_numpy(action_mask[i].astype(bool, copy=False)),
                        torch.tensor(int(action_idx[i]), dtype=torch.long),
                    )


def make_loader(dataset: BcShardDataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)


def move_batch(batch, device: torch.device):
    edge_features, legal_mask, action_mask, action_idx = batch
    return (
        edge_features.to(device, non_blocking=True),
        legal_mask.to(device, non_blocking=True),
        action_mask.to(device, non_blocking=True),
        action_idx.to(device, non_blocking=True),
    )


def run_epoch(model, loader, device, optimizer=None, log_every: int = 0,
              stop_weight: float = 1.0):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    total_loss = 0.0
    total = 0
    correct = 0
    stop_total = 0
    stop_correct = 0
    move_total = 0
    move_correct = 0
    t0 = time.perf_counter()

    for step, batch in enumerate(loader, start=1):
        edge_features, legal_mask, action_mask, action_idx = move_batch(batch, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            out = model(edge_features, legal_mask, action_mask)
            per_example_loss = loss_fn(out["action_logits"], action_idx)
            if stop_weight != 1.0:
                stop_idx = model.n_max * model.n_max
                weights = torch.where(
                    action_idx.eq(stop_idx),
                    torch.full_like(per_example_loss, float(stop_weight)),
                    torch.ones_like(per_example_loss),
                )
                loss = (per_example_loss * weights).sum() / weights.sum().clamp_min(1.0)
            else:
                loss = per_example_loss.mean()
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_n = int(action_idx.numel())
        total_loss += float(loss.item()) * batch_n
        total += batch_n

        pred = out["action_logits"].argmax(dim=-1)
        ok = pred.eq(action_idx)
        correct += int(ok.sum().item())

        stop_idx = model.n_max * model.n_max
        is_stop = action_idx.eq(stop_idx)
        stop_total += int(is_stop.sum().item())
        stop_correct += int((ok & is_stop).sum().item())
        is_move = ~is_stop
        move_total += int(is_move.sum().item())
        move_correct += int((ok & is_move).sum().item())

        if log_every and step % log_every == 0:
            elapsed = max(1e-6, time.perf_counter() - t0)
            print(
                f"  step={step} loss={total_loss / total:.4f} "
                f"acc={correct / total:.3f} ex/s={total / elapsed:.1f}",
                flush=True,
            )

    if device.type == "mps":
        torch.mps.synchronize()
    avg_loss = total_loss / max(1, total)
    return {
        "loss": avg_loss,
        "accuracy": correct / max(1, total),
        "move_accuracy": move_correct / max(1, move_total),
        "stop_accuracy": stop_correct / max(1, stop_total),
        "examples": total,
        "move_examples": move_total,
        "stop_examples": stop_total,
        "seconds": time.perf_counter() - t0,
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
    }, path)


def move_optimizer_state_to_device(optimizer, device: torch.device):
    """Move optimizer tensors after loading a checkpoint on a new device."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint(path: Path, model, optimizer, device: torch.device):
    # Local training checkpoints include argparse metadata with Path objects,
    # so PyTorch's default weights-only loader cannot read them.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    move_optimizer_state_to_device(optimizer, device)
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", default=[Path("data/bc")],
                        help="one or more shard files/directories; directories are searched recursively")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/bc_model.pt"),
                        help="checkpoint path")
    parser.add_argument("--resume", type=Path, default=None,
                        help="checkpoint to resume from, usually checkpoints/bc_model.last.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stop-weight", type=float, default=0.5,
                        help="loss multiplier for stop examples; <1 counters stop-heavy BC data")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, mps, or any torch device string")
    parser.add_argument("--limit-shards", type=int, default=0,
                        help="use only the first N shards for smoke tests")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = choose_device(args.device)
    shards = find_shards(args.data, args.limit_shards)
    train_shards, val_shards = split_shards(shards, args.val_fraction, args.seed)
    train_examples = count_examples(train_shards)
    val_examples = count_examples(val_shards) if val_shards else 0

    model = OrbitWarsTransformer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_epoch = 1
    resumed_from = None
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        resumed_from = str(args.resume)

    print(
        json.dumps({
            "device": device.type,
            "params": count_parameters(model),
            "train_shards": len(train_shards),
            "val_shards": len(val_shards),
            "train_examples": train_examples,
            "val_examples": val_examples,
            "batch_size": args.batch_size,
            "stop_weight": args.stop_weight,
            "resume": resumed_from,
            "start_epoch": start_epoch,
            "end_epoch": args.epochs,
        }, indent=2),
        flush=True,
    )

    if start_epoch > args.epochs:
        print(
            f"checkpoint already at epoch {start_epoch - 1}; "
            f"--epochs {args.epochs} leaves nothing to train"
        )
        return

    best_val = float("inf")
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        train_ds = BcShardDataset(train_shards, shuffle=True, seed=args.seed)
        train_ds.set_epoch(epoch)
        train_loader = make_loader(train_ds, args.batch_size)
        train_metrics = run_epoch(
            model, train_loader, device, optimizer=optimizer,
            log_every=args.log_every, stop_weight=args.stop_weight,
        )

        metrics = {"epoch": epoch, "train": train_metrics}

        if val_shards:
            val_ds = BcShardDataset(val_shards, shuffle=False, seed=args.seed)
            val_ds.set_epoch(epoch)
            val_loader = make_loader(val_ds, args.batch_size)
            with torch.no_grad():
                metrics["val"] = run_epoch(
                    model, val_loader, device, optimizer=None,
                    stop_weight=args.stop_weight,
                )
            val_loss = metrics["val"]["loss"]
        else:
            val_loss = train_metrics["loss"]

        history.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)

        save_checkpoint(args.out.with_suffix(".last.pt"), model, optimizer, epoch, metrics, args)
        if val_loss <= best_val:
            best_val = val_loss
            save_checkpoint(args.out, model, optimizer, epoch, metrics, args)

    history_path = args.out.with_suffix(".history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    print(f"wrote best checkpoint to {args.out}")
    print(f"wrote last checkpoint to {args.out.with_suffix('.last.pt')}")
    print(f"wrote history to {history_path}")


if __name__ == "__main__":
    main()
