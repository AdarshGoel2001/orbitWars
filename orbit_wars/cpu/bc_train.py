"""Behavior-cloning trainer for the CPU dynamic-edge Orbit Wars model."""

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

from model_cpu import OrbitWarsEdgeTransformer, count_parameters


REQUIRED_ARRAYS = (
    "edges_packed",
    "src_ids_packed",
    "tgt_ids_packed",
    "offsets",
    "n_tokens",
    "action_idx",
)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def find_shards(data_roots: list[Path], limit: int = 0) -> list[Path]:
    found = []
    for root in data_roots:
        if root.is_file() and root.name.startswith("bc_cpu_shard_") and root.suffix == ".npz":
            found.append(root)
        elif root.is_dir():
            found.extend(root.rglob("bc_cpu_shard_*.npz"))
    shards = sorted(set(found))
    if limit:
        shards = shards[:limit]
    if not shards:
        roots = ", ".join(str(p) for p in data_roots)
        raise FileNotFoundError(f"no bc_cpu_shard_*.npz files found under: {roots}")
    return shards


def split_shards(shards: list[Path], val_fraction: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(shards)
    rng.shuffle(shuffled)
    if val_fraction <= 0.0 or len(shuffled) == 1:
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


class CpuBcShardDataset(IterableDataset):
    def __init__(self, shards: list[Path], shuffle: bool = True, seed: int = 0):
        super().__init__()
        self.shards = list(shards)
        self.shuffle = shuffle
        self.seed = int(seed)
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

                n_examples = int(z["action_idx"].shape[0])
                order = np.arange(n_examples)
                if self.shuffle:
                    rng.shuffle(order)

                edges_packed = z["edges_packed"]
                src_ids_packed = z["src_ids_packed"]
                tgt_ids_packed = z["tgt_ids_packed"]
                offsets = z["offsets"]
                n_tokens = z["n_tokens"]
                action_idx = z["action_idx"]

                for row in order:
                    start = int(offsets[row])
                    end = int(offsets[row + 1])
                    n = int(n_tokens[row])
                    a = int(action_idx[row])
                    if end - start != n:
                        raise ValueError(f"{path} row {row} offset/n_tokens mismatch")
                    if a < 0 or a > n:
                        raise ValueError(f"{path} row {row} bad action_idx {a} for n={n}")
                    yield {
                        "edges": torch.from_numpy(
                            edges_packed[start:end].astype(np.float32, copy=False)
                        ),
                        "src_ids": torch.from_numpy(
                            src_ids_packed[start:end].astype(np.int64, copy=False)
                        ),
                        "tgt_ids": torch.from_numpy(
                            tgt_ids_packed[start:end].astype(np.int64, copy=False)
                        ),
                        "action_idx": a,
                        "n_tokens": n,
                    }


def collate_cpu(batch):
    batch_size = len(batch)
    n_max = max(1, max(int(item["n_tokens"]) for item in batch))
    feature_dim = int(batch[0]["edges"].shape[-1])

    edges = torch.zeros(batch_size, n_max, feature_dim, dtype=torch.float32)
    src_ids = torch.zeros(batch_size, n_max, dtype=torch.long)
    tgt_ids = torch.zeros(batch_size, n_max, dtype=torch.long)
    valid_mask = torch.zeros(batch_size, n_max, dtype=torch.bool)
    labels = torch.empty(batch_size, dtype=torch.long)
    is_stop = torch.zeros(batch_size, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = int(item["n_tokens"])
        if n:
            edges[i, :n] = item["edges"]
            src_ids[i, :n] = item["src_ids"]
            tgt_ids[i, :n] = item["tgt_ids"]
            valid_mask[i, :n] = True
        action_idx = int(item["action_idx"])
        if action_idx == n:
            labels[i] = n_max
            is_stop[i] = True
        else:
            labels[i] = action_idx

    return edges, src_ids, tgt_ids, valid_mask, labels, is_stop


def make_loader(dataset: CpuBcShardDataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_cpu,
        num_workers=0,
        pin_memory=True,
    )


def move_batch(batch, device: torch.device):
    edges, src_ids, tgt_ids, valid_mask, labels, is_stop = batch
    return (
        edges.to(device, non_blocking=True),
        src_ids.to(device, non_blocking=True),
        tgt_ids.to(device, non_blocking=True),
        valid_mask.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        is_stop.to(device, non_blocking=True),
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
    total_tokens = 0
    t0 = time.perf_counter()

    for step, batch in enumerate(loader, start=1):
        edges, src_ids, tgt_ids, valid_mask, labels, is_stop = move_batch(batch, device)
        batch_n = int(labels.numel())
        total_tokens += int(valid_mask.sum().item())

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits, _value = model(edges, src_ids, tgt_ids, valid_mask=valid_mask)
            per_example_loss = loss_fn(logits, labels)
            if stop_weight != 1.0:
                weights = torch.where(
                    is_stop,
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

        total_loss += float(loss.item()) * batch_n
        total += batch_n
        pred = logits.argmax(dim=-1)
        ok = pred.eq(labels)
        correct += int(ok.sum().item())

        stop_total += int(is_stop.sum().item())
        stop_correct += int((ok & is_stop).sum().item())
        is_move = ~is_stop
        move_total += int(is_move.sum().item())
        move_correct += int((ok & is_move).sum().item())

        if log_every and step % log_every == 0:
            elapsed = max(1e-6, time.perf_counter() - t0)
            print(
                f"  step={step} loss={total_loss / total:.4f} "
                f"acc={correct / total:.3f} move={move_correct / max(1, move_total):.3f} "
                f"stop={stop_correct / max(1, stop_total):.3f} ex/s={total / elapsed:.1f}",
                flush=True,
            )

    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "move_accuracy": move_correct / max(1, move_total),
        "stop_accuracy": stop_correct / max(1, stop_total),
        "examples": total,
        "move_examples": move_total,
        "stop_examples": stop_total,
        "tokens": total_tokens,
        "seconds": elapsed,
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
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint(path: Path, model, optimizer, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    move_optimizer_state_to_device(optimizer, device)
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", default=[Path("data/bc_cpu")],
                        help="one or more CPU BC shard files/directories")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/bc_cpu_model.pt"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stop-weight", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = choose_device(args.device)
    shards = find_shards(args.data, limit=args.limit_shards)
    train_shards, val_shards = split_shards(shards, args.val_fraction, args.seed)
    train_examples = count_examples(train_shards)
    val_examples = count_examples(val_shards) if val_shards else 0

    print(
        f"device={device} shards={len(shards)} train={len(train_shards)} "
        f"({train_examples} ex) val={len(val_shards)} ({val_examples} ex)"
    )

    model = OrbitWarsEdgeTransformer().to(device)
    print(
        f"params={count_parameters(model):,} "
        f"inference_params={count_parameters(model, include_value=False):,}"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_epoch = 1
    history = []
    best_val = float("inf")
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history = list(checkpoint.get("metrics", {}).get("history", []))
        if history:
            best_val = min(float(h.get("val_loss", float("inf"))) for h in history)
        print(f"resumed {args.resume} at epoch {start_epoch}")

    train_ds = CpuBcShardDataset(train_shards, shuffle=True, seed=args.seed)
    val_ds = CpuBcShardDataset(val_shards, shuffle=False, seed=args.seed) if val_shards else None

    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        train_loader = make_loader(train_ds, args.batch_size)
        print(f"epoch {epoch}/{args.epochs} train")
        train_metrics = run_epoch(
            model, train_loader, device, optimizer=optimizer,
            log_every=args.log_every, stop_weight=args.stop_weight,
        )

        val_metrics = None
        if val_ds is not None:
            val_ds.set_epoch(epoch)
            val_loader = make_loader(val_ds, args.batch_size)
            print(f"epoch {epoch}/{args.epochs} val")
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_move_accuracy": train_metrics["move_accuracy"],
            "train_stop_accuracy": train_metrics["stop_accuracy"],
            "train_examples": train_metrics["examples"],
            "train_seconds": train_metrics["seconds"],
        }
        if val_metrics is not None:
            row.update({
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_move_accuracy": val_metrics["move_accuracy"],
                "val_stop_accuracy": val_metrics["stop_accuracy"],
                "val_examples": val_metrics["examples"],
                "val_seconds": val_metrics["seconds"],
            })
        history.append(row)

        print(json.dumps(row, sort_keys=True), flush=True)
        save_checkpoint(
            args.out.with_suffix(".last.pt"),
            model,
            optimizer,
            epoch,
            {"history": history, "latest": row},
            args,
        )

        score = row.get("val_loss", row["train_loss"])
        if score < best_val:
            best_val = float(score)
            save_checkpoint(
                args.out,
                model,
                optimizer,
                epoch,
                {"history": history, "best": row},
                args,
            )

    history_path = args.out.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} and {args.out.with_suffix('.last.pt')}")


if __name__ == "__main__":
    main()
