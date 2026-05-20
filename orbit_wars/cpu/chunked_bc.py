"""Behavior-cloning helpers for ``ChunkedEdgePolicy``.

The current CPU teacher acts sequentially: choose one token, mutate the view,
rebuild tokens, repeat.  The chunked policy sees the initial candidate list
once, so teacher moves are mapped back to that initial list by ``src_pid`` and
``tgt_pid``.  The ship target is a log multiplier against the initial token's
harness-computed ship count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from agents_cpu import choose_heuristic_token_cpu
from harness_cpu import (
    FEATURE_DIM,
    FEATURE_KIND_REINFORCE,
    FEATURE_SHIPS_NEEDED,
    FEATURE_SRC_SHIPS,
    FEATURE_TGT_WILL_FALL,
    GameView_CPU,
)


@dataclass
class ChunkExample:
    edges: np.ndarray
    src_ids: np.ndarray
    tgt_ids: np.ndarray
    n_tokens: int
    pointer_idx: np.ndarray
    active: np.ndarray
    ship_delta: np.ndarray


def _initial_edge_lookup(bundle) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for idx in range(bundle.n):
        src_slot = int(bundle.src_ids[idx])
        tgt_slot = int(bundle.tgt_ids[idx])
        src_pid = int(bundle.planet_ids[src_slot])
        tgt_pid = int(bundle.planet_ids[tgt_slot])
        lookup.setdefault((src_pid, tgt_pid), idx)
    return lookup


def build_teacher_chunk(obs, max_slots: int) -> ChunkExample:
    """Build one supervised chunk target from the CPU heuristic teacher."""
    if max_slots < 1:
        raise ValueError("max_slots must be >= 1")

    view = GameView_CPU(obs)
    initial = view.tokens()
    lookup = _initial_edge_lookup(initial)

    pointer_idx = np.full(max_slots, int(initial.n), dtype=np.int64)
    active = np.zeros(max_slots, dtype=np.bool_)
    ship_delta = np.zeros(max_slots, dtype=np.float32)

    for slot in range(max_slots):
        bundle = view.tokens()
        token_idx = choose_heuristic_token_cpu(view)
        if token_idx is None:
            break
        if not (0 <= token_idx < bundle.n):
            break

        src_slot = int(bundle.src_ids[token_idx])
        tgt_slot = int(bundle.tgt_ids[token_idx])
        src_pid = int(bundle.planet_ids[src_slot])
        tgt_pid = int(bundle.planet_ids[tgt_slot])
        initial_idx = lookup.get((src_pid, tgt_pid))
        if initial_idx is None:
            break

        action = view.apply_planned_move(token_idx)
        if action is None:
            break

        teacher_ships = max(1, int(action[2]))
        base_ships = max(1, int(initial.ships[initial_idx]))
        pointer_idx[slot] = int(initial_idx)
        active[slot] = True
        ship_delta[slot] = float(math.log(teacher_ships / base_ships))

    return ChunkExample(
        edges=initial.edges.copy(),
        src_ids=initial.src_ids.copy(),
        tgt_ids=initial.tgt_ids.copy(),
        n_tokens=int(initial.n),
        pointer_idx=pointer_idx,
        active=active,
        ship_delta=ship_delta,
    )


def _discover_npz(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.npz")))
        elif path.suffix == ".npz":
            files.append(path)
    return sorted(dict.fromkeys(files))


def _base_ships_from_features(edge: np.ndarray, safety_margin: int = 1) -> int:
    src_ships = max(1, int(edge[FEATURE_SRC_SHIPS]))
    if edge[FEATURE_KIND_REINFORCE] > 0.5 and edge[FEATURE_TGT_WILL_FALL] < 0.5:
        need = 1
    else:
        need = max(1, int(math.ceil(float(edge[FEATURE_SHIPS_NEEDED]))) + safety_margin)
    return max(1, min(src_ships, need))


def _row_edges(shard, row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = int(shard["offsets"][row])
    end = int(shard["offsets"][row + 1])
    return (
        shard["edges_packed"][start:end].astype(np.float32, copy=False),
        shard["src_ids_packed"][start:end].astype(np.int32, copy=False),
        shard["tgt_ids_packed"][start:end].astype(np.int32, copy=False),
    )


def load_chunk_examples_from_cpu_shards(
    paths: Iterable[str | Path],
    max_slots: int,
) -> list[ChunkExample]:
    """Reconstruct chunk targets from stored sequential CPU BC shards.

    The CPU shards contain one row per sequential teacher decision.  Within a
    turn, planet slots are stable, so later submoves can be mapped back to the
    initial candidate list by ``(src_slot, tgt_slot)``.
    """
    if max_slots < 1:
        raise ValueError("max_slots must be >= 1")

    examples: list[ChunkExample] = []
    for path in _discover_npz(paths):
        shard = np.load(path)
        groups: dict[tuple[int, int, int], list[int]] = {}
        for row in range(int(shard["action_idx"].shape[0])):
            key = (
                int(shard["game"][row]),
                int(shard["step"][row]),
                int(shard["player"][row]),
            )
            groups.setdefault(key, []).append(row)

        for rows in groups.values():
            rows.sort(key=lambda r: int(shard["submove"][r]))
            first = rows[0]
            edges, src_ids, tgt_ids = _row_edges(shard, first)
            n_tokens = int(shard["n_tokens"][first])
            edges = edges[:n_tokens].copy()
            src_ids = src_ids[:n_tokens].copy()
            tgt_ids = tgt_ids[:n_tokens].copy()

            lookup = {
                (int(src_ids[idx]), int(tgt_ids[idx])): idx
                for idx in range(n_tokens)
            }
            pointer_idx = np.full(max_slots, n_tokens, dtype=np.int64)
            active = np.zeros(max_slots, dtype=np.bool_)
            ship_delta = np.zeros(max_slots, dtype=np.float32)

            for slot, row in enumerate(rows[:max_slots]):
                action_idx = int(shard["action_idx"][row])
                row_n = int(shard["n_tokens"][row])
                if action_idx < 0 or action_idx >= row_n:
                    break
                src_slot = int(shard["src_slot"][row])
                tgt_slot = int(shard["tgt_slot"][row])
                initial_idx = lookup.get((src_slot, tgt_slot))
                if initial_idx is None:
                    break
                teacher_ships = max(1, int(shard["ships"][row]))
                base_ships = _base_ships_from_features(edges[initial_idx])
                pointer_idx[slot] = initial_idx
                active[slot] = True
                ship_delta[slot] = float(math.log(teacher_ships / base_ships))

            examples.append(ChunkExample(
                edges=edges,
                src_ids=src_ids,
                tgt_ids=tgt_ids,
                n_tokens=n_tokens,
                pointer_idx=pointer_idx,
                active=active,
                ship_delta=ship_delta,
            ))

    return examples


def collate_chunk_examples(examples: Iterable[ChunkExample]) -> dict[str, torch.Tensor]:
    examples = list(examples)
    if not examples:
        raise ValueError("examples must not be empty")

    batch_size = len(examples)
    n_slots = int(examples[0].pointer_idx.shape[0])
    n_max = max(1, max(int(ex.n_tokens) for ex in examples))
    feature_dim = int(examples[0].edges.shape[-1]) if examples[0].edges.size else FEATURE_DIM

    edges = torch.zeros(batch_size, n_max, feature_dim, dtype=torch.float32)
    src_ids = torch.zeros(batch_size, n_max, dtype=torch.long)
    tgt_ids = torch.zeros(batch_size, n_max, dtype=torch.long)
    valid_mask = torch.zeros(batch_size, n_max, dtype=torch.bool)
    pointer_idx = torch.empty(batch_size, n_slots, dtype=torch.long)
    active = torch.zeros(batch_size, n_slots, dtype=torch.bool)
    ship_delta = torch.zeros(batch_size, n_slots, dtype=torch.float32)

    for row, ex in enumerate(examples):
        if int(ex.pointer_idx.shape[0]) != n_slots:
            raise ValueError("all examples must have the same slot count")
        n = int(ex.n_tokens)
        if n:
            edges[row, :n] = torch.from_numpy(ex.edges.astype(np.float32, copy=False))
            src_ids[row, :n] = torch.from_numpy(ex.src_ids.astype(np.int64, copy=False))
            tgt_ids[row, :n] = torch.from_numpy(ex.tgt_ids.astype(np.int64, copy=False))
            valid_mask[row, :n] = True
        labels = torch.from_numpy(ex.pointer_idx.astype(np.int64, copy=False))
        labels = torch.where(labels == n, torch.full_like(labels, n_max), labels)
        pointer_idx[row] = labels
        active[row] = torch.from_numpy(ex.active.astype(np.bool_, copy=False))
        ship_delta[row] = torch.from_numpy(ex.ship_delta.astype(np.float32, copy=False))

    return {
        "edges": edges,
        "src_ids": src_ids,
        "tgt_ids": tgt_ids,
        "valid_mask": valid_mask,
        "pointer_idx": pointer_idx,
        "active": active,
        "ship_delta": ship_delta,
    }


def chunked_bc_loss(
    model,
    batch: dict[str, torch.Tensor],
    pointer_weight: float = 1.0,
    active_weight: float = 0.25,
    ship_weight: float = 0.25,
):
    """Return ``(loss, metrics)`` for one chunked BC batch."""
    out = model(
        batch["edges"],
        batch["src_ids"],
        batch["tgt_ids"],
        valid_mask=batch["valid_mask"],
        compute_value=False,
    )
    B, K, C = out.pointer_logits.shape
    pointer_loss = F.cross_entropy(
        out.pointer_logits.reshape(B * K, C),
        batch["pointer_idx"].reshape(B * K),
    )
    active_float = batch["active"].to(out.active_logits.dtype)
    active_loss = F.binary_cross_entropy_with_logits(out.active_logits, active_float)

    active = batch["active"]
    if bool(active.any()):
        ship_loss = F.smooth_l1_loss(
            out.ship_delta_mu[active],
            batch["ship_delta"][active].to(out.ship_delta_mu.dtype),
        )
    else:
        ship_loss = out.ship_delta_mu.sum() * 0.0

    loss = (
        float(pointer_weight) * pointer_loss
        + float(active_weight) * active_loss
        + float(ship_weight) * ship_loss
    )
    pred = out.pointer_logits.argmax(dim=-1)
    pointer_correct = pred.eq(batch["pointer_idx"])
    metrics = {
        "loss": float(loss.detach().item()),
        "pointer_loss": float(pointer_loss.detach().item()),
        "active_loss": float(active_loss.detach().item()),
        "ship_loss": float(ship_loss.detach().item()),
        "examples": int(B),
        "slots": int(B * K),
        "active_slots": int(active.sum().item()),
        "pointer_accuracy": float(pointer_correct.float().mean().item()),
    }
    return loss, metrics
