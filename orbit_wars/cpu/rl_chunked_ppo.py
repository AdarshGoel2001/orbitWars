"""PPO update for chunked Orbit Wars policies."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.distributions import Bernoulli, Categorical

from orbit_wars.cpu.chunked_model import ChunkedEdgePolicy
from orbit_wars.cpu.rl_chunked_rollout import ChunkedGameTrajectory, ChunkedTurnRecord


def compute_gae_returns_chunked(
    trajectory: ChunkedGameTrajectory,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> tuple[list[float], list[float]]:
    records = trajectory.records
    n = len(records)
    if n == 0:
        return [], []

    values = [r.value for r in records]
    rewards = [r.reward for r in records]
    dones = [r.done for r in records]
    advantages = [0.0] * n
    gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = 0.0 if dones[t] else values[t]
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_ * gae * (1.0 - float(dones[t]))
        advantages[t] = gae

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def pad_chunk_records(records: Sequence[ChunkedTurnRecord], device: torch.device) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("cannot pad empty record batch")

    batch_size = len(records)
    n_max = max(1, max(int(r.n_tokens) for r in records))
    n_slots = int(records[0].pointer_idx.shape[0])
    feature_dim = int(records[0].edges.shape[-1])

    edges = torch.zeros(batch_size, n_max, feature_dim, dtype=torch.float32)
    src_ids = torch.zeros(batch_size, n_max, dtype=torch.long)
    tgt_ids = torch.zeros(batch_size, n_max, dtype=torch.long)
    valid_mask = torch.zeros(batch_size, n_max, dtype=torch.bool)
    pointer_idx = torch.empty(batch_size, n_slots, dtype=torch.long)
    active = torch.zeros(batch_size, n_slots, dtype=torch.bool)
    ship_delta = torch.zeros(batch_size, n_slots, dtype=torch.float32)

    for i, record in enumerate(records):
        if int(record.pointer_idx.shape[0]) != n_slots:
            raise ValueError("all records in a batch must have the same slot count")
        n = int(record.n_tokens)
        if n > 0:
            edges[i, :n] = torch.from_numpy(record.edges)
            src_ids[i, :n] = torch.from_numpy(record.src_ids.astype(np.int64, copy=False))
            tgt_ids[i, :n] = torch.from_numpy(record.tgt_ids.astype(np.int64, copy=False))
            valid_mask[i, :n] = True
        labels = torch.from_numpy(record.pointer_idx.astype(np.int64, copy=False))
        labels = torch.where(labels == n, torch.full_like(labels, n_max), labels)
        pointer_idx[i] = labels
        active[i] = torch.from_numpy(record.active.astype(np.bool_, copy=False))
        ship_delta[i] = torch.from_numpy(record.ship_delta.astype(np.float32, copy=False))

    return {
        "edges": edges.to(device, non_blocking=True),
        "src_ids": src_ids.to(device, non_blocking=True),
        "tgt_ids": tgt_ids.to(device, non_blocking=True),
        "valid_mask": valid_mask.to(device, non_blocking=True),
        "pointer_idx": pointer_idx.to(device, non_blocking=True),
        "active": active.to(device, non_blocking=True),
        "ship_delta": ship_delta.to(device, non_blocking=True),
    }


def chunked_action_logprob_entropy(model: ChunkedEdgePolicy, batch: dict[str, torch.Tensor]):
    out = model(
        batch["edges"],
        batch["src_ids"],
        batch["tgt_ids"],
        valid_mask=batch["valid_mask"],
        compute_value=True,
    )
    pointer_dist = Categorical(logits=out.pointer_logits)
    active_dist = Bernoulli(logits=out.active_logits)
    pointer_logprob = pointer_dist.log_prob(batch["pointer_idx"])
    active_float = batch["active"].to(out.active_logits.dtype)
    active_logprob = active_dist.log_prob(active_float)
    active_mask = batch["active"].to(pointer_logprob.dtype)
    logprob = (active_logprob + pointer_logprob * active_mask).sum(dim=-1)
    pointer_entropy = pointer_dist.entropy().mean(dim=-1)
    active_entropy = active_dist.entropy().mean(dim=-1)
    entropy = pointer_entropy + active_entropy
    return logprob, entropy, out.value


def ppo_update_step_chunked(
    model: ChunkedEdgePolicy,
    trajectories: list[ChunkedGameTrajectory],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    ppo_epochs: int = 4,
    ppo_batch_size: int = 64,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    max_grad_norm: float = 1.0,
    target_kl: float | None = 0.03,
    normalize_advantages: bool = True,
) -> dict[str, float]:
    if not isinstance(device, torch.device):
        device = torch.device(device)

    all_records: list[ChunkedTurnRecord] = []
    all_advantages: list[float] = []
    all_returns: list[float] = []
    for traj in trajectories:
        advantages, returns = compute_gae_returns_chunked(traj, gamma=gamma, lambda_=lambda_)
        all_records.extend(traj.records)
        all_advantages.extend(advantages)
        all_returns.extend(returns)

    metrics = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "updates": 0.0,
        "early_stop": 0.0,
    }
    n = len(all_records)
    if n == 0:
        return metrics

    advantages_np = np.asarray(all_advantages, dtype=np.float32)
    returns_np = np.asarray(all_returns, dtype=np.float32)
    old_logprobs_np = np.asarray([r.logprob for r in all_records], dtype=np.float32)
    if normalize_advantages and n > 1:
        advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)

    advantages_t = torch.from_numpy(advantages_np)
    returns_t = torch.from_numpy(returns_np)
    old_logprobs_t = torch.from_numpy(old_logprobs_np)

    model.train()
    update_count = 0
    stopped_early = False
    for _epoch in range(ppo_epochs):
        permutation = np.random.permutation(n)
        epoch_kl = 0.0
        epoch_updates = 0

        for start in range(0, n, ppo_batch_size):
            batch_idx_np = permutation[start:start + ppo_batch_size]
            batch_records = [all_records[int(i)] for i in batch_idx_np]
            batch = pad_chunk_records(batch_records, device)

            batch_idx = torch.from_numpy(batch_idx_np.astype(np.int64, copy=False))
            adv = advantages_t[batch_idx].to(device, non_blocking=True)
            ret = returns_t[batch_idx].to(device, non_blocking=True)
            old_logprob = old_logprobs_t[batch_idx].to(device, non_blocking=True)

            new_logprob, entropy, value_pred = chunked_action_logprob_entropy(model, batch)
            ratio = torch.exp(new_logprob - old_logprob)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (value_pred - ret).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = float((old_logprob - new_logprob).mean().item())
                clip_frac = float(((ratio - 1.0).abs() > clip_ratio).float().mean().item())

            metrics["loss"] += float(loss.item())
            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy_mean.item())
            metrics["approx_kl"] += approx_kl
            metrics["clip_frac"] += clip_frac
            update_count += 1
            epoch_kl += approx_kl
            epoch_updates += 1

        if target_kl is not None and epoch_updates > 0:
            if epoch_kl / epoch_updates > 1.5 * target_kl:
                stopped_early = True
                break

    for key in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
        metrics[key] /= max(1, update_count)
    metrics["updates"] = float(update_count)
    metrics["early_stop"] = 1.0 if stopped_early else 0.0
    return metrics
