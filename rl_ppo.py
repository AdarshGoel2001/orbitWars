"""PPO training loop and loss computation.

Takes trajectories from rl_rollout, computes GAE advantages, and performs
clipped PPO updates on the model.
"""
from __future__ import annotations

import torch
from torch.distributions import Categorical

from model import OrbitWarsTransformer
from rl_rollout import GameTrajectory


def compute_gae_returns(
    trajectory: GameTrajectory,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> tuple[list[float], list[float]]:
    """Compute advantages and returns via GAE for a single game.

    Returns:
        (advantages, returns) — lists of length len(trajectory.records).
    """
    records = trajectory.records
    values = [r.value for r in records]
    rewards = [r.reward for r in records]
    dones = [r.done for r in records]

    advantages = []
    gae = 0.0
    for t in reversed(range(len(records))):
        if t == len(records) - 1:
            next_value = 0.0 if dones[t] else values[t]
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_ * gae * (1.0 - float(dones[t]))
        advantages.insert(0, gae)

    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def ppo_update_step(
    model: OrbitWarsTransformer,
    trajectories: list[GameTrajectory],
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    ppo_epochs: int = 4,
    ppo_batch_size: int = 32,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> dict[str, float]:
    """Single PPO update on a list of game trajectories.

    Args:
        model: The learner model.
        trajectories: List of GameTrajectory from rollout.
        optimizer: AdamW or similar.
        device: CPU or MPS.
        ppo_epochs: Number of passes over the replay buffer.
        ppo_batch_size: Mini-batch size for updates.
        clip_ratio: Clipping range for policy ratio.
        value_coef: Weight on value loss.
        entropy_coef: Weight on entropy bonus.
        gamma, lambda_: GAE parameters.

    Returns:
        Dict with mean loss, policy_loss, value_loss, entropy over all updates.
    """
    # Gather all sub-moves from all trajectories.
    all_edge_features = []
    all_legal_masks = []
    all_action_masks = []
    all_action_idxs = []
    all_old_logprobs = []
    all_advantages = []
    all_returns = []

    for traj in trajectories:
        adv, ret = compute_gae_returns(traj, gamma, lambda_)
        all_edge_features.extend([r.edge_features for r in traj.records])
        all_legal_masks.extend([r.legal_mask for r in traj.records])
        all_action_masks.extend([r.action_mask for r in traj.records])
        all_action_idxs.extend([r.action_idx for r in traj.records])
        all_old_logprobs.extend([r.logprob for r in traj.records])
        all_advantages.extend(adv)
        all_returns.extend(ret)

    # Convert to tensors.
    edge_features = torch.stack(
        [torch.from_numpy(x).to(device=device, dtype=torch.float32) for x in all_edge_features]
    )
    legal_masks = torch.stack(
        [torch.from_numpy(x).to(device=device, dtype=torch.bool) for x in all_legal_masks]
    )
    action_masks = torch.stack(
        [torch.from_numpy(x).to(device=device, dtype=torch.bool) for x in all_action_masks]
    )
    action_idxs = torch.tensor(all_action_idxs, dtype=torch.long, device=device)
    old_logprobs = torch.tensor(all_old_logprobs, dtype=torch.float32, device=device)
    advantages = torch.tensor(all_advantages, dtype=torch.float32, device=device)
    returns = torch.tensor(all_returns, dtype=torch.float32, device=device)

    # Normalize advantages.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO epochs over mini-batches.
    model.train()
    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n = len(all_action_idxs)
    if n == 0:
        return metrics

    update_count = 0
    for epoch in range(ppo_epochs):
        indices = torch.randperm(n, device=device)
        for start in range(0, n, ppo_batch_size):
            batch_idx = indices[start : start + ppo_batch_size]
            ef = edge_features[batch_idx]
            lm = legal_masks[batch_idx]
            am = action_masks[batch_idx]
            ai = action_idxs[batch_idx]
            old_lp = old_logprobs[batch_idx]
            adv = advantages[batch_idx]
            ret = returns[batch_idx]

            out = model(ef, lm, am)
            logits = out["action_logits"]
            value_pred = out["value"].squeeze(-1)

            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(ai)
            ratio = torch.exp(new_logprobs - old_lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (value_pred - ret).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            metrics["loss"] += float(loss.item())
            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())
            update_count += 1

    # Average over all updates.
    for key in metrics:
        metrics[key] /= max(1, update_count)

    return metrics
