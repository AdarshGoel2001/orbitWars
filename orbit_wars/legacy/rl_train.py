"""PPO self-play training loop.

Loads a BC checkpoint, collects rollouts via self-play, updates via PPO,
and maintains an opponent pool of heuristic + past snapshots.

Run with:
    .venv/bin/python rl_train.py --checkpoint checkpoints/bc_baseline.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import OrbitWarsTransformer
from rl_rollout import play_one_game
from rl_opponent_pool import OpponentPool
from rl_ppo import ppo_update_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/bc_baseline.pt"),
        help="BC checkpoint to hot-start from",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/rl_model.pt"),
        help="Output checkpoint path",
    )
    parser.add_argument("--device", default="auto", help="cpu or mps")
    parser.add_argument("--iterations", type=int, default=100, help="Number of PPO iterations")
    parser.add_argument("--games-per-iter", type=int, default=8, help="Rollouts per iteration")
    parser.add_argument("--snapshot-every", type=int, default=5, help="Snapshot frequency")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else args.device
    )

    # Load BC checkpoint.
    model = OrbitWarsTransformer().to(device)
    if args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
        print(f"loaded BC checkpoint: {args.checkpoint}", flush=True)
    else:
        print(f"no checkpoint found, training from random init", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    opponent_pool = OpponentPool(heuristic_weight=0.5, max_snapshots=8)

    print(
        json.dumps(
            {
                "device": device.type,
                "checkpoint": str(args.checkpoint.resolve()),
                "iterations": args.iterations,
                "games_per_iter": args.games_per_iter,
                "snapshot_every": args.snapshot_every,
                "lr": args.lr,
            },
            indent=2,
        ),
        flush=True,
    )

    for iteration in range(args.iterations):
        # Rollout phase.
        trajectories = []
        for game_i in range(args.games_per_iter):
            opp_fn, opp_name = opponent_pool.sample(device=device.type)
            traj = play_one_game(
                model,
                opp_fn,
                opp_name,
                device=device.type,
                deterministic=False,
            )
            trajectories.append(traj)
            if game_i < 2 or game_i == args.games_per_iter - 1:
                print(
                    f"  game {game_i}: "
                    f"turns={traj.turns} margin={traj.final_margin:+.3f} vs {opp_name}",
                    flush=True,
                )

        # PPO update phase.
        metrics = ppo_update_step(model, trajectories, optimizer, device=device.type)
        mean_margin = sum(t.final_margin for t in trajectories) / len(trajectories)

        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "games": args.games_per_iter,
                    "mean_margin": mean_margin,
                    "loss": metrics["loss"],
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                },
                indent=2,
            ),
            flush=True,
        )

        # Snapshot for opponent pool.
        if iteration % args.snapshot_every == 0 and iteration > 0:
            opponent_pool.add_snapshot(model, f"iter_{iteration}")
            snapshot_path = args.out.with_stem(f"{args.out.stem}.snap_iter_{iteration}")
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), snapshot_path)

    # Final checkpoint.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"\nwrote final checkpoint: {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
