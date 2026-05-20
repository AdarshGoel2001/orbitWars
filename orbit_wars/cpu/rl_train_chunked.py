"""PPO trainer for the chunked Orbit Wars CPU policy."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random
import time

import numpy as np
import torch

from orbit_wars.cpu.agents import heuristic_agent_cpu
from orbit_wars.cpu.chunked_model import ChunkedEdgePolicy
from orbit_wars.cpu.rl_chunked_ppo import ppo_update_step_chunked
from orbit_wars.cpu.rl_chunked_rollout import play_one_game_chunked, play_one_game_chunked_worker

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, include_value: bool = True) -> int:
    total = 0
    for name, param in model.named_parameters():
        if not include_value and name.startswith("value_head."):
            continue
        total += param.numel()
    return total


class NullWriter:
    def add_scalar(self, *_args, **_kwargs):
        return None

    def close(self):
        return None


def make_writer(path: Path | None):
    if path is None or SummaryWriter is None:
        return NullWriter()
    path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(path))


def load_checkpoint(path: Path, model: ChunkedEdgePolicy, optimizer, device: torch.device, resume: bool) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    if not resume:
        return 0
    if isinstance(checkpoint, dict) and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state_dict in optimizer.state.values():
            for key, value in list(state_dict.items()):
                if torch.is_tensor(value):
                    state_dict[key] = value.to(device)
    if isinstance(checkpoint, dict):
        return int(checkpoint.get("iteration", -1)) + 1
    return 0


def save_checkpoint(path: Path, model, optimizer, iteration: int, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iteration": int(iteration),
            "args": {
                **vars(args),
                "checkpoint": str(args.checkpoint),
                "resume": str(args.resume) if args.resume else None,
                "out": str(args.out),
                "tb_logdir": str(args.tb_logdir) if args.tb_logdir else None,
            },
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/chunked_bc_model.pt"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/rl_chunked_model.pt"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--max-slots", type=int, default=10)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-batch-size", type=int, default=64)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda_", "--lam", dest="lambda_", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--no-advantage-norm", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tb-logdir", type=Path, default=None)
    parser.add_argument("--deterministic-rollout", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    model = ChunkedEdgePolicy(n_slots=args.max_slots).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_iteration = 0
    if args.resume is not None:
        start_iteration = load_checkpoint(args.resume, model, optimizer, device, resume=True)
        print(f"resumed {args.resume} at iteration {start_iteration}", flush=True)
    elif args.checkpoint.exists():
        load_checkpoint(args.checkpoint, model, optimizer, device, resume=False)
        print(f"loaded chunked BC checkpoint: {args.checkpoint}", flush=True)
    else:
        print(f"no checkpoint found at {args.checkpoint}; training from random init", flush=True)

    if args.tb_logdir is None:
        args.tb_logdir = Path("runs") / f"{args.out.stem}_{int(time.time())}"
    writer = make_writer(args.tb_logdir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    last_path = args.out.with_suffix(".last.pt")
    target_kl = args.target_kl if args.target_kl and args.target_kl > 0 else None

    print(json.dumps({
        "device": device.type,
        "params": count_parameters(model),
        "inference_params": count_parameters(model, include_value=False),
        "checkpoint": str(args.checkpoint),
        "out": str(args.out),
        "start_iteration": start_iteration,
        "iterations": args.iterations,
        "games_per_iter": args.games_per_iter,
        "max_turns": args.max_turns,
        "max_slots": args.max_slots,
        "num_workers": args.num_workers,
        "target_kl": target_kl,
        "tb_logdir": str(args.tb_logdir),
    }, indent=2), flush=True)

    executor: ProcessPoolExecutor | None = None
    num_workers = max(1, int(args.num_workers))
    if num_workers > 1:
        executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn"))
        print(f"parallel chunked rollout: {num_workers} worker processes", flush=True)

    try:
        for iteration in range(start_iteration, args.iterations):
            iter_start = time.perf_counter()
            if executor is not None:
                model_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                tasks = [
                    {
                        "model_state": model_state_cpu,
                        "seed": args.seed + iteration * 100_003 + game_i,
                        "deterministic": args.deterministic_rollout,
                        "max_turns": args.max_turns,
                        "max_slots": args.max_slots,
                        "n_slots": args.max_slots,
                    }
                    for game_i in range(args.games_per_iter)
                ]
                trajectories = list(executor.map(play_one_game_chunked_worker, tasks))
            else:
                trajectories = [
                    play_one_game_chunked(
                        model,
                        heuristic_agent_cpu,
                        "heuristic",
                        device=device,
                        deterministic=args.deterministic_rollout,
                        max_turns=args.max_turns,
                        max_slots=args.max_slots,
                    )
                    for _ in range(args.games_per_iter)
                ]

            rollout_seconds = time.perf_counter() - iter_start
            for game_i, traj in enumerate(trajectories):
                if game_i < 2 or game_i == len(trajectories) - 1:
                    print(
                        f"  game {game_i}: records={len(traj.records)} turns={traj.turns} "
                        f"margin={traj.final_margin:+.3f} vs {traj.opponent_name}",
                        flush=True,
                    )

            update_start = time.perf_counter()
            metrics = ppo_update_step_chunked(
                model,
                trajectories,
                optimizer,
                device=device,
                ppo_epochs=args.ppo_epochs,
                ppo_batch_size=args.ppo_batch_size,
                clip_ratio=args.clip_ratio,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                gamma=args.gamma,
                lambda_=args.lambda_,
                max_grad_norm=args.max_grad_norm,
                target_kl=target_kl,
                normalize_advantages=not args.no_advantage_norm,
            )
            update_seconds = time.perf_counter() - update_start

            total_records = sum(len(t.records) for t in trajectories)
            total_sampled_active = sum(r.sampled_active for t in trajectories for r in t.records)
            total_decoded_moves = sum(r.decoded_moves for t in trajectories for r in t.records)
            total_dropped = sum(r.dropped_slots for t in trajectories for r in t.records)
            mean_margin = sum(t.final_margin for t in trajectories) / max(1, len(trajectories))
            win_rate = sum(1 for t in trajectories if t.final_margin > 0) / max(1, len(trajectories))
            mean_turns = sum(t.turns for t in trajectories) / max(1, len(trajectories))

            row = {
                "iteration": iteration,
                "games": args.games_per_iter,
                "records": total_records,
                "sampled_active": total_sampled_active,
                "decoded_moves": total_decoded_moves,
                "dropped_slots": total_dropped,
                "mean_margin": round(mean_margin, 4),
                "win_rate_vs_heuristic": round(win_rate, 3),
                "mean_turns": round(mean_turns, 1),
                "loss": round(metrics["loss"], 5),
                "policy_loss": round(metrics["policy_loss"], 5),
                "value_loss": round(metrics["value_loss"], 5),
                "entropy": round(metrics["entropy"], 5),
                "approx_kl": round(metrics["approx_kl"], 5),
                "clip_frac": round(metrics["clip_frac"], 4),
                "updates": int(metrics["updates"]),
                "early_stop": bool(metrics["early_stop"]),
                "rollout_s": round(rollout_seconds, 1),
                "update_s": round(update_seconds, 1),
            }
            print(json.dumps(row, sort_keys=True), flush=True)

            writer.add_scalar("train/mean_margin", mean_margin, iteration)
            writer.add_scalar("train/win_rate_vs_heuristic", win_rate, iteration)
            writer.add_scalar("train/decoded_moves", total_decoded_moves, iteration)
            writer.add_scalar("train/dropped_slots", total_dropped, iteration)
            writer.add_scalar("loss/total", metrics["loss"], iteration)
            writer.add_scalar("loss/policy", metrics["policy_loss"], iteration)
            writer.add_scalar("loss/value", metrics["value_loss"], iteration)
            writer.add_scalar("loss/entropy", metrics["entropy"], iteration)
            writer.add_scalar("loss/approx_kl", metrics["approx_kl"], iteration)
            writer.add_scalar("perf/rollout_seconds", rollout_seconds, iteration)
            writer.add_scalar("perf/update_seconds", update_seconds, iteration)

            if iteration > 0 and iteration % args.snapshot_every == 0:
                snapshot_path = args.out.with_stem(f"{args.out.stem}.snap_iter_{iteration}")
                torch.save(model.state_dict(), snapshot_path)
            save_checkpoint(last_path, model, optimizer, iteration, args)

        torch.save(model.state_dict(), args.out)
        writer.close()
        print(f"wrote final checkpoint: {args.out.resolve()}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
