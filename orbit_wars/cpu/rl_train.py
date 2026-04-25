"""PPO self-play trainer for the CPU dynamic-edge Orbit Wars model."""

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

from orbit_wars.cpu.model import OrbitWarsEdgeTransformer, count_parameters
from orbit_wars.cpu.rl_opponent_pool import OpponentPool
from orbit_wars.cpu.rl_ppo import ppo_update_step
from orbit_wars.cpu.rl_rollout import play_one_game, play_one_game_worker

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional for smoke runs.
    SummaryWriter = None


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # For this RL loop, rollout is dominated by CPU-side environment and token
    # construction while MPS pays many small-transfer costs. Keep MPS opt-in.
    return torch.device("cpu")


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


def load_checkpoint(
    path: Path,
    model: OrbitWarsEdgeTransformer,
    optimizer: torch.optim.Optimizer,
    opponent_pool: OpponentPool,
    device: torch.device,
    is_resume: bool,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    if not is_resume:
        return 0

    if isinstance(checkpoint, dict) and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    if isinstance(checkpoint, dict) and "opponent_pool" in checkpoint:
        opponent_pool.load_state_dict(checkpoint["opponent_pool"])

    if isinstance(checkpoint, dict):
        return int(checkpoint.get("iteration", -1)) + 1
    return 0


def save_resume_checkpoint(path: Path, model, optimizer, opponent_pool, iteration, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "opponent_pool": opponent_pool.state_dict(),
            "iteration": int(iteration),
            "args": {
                **vars(args),
                "checkpoint": str(args.checkpoint),
                "resume": str(args.resume) if args.resume else None,
                "out": str(args.out),
                "tb_logdir": str(args.tb_logdir) if args.tb_logdir else None,
            },
        },
        tmp_path,
    )
    tmp_path.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/bc_cpu_model.pt"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/rl_cpu_model.pt"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, mps, or cpu. Auto prefers CUDA, otherwise CPU; MPS is opt-in.",
    )

    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--max-snapshots", type=int, default=8)
    parser.add_argument("--heuristic-weight", type=float, default=0.5)

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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel rollout worker processes. 1 keeps the serial path; >1 "
        "runs games concurrently via a spawn-based ProcessPoolExecutor. "
        "Workers always use CPU with torch.set_num_threads(1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    model = OrbitWarsEdgeTransformer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    opponent_pool = OpponentPool(
        heuristic_weight=args.heuristic_weight,
        max_snapshots=args.max_snapshots,
    )

    start_iteration = 0
    if args.resume is not None:
        start_iteration = load_checkpoint(
            args.resume,
            model,
            optimizer,
            opponent_pool,
            device,
            is_resume=True,
        )
        print(
            f"resumed {args.resume} at iteration {start_iteration} "
            f"with {len(opponent_pool._snapshots)} snapshots",
            flush=True,
        )
    elif args.checkpoint.exists():
        load_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            opponent_pool,
            device,
            is_resume=False,
        )
        print(f"loaded CPU BC checkpoint: {args.checkpoint}", flush=True)
    else:
        print(f"no checkpoint found at {args.checkpoint}; training from random init", flush=True)

    if args.tb_logdir is None:
        args.tb_logdir = Path("runs") / f"{args.out.stem}_{int(time.time())}"
    writer = make_writer(args.tb_logdir)

    target_kl = args.target_kl if args.target_kl and args.target_kl > 0 else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    last_path = args.out.with_suffix(".last.pt")

    print(
        json.dumps(
            {
                "device": device.type,
                "params": count_parameters(model),
                "inference_params": count_parameters(model, include_value=False),
                "checkpoint": str(args.checkpoint),
                "resume": str(args.resume) if args.resume else None,
                "out": str(args.out),
                "start_iteration": start_iteration,
                "iterations": args.iterations,
                "games_per_iter": args.games_per_iter,
                "max_turns": args.max_turns,
                "lr": args.lr,
                "ppo_batch_size": args.ppo_batch_size,
                "target_kl": target_kl,
                "tb_logdir": str(args.tb_logdir),
            },
            indent=2,
        ),
        flush=True,
    )

    if start_iteration >= args.iterations:
        print(f"already completed {start_iteration} iterations; nothing to do", flush=True)
        writer.close()
        return

    num_workers = max(1, int(args.num_workers))
    executor: ProcessPoolExecutor | None = None
    if num_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
        )
        print(f"parallel rollout: {num_workers} worker processes (CPU, 1 BLAS thread each)", flush=True)

    try:
        for iteration in range(start_iteration, args.iterations):
            iter_start = time.perf_counter()
            trajectories = []

            if executor is not None:
                model_state_cpu = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                pool_state = opponent_pool.state_dict()
                tasks = [
                    {
                        "model_state": model_state_cpu,
                        "pool_state": pool_state,
                        "heuristic_weight": opponent_pool.heuristic_weight,
                        "max_snapshots": opponent_pool.max_snapshots,
                        "seed": args.seed + iteration * 100_003 + game_i,
                        "deterministic": args.deterministic_rollout,
                        "max_turns": args.max_turns,
                    }
                    for game_i in range(args.games_per_iter)
                ]
                trajectories = list(executor.map(play_one_game_worker, tasks))
                for game_i, traj in enumerate(trajectories):
                    if game_i < 2 or game_i == args.games_per_iter - 1:
                        print(
                            f"  game {game_i}: submoves={len(traj.records)} "
                            f"turns={traj.turns} margin={traj.final_margin:+.3f} "
                            f"vs {traj.opponent_name}",
                            flush=True,
                        )
            else:
                for game_i in range(args.games_per_iter):
                    opp_fn, opp_name = opponent_pool.sample(device=device)
                    traj = play_one_game(
                        model,
                        opp_fn,
                        opp_name,
                        device=device,
                        deterministic=args.deterministic_rollout,
                        max_turns=args.max_turns,
                    )
                    trajectories.append(traj)
                    if game_i < 2 or game_i == args.games_per_iter - 1:
                        print(
                            f"  game {game_i}: submoves={len(traj.records)} "
                            f"turns={traj.turns} margin={traj.final_margin:+.3f} "
                            f"vs {opp_name}",
                            flush=True,
                        )

            rollout_seconds = time.perf_counter() - iter_start
            update_start = time.perf_counter()
            metrics = ppo_update_step(
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

            total_submoves = sum(len(t.records) for t in trajectories)
            mean_margin = sum(t.final_margin for t in trajectories) / len(trajectories)
            mean_turns = sum(t.turns for t in trajectories) / len(trajectories)
            win_rate_overall = sum(1 for t in trajectories if t.final_margin > 0) / len(trajectories)
            heuristic_games = [t for t in trajectories if t.opponent_name == "heuristic"]
            win_rate_vs_heuristic = (
                sum(1 for t in heuristic_games if t.final_margin > 0) / len(heuristic_games)
                if heuristic_games else None
            )

            writer.add_scalar("train/mean_margin", mean_margin, iteration)
            writer.add_scalar("train/win_rate_overall", win_rate_overall, iteration)
            if win_rate_vs_heuristic is not None:
                writer.add_scalar("train/win_rate_vs_heuristic", win_rate_vs_heuristic, iteration)
            writer.add_scalar("train/mean_turns", mean_turns, iteration)
            writer.add_scalar("train/total_submoves", total_submoves, iteration)
            writer.add_scalar("loss/total", metrics["loss"], iteration)
            writer.add_scalar("loss/policy", metrics["policy_loss"], iteration)
            writer.add_scalar("loss/value", metrics["value_loss"], iteration)
            writer.add_scalar("loss/entropy", metrics["entropy"], iteration)
            writer.add_scalar("loss/approx_kl", metrics["approx_kl"], iteration)
            writer.add_scalar("loss/clip_frac", metrics["clip_frac"], iteration)
            writer.add_scalar("perf/rollout_seconds", rollout_seconds, iteration)
            writer.add_scalar("perf/update_seconds", update_seconds, iteration)
            writer.add_scalar("pool/n_snapshots", len(opponent_pool._snapshots), iteration)

            row = {
                "iteration": iteration,
                "games": args.games_per_iter,
                "submoves": total_submoves,
                "mean_margin": round(mean_margin, 4),
                "win_rate_overall": round(win_rate_overall, 3),
                "win_rate_vs_heuristic": (
                    round(win_rate_vs_heuristic, 3)
                    if win_rate_vs_heuristic is not None else None
                ),
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
                "pool_snapshots": len(opponent_pool._snapshots),
            }
            print(json.dumps(row, sort_keys=True), flush=True)

            if iteration > 0 and iteration % args.snapshot_every == 0:
                opponent_pool.add_snapshot(model, f"iter_{iteration}")
                snapshot_path = args.out.with_stem(f"{args.out.stem}.snap_iter_{iteration}")
                torch.save(model.state_dict(), snapshot_path)

            save_resume_checkpoint(last_path, model, optimizer, opponent_pool, iteration, args)

        torch.save(model.state_dict(), args.out)
        writer.close()
        print(f"wrote final checkpoint: {args.out.resolve()}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
