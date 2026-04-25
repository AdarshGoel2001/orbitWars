"""Async rotating-learner driver: workers also run PPO when the buffer fills.

Variant of rl_async.py. Same worker pool semantics, but instead of having the
parent process run PPO while N workers contend for the same cores (OpenMP
spin-wait + cache thrash), the FIRST worker to finish its current game after
the buffer fills picks up the PPO ticket and runs the update *single-threaded*
on its own core. The other N-1 workers keep playing. When the PPO worker
publishes its result, it rejoins the rollout pool.

Steady-state cores: N workers playing during rollout, (N-1) workers + 1 PPO
worker during update. Recovers the core-utilization gap of the
dedicated-learner variant without the OpenMP oversubscription penalty of the
parent-process learner.

Trade-offs vs rl_async.py:
- Per-iter ticket round-trip (model_state + optim_state + trajectories) goes
  through mp.Queue. Payload is on the order of a few MB; overhead ~200 ms.
- The chosen learner is non-deterministic (whichever worker reaches the top
  of its loop first). That's intentional and matches "first worker to finish
  becomes the learner" — no deterministic election needed.
- Staleness profile is similar to rl_async.py: trajectories in flight during
  PPO are based on the previous generation's weights and are consumed at
  iter+1 once weights.pt is republished.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import random as _rnd
import time
from pathlib import Path

import numpy as np
import torch

from orbit_wars.cpu.model import OrbitWarsEdgeTransformer
from orbit_wars.cpu.rl_opponent_pool import OpponentPool
from orbit_wars.cpu.rl_rollout import play_one_game


def write_weights_atomic(path: Path, model, opponent_pool, generation: int) -> None:
    """Tmp+rename so workers never read a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "pool_state": opponent_pool.state_dict(),
        "generation": int(generation),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _move_optim_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _run_ppo_in_worker(ticket: dict, device: torch.device) -> dict:
    """Build a local model + optimizer from the ticket, run one PPO update,
    return the new state and metrics. Runs single-threaded — caller has
    already set torch.set_num_threads(1)."""
    from orbit_wars.cpu.rl_ppo import ppo_update_step

    model = OrbitWarsEdgeTransformer().to(device)
    model.load_state_dict(ticket["model_state"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ticket["lr"],
        weight_decay=ticket["weight_decay"],
    )
    optimizer.load_state_dict(ticket["optim_state"])
    _move_optim_state_to_device(optimizer, device)

    metrics = ppo_update_step(
        model,
        ticket["trajectories"],
        optimizer,
        device=device,
        ppo_epochs=ticket["ppo_epochs"],
        ppo_batch_size=ticket["ppo_batch_size"],
        clip_ratio=ticket["clip_ratio"],
        value_coef=ticket["value_coef"],
        entropy_coef=ticket["entropy_coef"],
        gamma=ticket["gamma"],
        lambda_=ticket["lambda_"],
        max_grad_norm=ticket["max_grad_norm"],
        target_kl=ticket["target_kl"],
        normalize_advantages=ticket["normalize_advantages"],
    )

    return {
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optim_state": optimizer.state_dict(),
        "metrics": metrics,
    }


def _rotating_worker_loop(
    weights_path: str,
    traj_queue,
    ppo_request_queue,
    ppo_result_queue,
    stop_event,
    worker_id: int,
    base_seed: int,
    max_turns: int,
    deterministic: bool,
    heuristic_weight: float,
    max_snapshots: int,
):
    # Single-threaded for both rollout AND PPO duty. This is the whole point
    # of the rotating design: when this worker becomes the learner, it must
    # not spawn extra MKL/OpenMP threads that contend with sibling workers.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    device = torch.device("cpu")
    weights_path_p = Path(weights_path)

    model = OrbitWarsEdgeTransformer().to(device)
    model.eval()
    pool = OpponentPool(heuristic_weight=heuristic_weight, max_snapshots=max_snapshots)

    # Wait for the first weights publish.
    while not stop_event.is_set():
        if weights_path_p.exists():
            break
        time.sleep(0.05)

    last_mtime: float | None = None
    current_gen = -1
    local_game_i = 0

    while not stop_event.is_set():
        # PPO check first: if this worker just finished a game (or is starting
        # up) and a ticket is waiting, become the learner. Skips the disk
        # weight-reload because the ticket carries its own model state.
        ticket = None
        try:
            ticket = ppo_request_queue.get_nowait()
        except queue.Empty:
            pass

        if ticket is not None:
            try:
                result = _run_ppo_in_worker(ticket, device)
            except Exception as exc:  # pragma: no cover - keep the run alive
                print(
                    f"[rotating worker {worker_id}] PPO error: {exc!r}",
                    flush=True,
                )
                result = {"error": repr(exc)}
            ppo_result_queue.put(result)
            # Force a fresh weights reload on the next loop iter so this
            # worker plays under the just-published post-PPO weights.
            last_mtime = None
            continue

        try:
            mtime = weights_path_p.stat().st_mtime
        except FileNotFoundError:
            time.sleep(0.05)
            continue

        if last_mtime is None or mtime > last_mtime:
            try:
                snap = torch.load(weights_path_p, map_location=device, weights_only=False)
            except Exception:
                time.sleep(0.02)
                continue
            model.load_state_dict(snap["model_state"])
            model.eval()
            pool.load_state_dict(snap["pool_state"])
            current_gen = int(snap["generation"])
            last_mtime = mtime

        seed = base_seed + worker_id * 1_000_003 + local_game_i
        _rnd.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        torch.manual_seed(seed)

        opp_fn, opp_name = pool.sample(device=device)
        try:
            traj = play_one_game(
                model,
                opp_fn,
                opp_name,
                device=device,
                deterministic=deterministic,
                max_turns=max_turns,
            )
        except Exception as exc:  # pragma: no cover - keep worker alive on env hiccups
            print(f"[rotating worker {worker_id}] game error: {exc!r}", flush=True)
            local_game_i += 1
            continue

        while not stop_event.is_set():
            try:
                traj_queue.put((traj, current_gen), timeout=1.0)
                break
            except queue.Full:
                continue

        local_game_i += 1


def run_async_rotating(
    args,
    model,
    optimizer,
    opponent_pool: OpponentPool,
    start_iteration: int,
    device: torch.device,
    writer,
    save_resume_checkpoint_fn,
    target_kl,
):
    """Drive PPO via a rotating-learner async pool.

    Same checkpoint/resume contract as run_async — drop-in compatible.
    """
    last_path = args.out.with_suffix(".last.pt")
    weights_path = args.out.with_suffix(".weights.pt")

    write_weights_atomic(weights_path, model, opponent_pool, generation=start_iteration)

    ctx = mp.get_context("spawn")
    queue_cap = max(args.games_per_iter * 2, args.num_workers * 2)
    traj_queue = ctx.Queue(maxsize=queue_cap)
    # Exactly-one outstanding PPO at a time. Bounded queues prevent any
    # accidental double-dispatch even if a logic bug tries to.
    ppo_request_queue = ctx.Queue(maxsize=1)
    ppo_result_queue = ctx.Queue(maxsize=1)
    stop_event = ctx.Event()

    workers = []
    for w in range(args.num_workers):
        p = ctx.Process(
            target=_rotating_worker_loop,
            args=(
                str(weights_path),
                traj_queue,
                ppo_request_queue,
                ppo_result_queue,
                stop_event,
                w,
                int(args.seed),
                int(args.max_turns),
                bool(args.deterministic_rollout),
                float(opponent_pool.heuristic_weight),
                int(opponent_pool.max_snapshots),
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)

    print(
        f"async rotating-learner: {args.num_workers} workers, "
        f"games-per-update={args.games_per_iter}, queue_cap={queue_cap}, "
        f"weights={weights_path}",
        flush=True,
    )

    try:
        for iteration in range(start_iteration, args.iterations):
            iter_start = time.perf_counter()
            trajectories = []
            staleness: list[int] = []
            while len(trajectories) < args.games_per_iter:
                traj, gen = traj_queue.get()
                trajectories.append(traj)
                staleness.append(iteration - int(gen))

            rollout_seconds = time.perf_counter() - iter_start

            update_start = time.perf_counter()
            dispatch_start = time.perf_counter()

            ticket = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optim_state": optimizer.state_dict(),
                "trajectories": trajectories,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "ppo_epochs": args.ppo_epochs,
                "ppo_batch_size": args.ppo_batch_size,
                "clip_ratio": args.clip_ratio,
                "value_coef": args.value_coef,
                "entropy_coef": args.entropy_coef,
                "gamma": args.gamma,
                "lambda_": args.lambda_,
                "max_grad_norm": args.max_grad_norm,
                "target_kl": target_kl,
                "normalize_advantages": not args.no_advantage_norm,
            }
            ppo_request_queue.put(ticket)

            result = ppo_result_queue.get()
            ppo_dispatch_seconds = time.perf_counter() - dispatch_start

            if "error" in result:
                raise RuntimeError(f"PPO worker error: {result['error']}")

            model.load_state_dict(result["model_state"])
            optimizer.load_state_dict(result["optim_state"])
            _move_optim_state_to_device(optimizer, device)
            metrics = result["metrics"]

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

            stale_arr = np.asarray(staleness, dtype=np.int64)
            stale_mean = float(stale_arr.mean()) if stale_arr.size else 0.0
            stale_p99 = float(np.percentile(stale_arr, 99)) if stale_arr.size else 0.0
            stale_max = int(stale_arr.max()) if stale_arr.size else 0

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
            writer.add_scalar("perf/ppo_dispatch_seconds", ppo_dispatch_seconds, iteration)
            writer.add_scalar("pool/n_snapshots", len(opponent_pool._snapshots), iteration)
            writer.add_scalar("async/staleness_mean", stale_mean, iteration)
            writer.add_scalar("async/staleness_p99", stale_p99, iteration)
            writer.add_scalar("async/staleness_max", stale_max, iteration)

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
                "ppo_dispatch_s": round(ppo_dispatch_seconds, 1),
                "pool_snapshots": len(opponent_pool._snapshots),
                "stale_mean": round(stale_mean, 2),
                "stale_p99": round(stale_p99, 2),
                "stale_max": stale_max,
                "mode": "async_rotating",
            }
            print(json.dumps(row, sort_keys=True), flush=True)

            if iteration > 0 and iteration % args.snapshot_every == 0:
                opponent_pool.add_snapshot(model, f"iter_{iteration}")
                snapshot_path = args.out.with_stem(f"{args.out.stem}.snap_iter_{iteration}")
                torch.save(model.state_dict(), snapshot_path)

            save_resume_checkpoint_fn(last_path, model, optimizer, opponent_pool, iteration, args)

            write_weights_atomic(weights_path, model, opponent_pool, generation=iteration + 1)

        torch.save(model.state_dict(), args.out)
        writer.close()
        print(f"wrote final checkpoint: {args.out.resolve()}", flush=True)

    finally:
        stop_event.set()
        deadline = time.time() + 60.0
        while time.time() < deadline:
            if not any(p.is_alive() for p in workers):
                break
            try:
                traj_queue.get(timeout=0.5)
            except queue.Empty:
                pass
        for p in workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
        try:
            weights_path.unlink(missing_ok=True)
        except Exception:
            pass
