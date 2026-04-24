"""Evaluate a trained CPU edge model against Orbit Wars baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from kaggle_environments import make

from agents import nearest_planet_sniper
from agents_cpu import heuristic_agent_cpu, load_cpu_model_agent


def total_ships(obs, player: int) -> int:
    score = 0
    for p in obs["planets"]:
        if p[1] == player:
            score += int(p[5])
    for f in obs["fleets"]:
        if f[1] == player:
            score += int(f[6])
    return score


def agent_factory(name: str, checkpoint: Path | None, device: str):
    if name == "model":
        if checkpoint is None:
            raise ValueError("model agent requires --checkpoint")
        return load_cpu_model_agent(checkpoint, device=device)
    if name == "heuristic_cpu":
        return heuristic_agent_cpu
    if name == "sniper":
        return nearest_planet_sniper
    raise ValueError(f"unknown agent {name!r}")


def reset_agent(agent):
    if hasattr(agent, "reset"):
        agent.reset()


def run_game(left, right):
    reset_agent(left)
    reset_agent(right)
    env = make("orbit_wars", debug=False)
    env.reset()
    env.step([[], []])
    while not env.done:
        obs0 = env.state[0].observation
        obs1 = env.state[1].observation
        env.step([left(obs0), right(obs1)])
    final0 = env.state[0].observation
    final1 = env.state[1].observation
    score0 = total_ships(final0, 0)
    score1 = total_ships(final1, 1)
    return {
        "score0": score0,
        "score1": score1,
        "margin0": score0 - score1,
        "steps": int(final0["step"]),
        "status0": env.state[0].status,
        "status1": env.state[1].status,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/bc_cpu_model.pt"))
    parser.add_argument("--opponent", choices=["heuristic_cpu", "sniper"], default="heuristic_cpu")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    model_wins = 0
    opp_wins = 0
    draws = 0
    margins = []
    rows = []
    t0 = time.perf_counter()

    for game_idx in range(args.games):
        # Alternate seats to avoid seat-specific map effects.
        model_left = (game_idx % 2 == 0)
        model_agent = agent_factory("model", args.checkpoint, args.device)
        opp_agent = agent_factory(args.opponent, args.checkpoint, args.device)
        if model_left:
            result = run_game(model_agent, opp_agent)
            model_margin = result["margin0"]
        else:
            result = run_game(opp_agent, model_agent)
            model_margin = -result["margin0"]
        margins.append(model_margin)
        if model_margin > 0:
            model_wins += 1
        elif model_margin < 0:
            opp_wins += 1
        else:
            draws += 1
        row = {
            "game": game_idx,
            "model_left": model_left,
            "model_margin": model_margin,
            **result,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "opponent": args.opponent,
        "games": args.games,
        "model_wins": model_wins,
        "opponent_wins": opp_wins,
        "draws": draws,
        "win_rate": model_wins / max(1, args.games),
        "avg_margin": sum(margins) / max(1, len(margins)),
        "seconds": time.perf_counter() - t0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
