"""Per-turn latency bench for Orbit Wars.

Runs a single heuristic-vs-heuristic game, times each agent call and the main
components on the actual obs stream. Capped at 100 turns.

Run:
    .venv/bin/python bench.py           # CPU only
    .venv/bin/python bench.py --mps     # also time model forward on MPS
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from kaggle_environments import make

from agents import heuristic_agent, model_agent_actions, StatefulModelAgent
from harness import GameView
from model import OrbitWarsTransformer, count_parameters


BUDGET_MS = 1000.0
MAX_TURNS = 100


def stats(samples):
    samples_sorted = sorted(samples)
    n = len(samples_sorted)
    if n == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n": n,
        "mean": statistics.fmean(samples_sorted),
        "p50": samples_sorted[n // 2],
        "p99": samples_sorted[min(n - 1, int(0.99 * n))],
        "max": samples_sorted[-1],
    }


def fmt(name, s, budget=None):
    tail = f"  ({s['p99'] / budget * 100:.1f}% of {budget:.0f}ms)" if budget else ""
    return (f"  {name:<34} mean={s['mean']:7.3f}ms  p50={s['p50']:7.3f}  "
            f"p99={s['p99']:7.3f}  max={s['max']:7.3f}  n={s['n']}{tail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps", action="store_true")
    args = parser.parse_args()

    device = torch.device("mps" if args.mps else "cpu")
    model = OrbitWarsTransformer().to(device).eval()
    nparams = count_parameters(model)

    env = make("orbit_wars", debug=False)
    env.reset()
    env.step([[], []])

    heur = []
    gv_build = []
    gv_update = []
    mask_cold = []
    mask_persistent = []
    fwd = []
    model_agent = []
    model_agent_stateful = []
    stateful = StatefulModelAgent(model)

    # Warmup model on one real obs
    obs0 = dict(env.state[0].observation)
    with torch.no_grad():
        view = GameView(obs0)
        ef = torch.as_tensor(view.edge_features, device=device).unsqueeze(0)
        lm = torch.as_tensor(view.legal_mask, device=device).unsqueeze(0)
        am = torch.as_tensor(view.action_mask(), device=device).unsqueeze(0)
        for _ in range(3):
            model(ef, lm, am)
        if device.type == "mps":
            torch.mps.synchronize()

    # Persistent GameView for the cross-turn update path.
    persistent: GameView | None = None

    turn = 0
    while not env.done and turn < MAX_TURNS:
        turn += 1
        obs0 = dict(env.state[0].observation)
        obs1 = dict(env.state[1].observation)

        # heuristic agent (seat 0, used to step the game too)
        t = time.perf_counter()
        a0 = heuristic_agent(obs0)
        heur.append((time.perf_counter() - t) * 1000)

        # Cold GameView build (current path — one per turn).
        t = time.perf_counter()
        view = GameView(obs0)
        gv_build.append((time.perf_counter() - t) * 1000)

        # Cross-turn update on persistent view.
        if persistent is None:
            persistent = GameView(obs0)
        else:
            t = time.perf_counter()
            persistent.update_from_obs(obs0)
            gv_update.append((time.perf_counter() - t) * 1000)

        # action_mask on cold view (first-call, full radar validation).
        t = time.perf_counter()
        action_mask = view.action_mask()
        mask_cold.append((time.perf_counter() - t) * 1000)

        # action_mask on persistent view (same cost on first call of a turn —
        # update_from_obs invalidates the cache, so this is also a full build).
        t = time.perf_counter()
        persistent.action_mask()
        mask_persistent.append((time.perf_counter() - t) * 1000)

        # model forward (single pass)
        ef = torch.as_tensor(view.edge_features, device=device).unsqueeze(0)
        lm = torch.as_tensor(view.legal_mask, device=device).unsqueeze(0)
        am = torch.as_tensor(action_mask, device=device).unsqueeze(0)
        t = time.perf_counter()
        with torch.no_grad():
            model(ef, lm, am)
        if device.type == "mps":
            torch.mps.synchronize()
        fwd.append((time.perf_counter() - t) * 1000)

        # full model_agent_actions (cold — 3-move loop with sub-move incremental)
        t = time.perf_counter()
        model_agent_actions(model, obs0)
        model_agent.append((time.perf_counter() - t) * 1000)

        # StatefulModelAgent (reuses a GameView across turns via update_from_obs)
        t = time.perf_counter()
        stateful(obs0)
        model_agent_stateful.append((time.perf_counter() - t) * 1000)

        a1 = heuristic_agent(obs1)
        env.step([a0, a1])

    print(f"device={device.type}  model params={nparams:,}  turns sampled={turn}\n")
    print(fmt("heuristic_agent", stats(heur), budget=BUDGET_MS))
    print(fmt("GameView.__init__  (cold)", stats(gv_build)))
    print(fmt("GameView.update_from_obs", stats(gv_update)))
    print(fmt("action_mask  (cold view)", stats(mask_cold)))
    print(fmt("action_mask  (persistent)", stats(mask_persistent)))
    print(fmt(f"model.forward ({device.type})", stats(fwd)))
    print(fmt("model_agent_actions  (cold)", stats(model_agent), budget=BUDGET_MS))
    print(fmt("StatefulModelAgent   (warm)", stats(model_agent_stateful), budget=BUDGET_MS))

    # Headline comparisons.
    if gv_build and gv_update:
        cold = stats(gv_build)
        warm = stats(gv_update)
        print()
        print(f"  cross-turn state-build speedup: "
              f"cold mean={cold['mean']:.2f}ms → warm mean={warm['mean']:.2f}ms  "
              f"({(1 - warm['mean']/cold['mean']) * 100:.1f}% reduction)")
    if model_agent and model_agent_stateful:
        cold_agent = stats(model_agent)
        warm_agent = stats(model_agent_stateful)
        print(f"  model-agent end-to-end speedup: "
              f"cold mean={cold_agent['mean']:.2f}ms → warm mean={warm_agent['mean']:.2f}ms  "
              f"({(1 - warm_agent['mean']/cold_agent['mean']) * 100:.1f}% reduction)")


if __name__ == "__main__":
    main()
