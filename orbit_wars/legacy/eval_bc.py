"""Quick eval of a BC checkpoint vs heuristic/sniper/random. Not a full eval.py."""
import argparse, time, torch
from kaggle_environments import make
from model import OrbitWarsTransformer
from agents import (
    StatefulModelAgent, heuristic_agent, nearest_planet_sniper, random_agent,
)


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    m = OrbitWarsTransformer()
    m.load_state_dict(state)
    m.to(device).eval()
    return m


def play(model_fn, opp_fn, model_seat):
    env = make("orbit_wars", debug=False)
    # kaggle-environments mishandles callable-class instances; wrap via closure.
    # MUST be 1-arg: env passes `config` as 2nd positional if signature allows,
    # which silently clobbers default-arg captures.
    fn = model_fn  # closure capture
    def wrapped_model(obs):
        return fn(obs)
    op = opp_fn
    def wrapped_opp(obs):
        return op(obs)
    agents = [wrapped_opp, wrapped_opp]
    agents[model_seat] = wrapped_model
    env.run(agents)
    final = env.steps[-1]
    rewards = [s.reward for s in final]
    statuses = [s.status for s in final]
    if rewards[0] is None or rewards[1] is None:
        print(f"    !! game ended with status={statuses} rewards={rewards}")
        # try to salvage: read last observation ship counts
        obs0 = final[0].observation
        planets = obs0.get("planets", [])
        fleets = obs0.get("fleets", [])
        ships = [0, 0]
        for p in planets:
            owner = p[1]
            if owner in (0, 1): ships[owner] += p[5]
        for f in fleets:
            owner = f[1]
            if owner in (0, 1): ships[owner] += f[6]
        rewards = ships
    mine, theirs = rewards[model_seat], rewards[1 - model_seat]
    return mine, theirs, statuses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"loading {args.checkpoint} on {args.device}")
    model = load_model(args.checkpoint, args.device)

    opponents = [
        ("heuristic", heuristic_agent),
        ("sniper", nearest_planet_sniper),
        ("random", random_agent),
    ]
    results = {}
    for name, opp in opponents:
        wins = losses = draws = 0
        margins = []
        t0 = time.time()
        for g in range(args.games):
            # Stateful agent holds a GameView across turns - one per game.
            stateful = StatefulModelAgent(model, deterministic=True)
            # alternate seat across games
            seat = g % 2
            mine, theirs, statuses = play(stateful, opp, model_seat=seat)
            margins.append(mine - theirs)
            if mine > theirs: wins += 1
            elif mine < theirs: losses += 1
            else: draws += 1
            print(f"  vs {name} game {g+1}/{args.games} seat={seat} ours={mine:.1f} opp={theirs:.1f} margin={mine-theirs:+.1f} status={statuses}")
        dur = time.time() - t0
        avg = sum(margins) / len(margins)
        results[name] = dict(wins=wins, losses=losses, draws=draws, avg_margin=avg, dur=dur)
        print(f"  -> vs {name}: {wins}W/{losses}L/{draws}D  avg margin {avg:+.1f}  ({dur:.1f}s)")
    print("\nSUMMARY")
    for name, r in results.items():
        print(f"  vs {name:10s}: {r['wins']}W/{r['losses']}L/{r['draws']}D  margin={r['avg_margin']:+.1f}")


if __name__ == "__main__":
    main()
