# Orbit Wars — Kaggle agent

Goal: competitive agent for [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars/overview). Repo also has a Flask UI (`server.py`) for human-vs-agent playtesting.

## How to run

**Always use the venv.** Never install into global Python.

```bash
.venv/bin/python server.py          # UI at http://127.0.0.1:5000
.venv/bin/pip install <pkg>         # adding deps
```

Python 3.13, `kaggle-environments>=1.28.0`, `flask`, `torch`. On import, `kaggle-environments` emits a benign `Loading environment cabt failed` and `OpenSpiel environments` line — ignore.

## Repo layout

```
orbit_wars/
├── core/    radar.py, targeting.py, action_space.py
├── cpu/     active dynamic-edge stack (harness, model, bc_*, rl_*, agents)
├── legacy/  padded 50×50 stack — reference only, not deployed
└── apps/server.py
*.py at root        # thin wrappers around orbit_wars/* so old commands still work
make_cpu_submission.py / submission_cpu.py   # NumPy submission builder + artifact
checkpoints/        # bc_cpu_model.pt, rl_cpu_model.pt, rl_cpu_model.last.pt
data/bc_cpu*        # CPU BC shards
markdowns/          # design + post-mortem notes (rl_findings_2026-04-26.md, RL-scripting.md)
LONG_TERM_FIXES.md  # deferred correctness items
```

## Active stack — CPU dynamic-edge

The deployable path is `orbit_wars/cpu/*`. Legacy `orbit_wars/legacy/*` (padded 50×50, 208K params) stays for A/B comparison; do not extend it.

Pipeline: `GameView_CPU → OrbitWarsEdgeTransformer → bc_data → bc_train → rl_train → make_cpu_submission → submission_cpu.py`.

Key properties:
- **Dynamic tokens**: one per radar-valid edge. Observed N: mean 174, p95 310. Comets dropped in v1.
- **Threshold-gated attention** (`axial_threshold=256`): full attention below, axial (src then tgt) above. Reason: sparse attention pays on serial CPU, not on parallel GPU.
- **Deterministic ship sizing**: harness sets ship count per edge; model only picks `(src, tgt)` or stop.
- **Tanh-approx GELU** everywhere so PyTorch training and NumPy submission match bit-for-bit.
- **Model**: d_model=32, 2 encoder blocks, 46,723 params (17,634 at inference, no value head).

Cross-turn fleet predictions are reused (`update_from_obs`); per-turn token list rebuilds fully (not incremental). That's the bottleneck for `max_moves > 1`.

## Checkpoints

- `bc_cpu_model.pt` — BC from heuristic teacher. 19,545 examples, 40 shards. Val acc 0.959 (move 0.941, stop 1.0). Eval: 4/4 vs sniper, 2/4 vs heuristic_cpu.
- `rl_cpu_model.pt` — first PPO pass on GCP `c3-highcpu-8`, ~134 iters total. Per-iter `win_rate_vs_heuristic` peaked at 0.63 (20-iter window, iters 129–148) then drifted to 0.50. No clean N-game eval yet — that's the next action item.
- `rl_cpu_model.last.pt` — resume state (model + optimizer + opp pool + iter).

## Latency

Generated submission, M2 CPU. Kaggle is 5–8× slower; budget is wall-clock 1s/turn + 60s overage (Bovard, 2026-05-19).

| `max_moves` | mean | p95 | max |
|---|---|---|---|
| 1 | 24.5 ms | 63.8 ms | 75.7 ms |
| 2 | 48 ms | 158 ms | 215 ms |
| 3 | — | ~1529 ms | — (not viable without token-rebuild optimization) |

Current submission uses `max_moves=2`. If it ever times out on LB, regenerate with `max_moves=1`.

## Kaggle runtime (Bovard confirmed, 2026-05-19)

- **1.6 vCPU**, **~8 GB RAM**, wall-clock timeout
- Fresh process per episode; module-level caches OK *within* an episode
- Same hardware every game including LB validation

## RL infrastructure

Three rollout drivers in `orbit_wars/cpu/rl_*`. All within 3% of theoretical lower bound at 8 cores — pick by ergonomics, not speed:
- **Sync** (default, `ProcessPoolExecutor`): workers idle during PPO.
- **Async** (`--async-rollout`, `rl_async.py`): workers play continuously, learner consumes a queue, atomic weight-file propagation via mtime poll.
- **Rotating-learner** (`rl_async_rotating.py`): first worker to finish a game after the buffer fills runs PPO single-threaded on its core; others keep playing.

PPO details:
- Reward = per-turn ΔΦ where Φ = `(my_ships - opp_ships) / total_ships`. Potential-based shaping — telescopes to terminal margin under undiscounted return; under γ=0.99 the discount attenuates the long-horizon credit somewhat but the optimum is unchanged (Ng-Harada-Russell '99).
- Opponent pool: 33% heuristic / 67% past snapshots once pool fills. FIFO at `--max-snapshots=8`. PFSP not implemented.
- Async staleness logged as `async/staleness_*`. Safe while `approx_kl < ~0.02`.

Known issues from prior run (see `markdowns/rl_findings_2026-04-26.md`):
- Entropy regularizer accumulation pushed policy toward uniform over ~30k steps; halving `ent_coef` at iter 174 didn't reverse within 17 iters.
- Self-play pool dilution: only 33% of training games against the actual eval target.
- No N-game tournament eval ever run.

Suggested resume command:

```bash
.venv/bin/python rl_train_cpu.py \
    --resume checkpoints/rl_cpu_model.last.pt \
    --out checkpoints/rl_cpu_model.pt \
    --iterations 200 --games-per-iter 16 --num-workers 8 \
    --ppo-batch-size 64 --lr 1e-4 --snapshot-every 5 \
    --tb-logdir runs/rl_cpu_big
```

`--iterations` is an absolute target, not "N more from now."

## Game mechanics — quick reference

Verified against `kaggle_environments/envs/orbit_wars/README.md` and empirically.

**World**: 100×100 continuous board; sun at (50,50), radius 10. 500 turns (`episodeSteps`), 2 or 4 players. 20–40 planets in 4-fold mirror symmetry. **Score = total ships (planets + in-flight) at game end. Highest wins.**

**Planets** `[id, owner, x, y, radius, ships, production]`:
- owner: 0–3 or −1 (neutral); home planets start with 10 ships
- production 1–5, `radius = 1 + ln(production)`
- Orbiting iff `dist_to_sun + radius < 50`; rotates CCW at `ω` rad/turn (0.025–0.05). Phase: `phase(step) = phase₀ + (step−1)·ω`, `phase₀` from `obs.initial_planets`.

**Fleets** `[id, owner, x, y, angle, from_planet_id, ships]`:
- Straight line, constant angle. `speed = 1 + 5·(log(ships)/log(1000))^1.5`. 1 ship → 1/turn, 1000 → 6/turn (max).
- Die on board exit, sun crossing, or planet clip. Spawn at `radius + 0.1` outside source.

**Combat** — attrition:
- Same owner → add to garrison
- Diff owner: `attackers ≥ garrison` → flips, new garrison = `attackers − garrison`. `attackers < garrison` → attackers destroyed AND garrison reduced by attackers' count (multi-wave attrition).
- Same-turn multi-attacker edge cases under-specified; harness resolves by ETA.

**Comets** — spawn at turns 50/150/250/350/450 (4 per spawn). Production 1, radius 1, speed 4. Start ships = min of four 1–99 rolls. Leave board after path. Identified via `obs.comet_planet_ids`.

**Turn order**: expire comets → spawn new → launches → produce → fleet move + collision → rotate orbiters/comets → resolve combat.

**Agent I/O**:
- Obs: `{step, player, planets, fleets, angular_velocity, initial_planets, comets, comet_planet_ids, next_fleet_id, remainingOverageTime}`
- Action: `[[from_planet_id, angle_rad, num_ships], …]`
- At step 0 `planets` is empty — must `env.step([[], []])` once.

## `radar.py` — authoritative trajectory simulator

Env-faithful per-turn march. Validates board bounds → sun crossing → planet collision → moving-planet sweep. **99.25% match** vs real env in prior audit. Used by `targeting.threats_per_planet` and `GameView.action_mask`.

API: `Radar(obs)`, `simulate_fleet(fleet)`, `simulate_launch(src, angle, ships)`, `launch_position(src, angle)`. `RadarHit.kind ∈ {hit_planet, swept_planet, sun, board, timeout}`.

## `targeting.py` — primitives

Pure functions: `fleet_speed`, `is_orbiting`, `orbit_params`, `future_position`, `lead_intercept` (fixed-point on `(angle, eta, pred_x, pred_y)`), `required_ships_to_capture`, `threats_per_planet`, `ring_order`. Mirrored in `server.py`'s JS for instant hover preview.

## Top-10% replay datasets (Bovard, ongoing)

Kaggle staff publishes daily top-10% replay datasets at `bovard/orbit-wars-top10-episodes-{YYYY-MM-DD}` (CC0). As of 2026-05-19 the datasets are **broken**: 3,536 of 37,760 expected episodes actually present; only 04-16/17/18 and 05-04 fully populated; all available replays are 4-player. Bovard said fix incoming. Re-check before relying.

If usable: standard Kaggle Environments JSON. Stack is 2P-only, so SFT requires either 4P harness adaptation or 4P→2P projection.

## Conventions

- **Tech decisions**: user said "take tech decisions yourself" — don't ask for trivia, do check before structural changes.
- **Venv only**: `.venv/bin/python` for everything.
- **Don't edit** `getting-started.ipynb` — Kaggle tutorial, truncated.
- **Edit before create**: new files only when the split is genuinely new (a new module, not a reshuffle).
- **Memory** at `~/.claude/projects/-Users-martian-Documents-Code-orbitWars/memory/` holds durable context. Update when facts change.

## Architecture principle (load-bearing)

*If the harness can compute X, don't make the NN learn it.* Angles, lead-intercept on orbiters, sun/planet avoidance, ship sizing — all deterministic, all in the harness. The model picks an edge; the harness handles physics. This is what made the NumPy submission fit in 1s on Kaggle CPU and also caps how much PPO can improve over heuristic (ship-count is harness-pinned). Both true.
