# Orbit Wars — Kaggle agent

Goal: build a competitive agent for [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars/overview). Repo also contains an interactive human-vs-agent UI for playtesting and strategy development.

## Environment & how to run

**Always use the venv.** Never install into global Python.

```bash
.venv/bin/python server.py          # launches UI at http://127.0.0.1:5000
.venv/bin/python -c "…"              # any one-off script
.venv/bin/pip install <pkg>          # if adding deps
```

The venv has `kaggle-environments>=1.28.0` and `flask`. Python 3.13.

On import, `kaggle-environments` emits a benign `Loading environment cabt failed: dlopen(...)` error plus `OpenSpiel environments` info — ignore both; unrelated to Orbit Wars.

## Repo layout

```
orbitWars/
├── getting-started.ipynb   # Kaggle tutorial (truncated at 50MB — don't try to fully parse)
├── orbit_wars/
│   ├── core/               # shared action_space, radar, targeting
│   ├── cpu/                # active dynamic-edge CPU stack
│   ├── legacy/             # old padded 50×50 stack + old RL scaffolding
│   └── apps/server.py      # Flask UI implementation
├── *_cpu.py, harness.py, model.py, agents.py, ...  # thin root compatibility wrappers
├── make_cpu_submission.py  # Builds pure-NumPy dynamic-edge Kaggle submission
├── make_nn_submission.py   # Legacy NN submission builder
├── make_submission.py      # Heuristic submission builder
├── submission_cpu.py       # Generated CPU dynamic-edge NumPy submission artifact
├── checkpoints/bc_baseline.pt    # first BC model (legacy padded stack)
├── checkpoints/bc_cpu_model.pt   # active CPU BC baseline checkpoint
├── checkpoints/rl_cpu_model.pt   # active CPU RL final checkpoint (PPO from BC)
├── checkpoints/rl_cpu_model.last.pt  # resume state: model + optimizer + opp-pool + iter
├── data/bc, data/bc_worker_{1,2,3}      # BC shards (legacy padded stack)
├── data/bc_cpu, data/bc_cpu_worker_{1..4} # active CPU BC shards
├── markdowns/              # ad-hoc design / planning notes
├── logs/                   # nohup'd training run output (gitignored)
├── runs/                   # TensorBoard event files (gitignored)
├── LONG_TERM_FIXES.md      # deferred correctness items (e.g. same-turn combat grouping)
└── .venv/                  # kaggle-environments + flask + torch
```

## Current active direction — CPU dynamic-edge stack

As of 2026-04-25, the active deployable path is the parallel `_cpu.py` stack,
not the old padded 50×50 model. The old stack stays in the repo for reference
and A/B comparison, but new BC/RL/submission work should target:

```text
GameView_CPU → OrbitWarsEdgeTransformer → bc_data_cpu.py / bc_train_cpu.py
→ rl_train_cpu.py → make_cpu_submission.py → submission_cpu.py
```

The implementation now lives under `orbit_wars/`; root `*.py` files are
compatibility wrappers so existing commands and Kaggle bundling scripts keep
working.

What is done:
- `harness_cpu.py` emits one token per radar-valid targetable edge rather than
  a padded 50×50 grid. Comets are dropped in v1.
- `model_cpu.py` trains in PyTorch on CPU/MPS/CUDA but skips the value head at
  inference. GELU is `approximate="tanh"` so training matches the NumPy
  submission exactly (no logit drift).
- `agents_cpu.py` contains the CPU-token heuristic teacher, `StatefulCpuModelAgent`,
  and checkpoint loader for UI/eval.
- `bc_data_cpu.py` captured a new CPU-format dataset from scratch; do not
  convert or reuse old padded BC shards unless explicitly asked.
- `bc_train_cpu.py` trained `checkpoints/bc_cpu_model.pt` from 19,545 examples
  across 40 shards.
- `make_cpu_submission.py` generated `submission_cpu.py`, a pure-NumPy
  submission with embedded weights and no `torch` import. `axial_threshold` is
  now read from the loaded model and stamped into the runtime, so future
  overrides flow through correctly.
- `rl_train_cpu.py` is the active PPO entrypoint. Implementation lives in
  `orbit_wars/cpu/rl_*.py` and uses ragged CPU token rollouts with padded PPO
  minibatches. Supports `--num-workers N` for parallel rollout via
  `ProcessPoolExecutor` (spawn context, `torch.set_num_threads(1)` per worker).

Latest CPU BC checkpoint:
- `checkpoints/bc_cpu_model.pt` is epoch 10, best validation checkpoint.
- Validation: total accuracy 0.959, move accuracy 0.941, stop accuracy 1.0.
- Eval: 4/4 wins vs `nearest_planet_sniper`, 2/4 vs `heuristic_cpu`.

Latest CPU RL checkpoint:
- `checkpoints/rl_cpu_model.pt` — first RL training pass complete. 60-iter
  PPO from BC baseline; LR ramped 3e-5 → 1e-4 mid-run because `approx_kl`
  stayed in 0.001–0.005 (policy barely moved). After ramp, KL crept to 0.005–0.022
  band, entropy held 0.07–0.17, no early-stop / collapse. Win-rate-vs-heuristic
  noisy but trended upward; needs proper N-game eval to confirm vs BC baseline.
- `checkpoints/rl_cpu_model.last.pt` — resume state including optimizer,
  opponent-pool snapshots, and iteration counter.

Deployment status:
- `submission_cpu.py` has been generated with `max_moves=2`.
- The CLI upload succeeded with message `CPU dynamic-edge BC max_moves_2`;
  the status was `PENDING` when last checked.
- Local generated-submission latency with `max_moves=2`: mean 48 ms, p95 158 ms,
  max 215 ms. This is plausible but risky on Kaggle because past submissions
  saw roughly 5×–8× slowdown.
- Local generated-submission latency with `max_moves=1`: mean 24.5 ms, p95 63.8 ms,
  max 75.7 ms. This is much safer.
- `max_moves=3` is not viable without further optimization: local p95 was
  ~1529 ms.

Next high-value work:
1. Check Kaggle result for `submission_cpu.py`.
2. N-game eval of `rl_cpu_model.pt` vs `heuristic_agent_cpu` (and vs
   `bc_cpu_model.pt`) to confirm RL improvement is real, not noise.
3. Bigger-batch RL: `--games-per-iter 16+` per update — current 4-game
   batches are noise-dominated. Real win comes from variance reduction.
4. Generate fresh `submission_cpu.py` from the trained RL checkpoint and
   resubmit when results look meaningfully better than BC.
5. If `max_moves=2` times out on Kaggle, submit/regenerate `max_moves=1`.
6. If higher move count is needed, optimize `GameView_CPU.tokens()`
   rebuilding before increasing `max_moves`.

## Inaccuracies corrected / stale guidance

- `rl_rollout.py`, `rl_ppo.py`, and `rl_train.py` are legacy padded-stack
  wrappers. Use `rl_train_cpu.py` for the active CPU model.
- The old statement "next bottleneck is model-side, not harness-side" applies
  to `harness.py` + `model.py`; it does **not** apply to `harness_cpu.py`.
  In the CPU stack, NumPy forward is cheap and token construction/rebuilding is
  the bottleneck when `max_moves > 1`.
- `GameView_CPU` has cross-turn fleet-prediction reuse, but not token-level
  incremental caching. `tokens()` rebuilds the full dynamic token list after
  `update_from_obs()` and after every `apply_planned_move()`.
- The active CPU feature layout is 11 features, but it is **not** the same 11
  as old `harness.py`: old `tgt_expiry` was dropped and `src_can_fund` was
  added.
- The active teacher is `heuristic_agent_cpu` / `choose_heuristic_token_cpu`,
  not `agents.heuristic_agent`, for CPU-format data.
- GAE bootstrap fix: `rl_rollout.py:239` previously zeroed the value bootstrap
  unconditionally; now gated on `env.done` so `--max-turns` truncation
  correctly leaves `done=False` and bootstraps with `V(s_T)`. Default
  `--max-turns 500` rarely fires truncation, but smoke runs with low
  `--max-turns` were biased before this fix.

## Game mechanics — quick reference

All verified against `kaggle_environments/envs/orbit_wars/README.md` and empirically.

**World**
- 100 × 100 continuous board; sun at (50, 50), radius 10; destroys any fleet whose path clips it
- 500 turns (`episodeSteps`), 2 or 4 players
- 20–40 planets in 4-fold mirror symmetry around the center
- **Score = total ships (planets + in-flight fleets)** at game end. Highest wins.

**Planets** `[id, owner, x, y, radius, ships, production]`
- owner: 0–3 or −1 (neutral); home planets start with 10 ships
- production 1–5 ships/turn, `radius = 1 + ln(production)`
- **Orbiting** iff `distance_to_sun + radius < 50`; rotates CCW at `angular_velocity` rad/turn (0.025–0.05)
- Rotation formula (verified): `phase(step) = phase₀ + (step − 1) × ω`, where `phase₀` comes from `obs.initial_planets` (captured at step 1)
- **Static** otherwise; doesn't move

**Fleets** `[id, owner, x, y, angle, from_planet_id, ships]`
- Straight-line, constant angle; `speed = 1 + 5 × (log(ships) / log(1000))^1.5`
  - 1 ship → 1/turn; 500 → ~5/turn; 1000 → max 6/turn
- Die if they exit the board, cross the sun, or clip any planet (continuous check)
- Spawn just outside source-planet radius in the given direction
- Can only launch from owned planets; ships ≤ current garrison

**Combat** — attrition model (not binary)
- Each arriving fleet resolves against the current garrison.
- Same-owner arrivals add to garrison.
- Different-owner arrivals:
  - `attackers ≥ garrison` → planet flips, new garrison = `attackers − garrison`
  - `attackers < garrison` → attackers destroyed **and garrison is reduced by the attackers' count** (production resumes from the weakened garrison on subsequent turns). This is what enables multi-wave attacks: small sequential strikes chip the garrison down so a later fleet can capture.
- Same-turn multi-attacker edge cases (two different owners landing on the same neutral/enemy the same turn) are not well-characterized in this doc; the harness resolves arrivals sequentially by eta. If you rely on same-turn grouping rules, verify empirically before trusting.

**Comets** — temporary "planets" on elliptical paths
- Spawn at turns 50, 150, 250, 350, 450 (4 per spawn, one per quadrant)
- production 1, radius 1, speed 4/turn
- Starting ships = `min` of four 1–99 rolls (usually low → cheap to capture)
- Identified via `obs.comet_planet_ids`; orbit data in `obs.comets`
- Leave the board after their path; garrison lost when they exit

**Turn order** (so you can reason about timing)
1. Remove expired comets
2. Spawn new comet groups
3. Process fleet launches
4. Produce (all owned planets/comets)
5. Move fleets, check collisions
6. Rotate orbiters, advance comets
7. Resolve combat

**Agent I/O**
- Observation: `{ step, player, planets, fleets, angular_velocity, initial_planets, comets, comet_planet_ids, next_fleet_id, remainingOverageTime }`
- Action: `[[from_planet_id, angle_rad, num_ships], …]`
- `actTimeout: 1` second per turn + `remainingOverageTime: 60` seconds shared across the game

**Gotcha**: at step 0 `planets` is empty. Must `env.step([[], []])` once to populate the world. The UI does this in `new_env()`.

## `radar.py` — authoritative trajectory simulator

Env-faithful per-turn march for fleets and candidate launches. Mirrors the environment's internal checks: board bounds, sun-segment crossing, planet-segment collision, then the moving-planet sweep after rotation.

- `Radar(obs, horizon=500)` — one instance per (obs snapshot). Position cache keyed by `(planet_id, t_ahead)` reused across all simulate calls.
- `simulate_fleet(fleet)` → `RadarHit` — predicts where an existing in-flight fleet lands.
- `simulate_launch(src_planet, angle, ships)` → `RadarHit` — predicts where a prospective launch lands, using the env's spawn rule `src_center + direction * (radius + 0.1)`.
- `launch_position(src_planet, angle)` — returns the env-accurate spawn point.
- `RadarHit.kind` ∈ `{"hit_planet", "swept_planet", "sun", "board", "timeout"}`; `hit_planet` is True for the two planet-hit cases.

Validation: 99.25% match vs the real env in prior audit. Used by both `targeting.threats_per_planet` and `GameView.action_mask` — those two are now the only radar call sites.

## `targeting.py` — primitives

Pure functions, no env state. Shared by the UI (imported in `server.py`, also mirrored in JS) and by the harness.

- `fleet_speed(ships)` — speed-per-turn formula
- `is_orbiting(initial_planet)` — boolean from initial position
- `orbit_params(ip)` — `(orbital_radius, phase₀)`
- `future_position(ip, current_step, t_ahead, ω)` — position at `step + t_ahead`
- `lead_intercept(src_xy, tgt_ip, ships, ω, step)` — fixed-point solves `(angle, eta, pred_x, pred_y)`. Returns `None` if non-convergent.
- `required_ships_to_capture(target, eta)` → `target.ships + prod × eta + 1`
- `predict_fleet_landing(...)` — legacy integer-turn sim; **kept but superseded by `Radar.simulate_fleet`**.
- `threats_per_planet(...)` → `{pid: [{eta, owner, ships, fleet_id}, …]}`. Now uses `Radar.simulate_fleet` internally.
- `ring_order(planets, initial_planets, include_orbiting=False)` — planet ids sorted CCW around the sun

## `server.py` — UI & harness for human play

Flask app, single-file, embedded HTML/JS. Launches at http://127.0.0.1:5000.

**Endpoints**
- `GET /` — the canvas game page
- `GET /state` — full game state enriched with `threats`, `ring`, `initial_planets`
- `POST /action` `{moves: [[pid,angle,ships], …]}` — applies human moves + runs agent, steps env once, returns new state
- `POST /reset` — new game

**Client-side**
- JS ports of `fleet_speed`, `future_position`, `lead_intercept` so hover-preview is instant (no server round-trip per mouse-move)
- Threat badges on each planet (`-N in Xt` red / `+N in Xt` green)
- Outer-ring table on the right panel
- Aim-mode toggle: **Lead** (uses lead-intercept angle) / **Raw** (atan2 to current position)
- Ship presets: 25% / 50% / 100% of source garrison
- **Realtime mode**: enforces the agent's 1 s per-turn + 60 s overage budget on the human. Timer keeps running across clicks; if it expires mid-click-sequence, the selected-source persists into the next turn so the target click still registers.

**Configuration knobs** at the top of `server.py`
- `HUMAN = 0`, `AGENT = 1` — player seats
- default agent is `heuristic_agent`
- To play against the trained CPU BC model:

```bash
ORBIT_AGENT=cpu_model \
ORBIT_CPU_CHECKPOINT=checkpoints/bc_cpu_model.pt \
ORBIT_CPU_DEVICE=cpu \
.venv/bin/python server.py
```

The UI labels the loaded agent in `/state` as `agent_name`.

## Legacy padded stack: `harness.py` / `model.py` / `agents.py`

This stack still exists and is useful as a reference implementation, but it is
not the active deployable path for Kaggle CPU inference.

### `harness.py` — GameView & incremental state

`GameView(obs)` builds `edge_features (50, 50, 11)`, `legal_mask (50, 50)`, and `planet_ids (50,)` from a single observation. The legal_mask covers sun-crossing only and is a cheap pre-filter; the **authoritative mask** for model actions is `view.action_mask(safety_margin)`, which radar-validates every legal edge under deterministic ship sizing.

**Deterministic ship sizing** lives on the view (no ship buckets):
- Attack: `ships_needed + safety_margin`
- Defense: `deficit_at_first_flip_eta + safety_margin`

### Incremental updates (key for latency)

GameView supports two kinds of in-place mutation so we don't rebuild from scratch:

1. **Sub-move within a turn** — `view.apply_planned_move(src_slot, tgt_slot, ships)`:
   - Decrements src garrison, appends planned fleet, inserts radar-predicted landing into `threats`.
   - Rebuilds only the 4 affected strips in `edge_features` + `legal_mask`: rows and columns of both `src_slot` and `tgt_slot`.
   - Patches the cached `action_mask` in place for the same strips.
   - The Radar instance is lazy-cached on the view and shared across sub-moves (planets don't rotate mid-turn).

2. **Turn-to-turn** — `view.update_from_obs(new_obs)`:
   - Diffs fleets by `fleet_id`: carry-overs keep their cached radar predictions with `eta -= 1`; new fleets get one `simulate_fleet` call; departed fleets drop.
   - Falls back to full rebuild if the planet set changes (comet spawn/expiry on turns 50/150/250/350/450).
   - Invalidates the cached action_mask (positions moved → radar legality shifts).

3. **Debug self-check** — `view.assert_equals_fresh_rebuild()` verifies the incremental state matches a cold rebuild from the same internal state. Use in tests; do not call in the hot path.

**`StatefulModelAgent` in `agents.py`** wraps this for inference: holds one `GameView` across turns and calls `update_from_obs` instead of re-constructing.

### Latency (per-turn, 100-turn heuristic-vs-heuristic game, M2 CPU)

Measured by `bench.py`. `actTimeout` is 1 s/turn + 60 s total overage — both paths below fit comfortably.

| Component | cost |
|---|---|
| `GameView.__init__` (cold) | ~12 ms |
| `GameView.update_from_obs` (warm) | ~3 ms  *(75.7% cheaper than cold)* |
| `action_mask` first call (cold or warm view) | ~97 ms |
| `action_mask` sub-move patch | ~10 ms |
| `model.forward` | ~95 ms CPU / ~60 ms MPS |
| `heuristic_agent` | ~12 ms |
| `model_agent_actions` (cold every turn) | mean ~400 ms, p99 ~650 ms |
| `StatefulModelAgent` (warm, uses `update_from_obs`) | mean ~396 ms, p99 ~880 ms |

**Where the time goes per turn**: the dominant cost is now `model.forward × ~3 sub-moves ≈ 285 ms`. State-build is only ~3% of per-turn cost — so the cross-turn warm path saves real wall time on `update_from_obs` (12 → 3 ms) but barely moves the end-to-end number. The higher warm p99 is a tail artifact: turns 50/150/… fall back to a full rebuild when comets spawn and the planet set changes.

**Next bottleneck is model-side, not harness-side.** Pursue batched/masked attention, smaller model, or cap `MAX_MODEL_MOVES = 2` at inference before further harness optimization.

This last bottleneck note is **legacy-only**. In the CPU dynamic stack,
`GameView_CPU.tokens()` rebuilding is the bottleneck once `max_moves > 1`.

### `model.py` — OrbitWarsTransformer

- Input: `(B, 50, 50, 11)` edge features, normalized by `FEATURE_SCALES` at input.
- 3 × `TransformerEncoderLayer`, d_model=64, nhead=4, GELU.
- **Two heads + stop**:
  - Policy: scalar per edge → masked softmax over `50*50 + 1 = 2501` slots (last slot = stop).
- Value: mean-pool → MLP → scalar.
- ~208K params. Legality-masked logits: illegal edges get `-1e9` before softmax.

### `agents.py` — agents & decoders

- `nearest_planet_sniper(obs)` — Kaggle tutorial baseline. Emits `(src, angle, ships)` directly; kept for regression comparison. **Not** used as a BC teacher (see memory).
- `heuristic_agent(obs, max_moves=MAX_MODEL_MOVES)` — rule-based teacher: defend falling planets → ROI-ranked expansion → hoard when `turns_left < 20`. Capped to the model's 3-move turn shape and uses `action_mask`, not just `legal_mask`, so BC labels stay inside the radar-validated action space. 10/10 vs sniper before the cap; quick post-cap regression was 3/3.
- `random_model_space_agent(obs)` — uniform random over `action_mask`; smoke-tests the pipeline.
- `model_agent_actions(model, obs)` — decode a model's policy into up to `MAX_MODEL_MOVES` actions. Uses `apply_planned_move` for sub-moves. Re-builds GameView each turn.
- `StatefulModelAgent(model)` — callable wrapper that holds one `GameView` across turns via `update_from_obs`. Use for live self-play / UI when per-turn latency matters.

## Active CPU dynamic stack: `harness_cpu.py` / `model_cpu.py`

### `harness_cpu.py` — GameView_CPU & TokenBundle

`GameView_CPU(obs)` emits a variable-length `TokenBundle` containing only
targetable, radar-valid action edges:

```text
edges      (N, 11) float32
src_ids    (N,) int32 slot ids
tgt_ids    (N,) int32 slot ids
ships      (N,) int32 deterministic ship count for this edge
angles     (N,) float32 lead-intercept angle
planet_ids (P,) int32 slot → env planet_id
```

Comets are excluded in v1: planets in `obs.comet_planet_ids` are filtered out.
This simplifies token shape and avoids comet-expiry bookkeeping; it may cap
eventual strength.

Feature order:

```text
eta, ships_needed,
kind_reinforce, kind_attack_enemy, kind_attack_neutral,
src_ships, src_net_threat,
tgt_production, tgt_will_fall,
src_can_fund,
turns_left
```

`src_can_fund` is important: the model can see underfunded targetable attacks
while also knowing whether the source has enough ships to execute the
harness-computed capture/defense amount. It is computed before the final
`ships = min(src_available, need)` clamp.

Ship sizing is deterministic:
- Attacks send `ships_needed + safety_margin`, clamped to source availability.
- Reinforcements send the projected defense deficit plus safety margin, or 1 if
  the friendly target is not projected to fall.

ETA/ship consistency: the CPU harness now iterates ship count and intercept so
the token's `eta`, `ships_needed`, and deterministic `ships` agree better.
This matters because fleet speed depends on ships.

Incremental status:
- Cross-turn `update_from_obs()` reuses carry-over fleet predictions where
  possible (`eta -= steps_advanced`).
- Token-level incremental updates are **not implemented**. Any `tokens()` call
  after `update_from_obs()` or `apply_planned_move()` rebuilds the full token
  list.
- This is the main latency bottleneck for `max_moves > 1`.

### `model_cpu.py` — OrbitWarsEdgeTransformer

- Input: packed edge tokens `(B, N, 11)` plus `src_ids`, `tgt_ids`, and optional
  `valid_mask`.
- Default architecture: `d_model=32`, `d_ff=64`, 2 encoder blocks, single-head
  fused Q/K/V attention. GELU is `approximate="tanh"` everywhere (FeedForward
  + ValueHead) to match the NumPy submission's `_gelu` exactly.
- Attention is threshold-gated by `axial_threshold` (default `DEFAULT_AXIAL_THRESHOLD = 256`):
  - `N <= axial_threshold`: full self-attention
  - `N > axial_threshold`: layer 1 attends within same source, layer 2 within same target
  - The threshold is read from the loaded model when generating
    `submission_cpu.py` and stamped into the runtime — overrides flow through.
- Stop action is logit index `N` for each sample.
- Value head is larger on purpose for future PPO, but `compute_value=False`
  skips it during inference.
- Parameter count: 46,723 total; 17,634 inference parameters excluding value head.

### `agents_cpu.py`

- `choose_heuristic_token_cpu(view)` scores tokens directly in CPU action space.
- `heuristic_agent_cpu(obs)` is the CPU-format teacher for BC capture.
- `model_agent_actions_cpu(...)` decodes a trained `OrbitWarsEdgeTransformer`.
- `StatefulCpuModelAgent` holds a `GameView_CPU` across turns.
- `load_cpu_model_agent(checkpoint)` loads `checkpoints/bc_cpu_model.pt` for
  eval or UI play.

## `bc_data.py` — behavior-cloning capture

Run heuristic-vs-heuristic games and write compressed `.npz` shards:

```bash
.venv/bin/python bc_data.py --games 10 --out data/bc
.venv/bin/python bc_data.py --games 1 --max-turns 60 --out /tmp/bc_smoke
```

Each example contains:
- `edge_features`: `(50, 50, 11)` float32
- `legal_mask`: `(50, 50)` bool cheap candidate mask
- `action_mask`: `(50, 50)` bool authoritative radar mask
- `planet_ids`: `(50,)` int32 slot map
- `action_idx`: int64, `src_slot * 50 + tgt_slot` or stop index `2500`
- metadata: game, step, player, submove, src/tgt slots, ships

Capture is sequential like inference: ask the heuristic for one move, save the
pre-move tensors/masks, apply the planned move to the `GameView`, then ask for
the next move. If the teacher stops before `MAX_MODEL_MOVES`, record a stop
example. Smoke validation through 60 turns produced 180 examples with 0 labels
outside `action_mask`.

## `bc_train.py` — behavior-cloning trainer

Plain PyTorch cross-entropy trainer over the flattened `50*50 + stop` action
space. It streams compressed shards one at a time, so the full BC corpus does
not need to fit in memory.

```bash
.venv/bin/python bc_train.py --data data/bc data/bc_worker_* --out checkpoints/bc_model.pt
.venv/bin/python bc_train.py --data data/bc data/bc_worker_* --out checkpoints/bc_model.pt --resume checkpoints/bc_model.last.pt --epochs 10
.venv/bin/python bc_train.py --data /tmp/bc_smoke --out /tmp/bc_model.pt --epochs 1 --device cpu
```

Default behavior:
- accepts one or more shard files/directories; directories are searched recursively, so parallel capture workers can write to separate directories
- auto-selects MPS if available, otherwise CPU
- splits shards into train/val by shard
- trains `OrbitWarsTransformer` using `action_logits` masked by `action_mask`
- logs total accuracy plus separate move/stop accuracy
- writes best checkpoint, last checkpoint, and `.history.json`
- can resume with `--resume`; optimizer tensors are moved to the selected device, so CPU checkpoints can resume on MPS
- downweights stop labels with `--stop-weight 0.5` by default because BC data
  includes many stop examples and an unweighted model can learn to stop too
  eagerly early in training
- pinned memory + `non_blocking=True` H2D transfers; `mps.synchronize()` only
  at epoch boundaries (per-step sync stalled the async pipeline and roughly
  halved throughput — do not reintroduce it)
- default `--batch-size 8`; B=16 OOMs on 8GB MPS because the attention matrix
  for 2500 tokens is `B × heads × 2500²` (≈1.6 GB/layer at B=16)

### Throughput notes (M2 MPS, 8GB)

BC training on MPS is bandwidth-limited by attention over 2500 tokens, not by
H2D or data loading. Observed ~1.7–1.8 ex/s at B=8 after pipeline fixes
(pinned memory, async transfers, no per-step sync). Full corpus × 30 epochs is
on the order of tens of hours. CUDA migration is the planned escape; do not
chase further MPS micro-optimizations unless the user asks.

## CPU PPO self-play

The active RL stack lives in `orbit_wars/cpu/rl_*.py` with root wrappers
`rl_train_cpu.py`, `rl_rollout_cpu.py`, `rl_ppo_cpu.py`, and
`rl_opponent_pool_cpu.py`. The old `rl_*.py` wrappers still point at the
legacy padded stack.

- **Rollout** records ragged CPU token states: `edges (N, 11)`, `src_ids`,
  `tgt_ids`, `n_tokens`, `action_idx`, old log-prob, value, and shaped reward.
  Stop is per-record index `N`.
- **PPO batching** pads each minibatch to `N_max` and remaps stop labels from
  per-record `n_tokens` to the shared padded stop index `N_max`, matching
  `bc_train_cpu.collate_cpu`.
- **Opponent pool** starts at 100% `heuristic_agent_cpu`. Once snapshots exist,
  default weights are `heuristic_weight=0.5` and `snapshot_weight=1.0`, so the
  mix is 33% heuristic and 67% past snapshots. Snapshots are FIFO-evicted at
  `--max-snapshots` (default 8). PFSP-style weighted sampling is **not**
  implemented yet — defer until snapshots have meaningfully different skill
  levels (i.e. `approx_kl` consistently above 0.01 across iters).
- **Resume** writes `checkpoints/rl_cpu_model.last.pt` every iteration with
  model, optimizer, opponent-pool state, args, and iteration index. Final clean
  completion writes `checkpoints/rl_cpu_model.pt`.

### Parallel rollout (`--num-workers N`)

Each iteration plays `--games-per-iter` games. With `--num-workers 1` they run
sequentially. With `N > 1`, a `ProcessPoolExecutor` (spawn context, persistent
across iterations) hands games out to workers as they free up.

- Each worker calls `play_one_game_worker` from `orbit_wars/cpu/rl_rollout.py`,
  which sets `torch.set_num_threads(1)` and forces CPU regardless of the main
  device — avoids BLAS oversubscription when N workers run in parallel.
- Each iteration ships the current `model.state_dict()` and the full
  `opponent_pool.state_dict()` to every task. Cheap (~46k params).
- Per-worker seed is `args.seed + iteration*100_003 + game_i`, so games are
  reproducible per (iter, game_i) and not duplicated across workers.
- Wall clock per iter ≈ `sum(per-game time) / num_workers` plus straggler
  variance. Workers do NOT idle waiting for siblings — `executor.map` hands
  out the next task immediately on completion. But when `games_per_iter`
  ≈ `num_workers`, variance can leave workers idle ~20-25% (long-game
  bottleneck). Mitigation: use `games_per_iter ≥ 2× num_workers` *or* use
  `--async-rollout` (below).

### Async rollout (`--async-rollout`)

Workers run continuously and never block on siblings. The learner consumes a
shared queue and runs PPO whenever `--games-per-iter` finished trajectories
have arrived. After each update, the learner atomically rewrites a shared
weights file (`<out>.weights.pt`); workers `mtime`-poll it at game start and
reload model + opponent-pool state when it changes.

Implementation lives in `orbit_wars/cpu/rl_async.py`. Sync mode is unchanged
and still the default. Both modes are drop-in resumable from the same
`*.last.pt` (the on-disk checkpoint format is identical).

When to use it: variable game length is the giveaway — if some games end at
~50 turns and others at the `--max-turns` cap, sync workers idle waiting for
the longest game in each batch. Async absorbs the variance because the next
game starts the moment any one finishes.

Mechanics worth knowing:
- "Iteration" still means *one PPO update*. `--snapshot-every`, `--iterations`,
  TB step indices, and the resume counter all use this. Resuming a sync
  `last.pt` into async mode just works (and vice versa).
- Determinism: per-worker seed is `seed + worker_id*1_000_003 + worker_local_game_i`.
  Each worker's stream is reproducible, but global trajectory ordering is
  intentionally not — that's the whole point. `--deterministic-rollout`
  (greedy argmax) still works.
- Staleness is logged as `async/staleness_{mean,p99,max}` (current iteration
  minus the generation a trajectory was started under). PPO's clip ratio
  handles small staleness fine while `approx_kl` stays under ~0.02; bail to
  sync if you ever see staleness consistently >2 with KL trending up.
- Opponent pool is owned by the learner. Snapshot adds happen post-PPO, then
  get bundled into the next `weights.pt` write — workers see the new pool on
  their next reload. Workers never call `add_snapshot`.
- Shutdown is best-effort: workers may be mid-game when the learner exits.
  The driver drains the queue while joining (60s deadline) and `terminate()`s
  stragglers. `weights.pt` is unlinked on clean exit; if a SIGKILL/SIGTERM
  leaves it behind, it's harmless — just delete it.

A/B vs sync (M2 4P+4E, 4 workers, 4 games/update, max-turns 200, same seed,
both starting from `bc_cpu_model.pt`, apples-to-apples 15 updates):

| | sync | async | delta |
|---|---|---|---|
| wall clock (15 updates) | 4480s (74.7 min) | 1909s (31.8 min) | **2.35× speedup, −57%** |
| mean rollout_s | 283 | 99 | sync waits on the slowest of 4 games every iter |
| mean update_s | 15 | 29 | async batches average more submoves (workers run continuously) |
| approx_kl mean | 0.0028 | 0.0071 | async sees mildly off-policy data; both ≪ target_kl 0.03 |
| clip_frac mean | 0.018 | 0.026 | same root cause; small enough that PPO doesn't waste samples |
| entropy mean | 0.096 | 0.110 | comparable; no collapse either side |
| mean_margin (mean over iters) | −0.080 | +0.101 | within noise at n=15, 4 games/iter |
| win_rate_vs_heuristic (mean) | 0.467 | 0.494 | within noise |

Speedup is much larger than the original 15–20% estimate — game length at
`max-turns=200` varies wider than I assumed (140–520 s/iter sync rollout),
so the sync "wait for the longest of 4 games" tax is large.

Staleness distribution observed (15 async iters):
- per-iter `stale_mean` averaged 1.08 (so a trajectory is on average 1 update
  old when consumed), worst-iter mean 1.50.
- per-iter `stale_p99` averaged 1.72, worst 3.91.
- `stale_max` ever observed: 4. With per-iter `approx_kl ≈ 0.007`, a 4-update
  stale trajectory is ~0.028 KL off-policy — still inside the clip band.

If `approx_kl` ever climbs to ≥0.05 per update, revisit: 3-update stale data
would be ~0.15 KL off, far outside clip, and async would start wasting
samples. At that point either lower lr, raise `--snapshot-every`, or fall
back to sync.

Smoke-tested command:

```bash
.venv/bin/python rl_train_cpu.py \
    --checkpoint checkpoints/bc_cpu_model.pt \
    --out /tmp/rl_async_smoke.pt \
    --iterations 1 --games-per-iter 4 --max-turns 50 \
    --device cpu --ppo-epochs 1 --ppo-batch-size 8 \
    --num-workers 4 --async-rollout
```

### Hardware sizing

- M2 (8 cores, 4P+4E): `--num-workers 4` is the sweet spot. More workers land
  on E-cores (~40% the throughput of P-cores) and don't help. Measured ~1.6×
  speedup vs serial.
- Linux x86 (all-equal cores): use `--num-workers = nproc`. No P/E asymmetry.
  Bottleneck shifts to memory bandwidth at high core counts.
- GPU: not currently exercised. The 46k-param model is too small to benefit
  from GPU rollout. Reserve GPU for batched learner-side PPO updates if
  scaling beyond ~32 vCPUs.

Smoke-tested command:

```bash
.venv/bin/python rl_train_cpu.py \
    --checkpoint checkpoints/bc_cpu_model.pt \
    --out /tmp/rl_cpu_smoke.pt \
    --iterations 1 --games-per-iter 4 --max-turns 50 \
    --device cpu --ppo-epochs 1 --ppo-batch-size 8 \
    --num-workers 4
```

Suggested real run (resume + bigger batches):

```bash
.venv/bin/python rl_train_cpu.py \
    --resume checkpoints/rl_cpu_model.last.pt \
    --out checkpoints/rl_cpu_model.pt \
    --iterations 200 --games-per-iter 16 --num-workers 8 \
    --ppo-batch-size 64 --lr 1e-4 --snapshot-every 5 \
    --tb-logdir runs/rl_cpu_big
```

Resume gotcha: when iter `N` completes, `last.pt` is saved with `iteration=N`,
so resume returns `start_iteration = N+1`. Pass `--iterations` as the
absolute target count, not "N more iterations from now."

Device note: `--device auto` uses CUDA when available, otherwise CPU. MPS is
opt-in because measured short PPO smokes were slower on MPS than CPU on M2
(`2 × 80` capped turns: CPU rollout/update `42.8s / 0.4s`; MPS `50.2s / 8.1s`).

Watch `win_rate_vs_heuristic`, `mean_margin`, `entropy`, `approx_kl`, and
`clip_frac`. If KL stays well below `0.01` for many iterations and margins do
not improve, consider a small LR increase; if early stopping fires frequently
or clip fraction jumps, lower LR. Per-iter `win_rate_vs_heuristic` is noisy
(only 1–3 heuristic games per iter once snapshot pool fills) — for
confident eval use a separate N-game tournament after training.

## Architecture

**Target: transformer-based NN with pair/edge representation, trained via behavior cloning then self-play PPO.**

**Authoritative decision records are in memory** (auto-loaded via `MEMORY.md`):
- *NN architecture decision* — locked choices: edge-only tokens, rejected hybrid node+edge
- *Training pipeline* — heuristic → BC → PPO; why sniper was rejected as BC teacher
- *User hardware* — M2 8GB / MPS constraints

The sections below are the inline reference; consult memory when recommending changes.

### Representation

The harness emits a `(N_max, N_max, F)` float tensor (`N_max=50` with zero-padding) plus a `(N_max, N_max)` bool legality mask. Each cell `[i][j]` is the edge "launch from planet i to planet j" — the action is a pair, so features are pair-shaped. There is no per-node tensor; per-node info rides on the edges it touches.

**Guiding principle**: if the harness computes X downstream of a feature, don't feature it. The NN picks `(src, tgt, ships)`; it never computes an angle. Orbital mechanics, sun/planet obstruction, lead-intercept — all absorbed into harness outputs.

### Per-edge features (11 scalars)

Canonical list and indices live in `harness.py` (`FEATURE_*` constants). Summary:

Core pair features — the reason edge-rep exists:
1. `eta` — lead-intercept turns to arrival
2. `ships_needed` — `future_garrison_at_eta + 1`, bakes in target production over transit, friendly reinforcements inbound, enemy threats resolved via combat rules
3–5. `kind_reinforce` / `kind_attack_enemy` / `kind_attack_neutral` — edge-kind one-hot (mine→mine, mine→enemy, mine→neutral)

Source context (replicated across outgoing edges from src):
6. `src_ships` — current garrison
7. `src_net_threat` — enemies incoming minus friendlies incoming

Target context (replicated across incoming edges to tgt):
8. `tgt_production` — long-term value of holding
9. `tgt_will_fall` — defense trigger (bool)
10. `tgt_expiry` — comet turns-until-gone, 999 sentinel for permanent planets

Global (broadcast to every edge):
11. `turns_left` — needed for endgame hoard decisions

Legality has **two masks**:
- `legal_mask` — sun-crossing only, built in `GameView.__init__`. Cheap, used as pre-filter.
- `action_mask(safety_margin)` — radar-validated under deterministic ship sizing. Authoritative; this is what the model sees. Cached per-view; patched incrementally by `apply_planned_move`.

**Explicitly cut as redundant**: `radius` (= 1+ln(prod)), `is_orbiting` (absorbed into `eta`), raw x/y (pairwise `eta` encodes positional relationships), `angular_velocity`, `future_position at t=k`, raw `current_ships` on target (superseded by `ships_needed_to_capture`).

### Model & action space

- Each of the `N_max²` edges is a token. Embed 11 features → dim 64 via linear layer.
- 3 × `TransformerEncoderLayer`. Self-attention lets every edge read every other edge.
- Ship sizing is **not** learned — it's deterministic per edge (`GameView.deterministic_ship_count`). The model's decision is purely `(src, tgt)` or stop.
- Action space: `N_max² + 1 = 2501` slots. Last slot is stop. Inside one turn we call forward up to `MAX_MODEL_MOVES` (currently 3) times, applying each chosen move via `apply_planned_move` before the next forward.
- Two heads: policy (scalar per edge + stop logit) and value (mean-pool → MLP → scalar).
- ~208K params. Target hardware: M2 8GB (MPS or CPU — MPS saves ~30 ms/forward but adds overhead elsewhere; default CPU for now).

### Build order

1. ✅ **`harness.py` / `radar.py`** — GameView with edge tensors, radar-validated action mask, deterministic ship sizing, incremental sub-move + cross-turn updates. Drift-checked against cold rebuild.
2. ✅ **Heuristic agent** — `heuristic_agent` in `agents.py`. Defend → ROI-expand → hoard. 10/10 vs sniper. Doubles as (a) BC teacher, (b) permanent eval baseline, (c) PPO opponent-pool member.
3. ✅ **Random model-space agent** — `random_model_space_agent`. Smoke-tests the full env→mask→action→step pipeline.
4. ✅ **Transformer skeleton** — `OrbitWarsTransformer` (legacy padded) and `OrbitWarsEdgeTransformer` (active CPU dynamic). Forward-pass verified; illegal-edge mass = 0; stateful agents play complete games under latency budget.
5. ✅ **Behavior cloning** — `bc_baseline.pt` (legacy padded) and `bc_cpu_model.pt` (active CPU stack) both trained. BC teacher is the heuristic (not sniper — see memory for why). CPU BC: 95.9% val accuracy, 4/4 vs sniper, 2/4 vs heuristic_cpu.
6. 🟨 **Self-play PPO** — *Active CPU stack:* first 60-iter PPO run complete on `bc_cpu_model.pt` → `rl_cpu_model.pt`. Loop is healthy (no collapse, no early-stop) but policy moves slowly at 4 games/iter — bigger batches needed for clear win-rate-vs-BC gains. *Legacy padded stack:* never trained for real. Opponent pool mixes 33% heuristic / 67% past snapshots once snapshots exist (100% heuristic before that), FIFO eviction at 8 snapshots; PFSP not implemented.
7. 🟨 **Eval** — `eval_bc.py` and `eval_bc_cpu.py` exist for ad-hoc head-to-head checks. No proper N-game tournament harness with CIs yet — needed before claiming RL beats BC.

### Debugging layers (symptom → likely culprit)

| Symptom | Layer | Likely cause |
|---|---|---|
| Model picks illegal moves | Harness / mask | Mask axis mismatch |
| Policy uniform regardless of state | Transformer | Feature collapse or embedding too small |
| One edge always picked | Policy head | Logit explosion / NaN |
| BC accuracy won't climb | Features | Missing signal; re-check Step 1 |
| RL reward rises but win rate doesn't | Reward shaping | Agent exploiting proxy reward |

## What we know about beating the sniper

The baseline `nearest_planet_sniper` is a greedy local optimizer with several exploitable weaknesses (it's ~80 ships down to no-op after 30 turns because its attacks fail en route). From user playtesting at 1 s / turn, the winning tactics that mattered most were:

1. **Garrison-aware targeting** — attack planets with low garrison relative to your attack force
2. **Travel-time-inflated sizing** — send enough that you still capture after the target produces during transit
3. **Reactive defense** — reinforce planets flagged as losing in the threat panel
4. **Sun avoidance** — don't fire through the middle
5. **Lead-targeting orbiters** — use the predicted-position line
6. **Endgame hoard** — stop attacking in the last ~20 turns

The user did *not* use ROI-by-production (targeting high-prod planets over low-prod ones). Whether that's actually decisive is an open question — worth A/B-ing once the harness exists.

**Core insight**: "coming back is very difficult — take as many planets as possible early." Early expansion > late comebacks.

## Conventions

- Tech decisions: user has said "take tech decisions yourself" — don't over-ask for trivia, but check in before large structural choices.
- **Venv discipline**: no global installs. Use `.venv/bin/python` for everything.
- Don't edit `getting-started.ipynb` — it's the Kaggle tutorial and is truncated anyway.
- Prefer editing existing files over creating new ones. New files only when the split is clear (e.g., `harness.py` is a genuine new module, not a reshuffle).
- Memory at `~/.claude/projects/-Users-martian-Documents-Code-orbitWars/memory/` holds durable project context. Update or replace entries when facts change; don't leave stale ones around.

## Quick experiments

```bash
# Sanity-check the environment
.venv/bin/python -c "
from kaggle_environments import make
env = make('orbit_wars', debug=False)
env.reset(); env.step([[], []])
obs = env.state[0].observation
print(f'planets={len(obs[\"planets\"])} ω={obs[\"angular_velocity\"]:.4f}')
"

# Run sniper vs sniper to see typical game lengths / scores
.venv/bin/python -c "
from kaggle_environments import make
from agents import nearest_planet_sniper
env = make('orbit_wars', debug=False)
env.run([nearest_planet_sniper, nearest_planet_sniper])
print([(s.reward, s.status) for s in env.steps[-1]])
"

# Per-turn latency bench (100-turn heuristic-vs-heuristic game)
.venv/bin/python bench.py            # CPU
.venv/bin/python bench.py --mps      # also time model forward on MPS
```
