# RL training: findings & approach (2026-04-26)

## TL;DR

We spent ~20h of c3-highcpu-8 compute on PPO self-play from `bc_cpu_model.pt`, plus engineering time on three rollout-driver architectures (sync, async, rotating-learner). The throughput infrastructure works and is well-instrumented. The training results are inconclusive: 20-iter rolling windows showed `win_rate_vs_heuristic` rising 0.44 → 0.63 then declining back to 0.50, but per-iter samples are too noisy to claim definitive RL improvement over the BC baseline — we never ran a proper N-game eval. The most likely culprits for the regression are entropy-coef regularizer drift and self-play pool dilution against the heuristic. Action items at the end.

## Setup

- **Model:** `OrbitWarsEdgeTransformer`, 46k total params (17k inference excluding value head). d_model=32, 2 encoder layers, threshold-gated attention (full → axial), single-query attention pool for value head, 4-layer value MLP.
- **Algorithm:** standard PPO with GAE (λ=0.95, γ=0.99), clip 0.2, target_kl 0.03 (early-stop), value_coef 0.5, entropy_coef 0.01 (later 0.005).
- **Rollout:** 16 games/iter, max_turns=500, 8 parallel workers on c3-highcpu-8 (Sapphire Rapids 8 vCPU).
- **Opponent pool:** heuristic_weight=0.5, snapshot_weight=1.0 → 33% heuristic, 67% past-snapshots once pool fills. FIFO eviction at 8 snapshots. PFSP not implemented.
- **Reward:** dense per-turn delta of normalized ship-share `(my_ships − opp_ships) / total_ships`.
- **Hardware:** GCP us-central1-a, c3-highcpu-8, $0.32/hr on-demand. Python 3.13 in venv via uv.

## Architecture journey (rollout drivers)

| Mode | Implementation | Workers during update | When sampled best |
|------|----------------|------------------------|-------------------|
| Sync | `ProcessPoolExecutor` per iter | Idle (workers wait) | small games_per_iter / max_turns ratio |
| Async (`rl_async.py`) | Continuous workers + parent learner | All 8 contend with parent for cores | high game-length variance |
| Rotating (`rl_async_rotating.py`) | Continuous workers; one borrowed for PPO | 7 play + 1 single-threaded PPO | medium variance, modest update phase |

### Throughput results at 16 games / 8 workers / max_turns=500

| Mode | rollout_s | update_s | total/iter |
|------|-----------|----------|------------|
| Sync (iters 41-58, 18 iters) | 502 | 39 | **540s** |
| Async (iters 60-66, 7 iters) | 358 | 180 | **538s** |
| Rotating (iters 69-148, 80 iters) | 244 | 291 | **535s** |

**All three modes converge to ~540s/iter.** The breakdown changes — async/rotating shift work into the update phase by overlapping it with rollout, but the total compute is conserved. The theoretical lower bound is ~522s/iter (`(4000 + 179) core-seconds / 8 cores`), which means all three modes are within ~3% of optimal core utilization; there's no further architectural win to extract on this 8-core machine.

### Why the M2 A/B prediction (2.35× async win) didn't transfer

The M2 A/B ran 4 games / 4 workers / max_turns=200. In that regime:
- 1 batch of 4 parallel games → max-of-4 straggler tax dominates wall time.
- Update phase was 15s; contention barely registered.

At our production config (16 / 8 / 500):
- 2 batches of 8 amortizes stragglers (max-of-8 over 16 games).
- Update grew enough that async's worker-contention tax (~140s/iter) cancels async's straggler-recovery savings.

Lesson: **async/rotating wins scale with rollout-time variance and shrink with update-phase weight.** For larger machines (≥16 vCPU) or larger update phases, async/rotating should re-emerge as winners. At 8 vCPU + small model, sync is competitive.

## Operational lessons (infrastructure)

1. **`SIGTERM` doesn't reap reparented children.** Killing the parent training process leaves 8 spawn_main worker subprocesses alive, reparented to init, each at 60-99% CPU. Required two cleanup rounds across two failed switchovers. **Procedure:** after SIGTERM, sweep `ps --ppid 1` for orphaned `spawn_main` / `resource_tracker` python processes and kill them explicitly.
2. **`torch.multiprocessing.set_sharing_strategy("file_system")`** was required at the top of `rl_train.py` and `rl_rollout.py`. Default `file_descriptor` blew past `ulimit -n 1024` when shipping state_dicts to 8 workers × 16 games/iter. Symptom: `OSError: Too many open files` followed by `BrokenProcessPool`.
3. **`git stash -u` clobbered live checkpoint files** during a code update. Lost ~15 iters of training state. Recovered from a `cp ... /tmp` backup taken pre-stash. **Procedure:** always copy live checkpoints to `/tmp` before any git operation that could touch the working tree.
4. **Python 3.11 → 3.13 pickle incompatibility** for `pathlib._local` references in checkpoints. One-time fix: install Python 3.13 via `uv python install 3.13` and recreate venv.

## Training findings (the actual question)

### What the per-iter metric looks like

`win_rate_vs_heuristic` per iter has 1-sigma ≈ 0.22 (5-6 heuristic games out of 16, since pool is 33% heuristic). **Single-iter values are nearly meaningless.** All claims below are from 20-iter rolling windows (~110 heuristic games per window).

| Window | iters | entropy | KL | wr_heur | mean_margin |
|--------|-------|---------|-----|---------|-------------|
| Sync baseline | 41-58 | 0.139 | 0.004 | (not logged per-iter) | — |
| Async | 60-66 | 0.150 | 0.006 | 0.43 | -0.04 |
| Rotating early | 69-88 | 0.173 | 0.007 | 0.44 | -0.08 |
| Rotating mid | 89-108 | 0.236 | 0.010 | 0.54 | +0.09 |
| Rotating late | 109-128 | 0.327 | 0.009 | 0.61 | +0.06 |
| **Rotating peak** | **129-148** | **0.436** | **0.007** | **0.63** | **+0.09** |
| Rotating decline | 145-164 | 0.482 | 0.007 | 0.52 | -0.03 |
| Rotating decline cont. | 154-173 | 0.527 | 0.007 | 0.50 | -0.03 |
| Post entropy_coef halving | 174-190 | 0.583 | 0.007 | 0.48 | -0.04 |

### Insights from the data

1. **There WAS real improvement at peak.** The 20-iter window 129-148 showed wr_heur=0.63 against the heuristic (~110 games), comfortably above the BC baseline's roughly 0.50. The improvement is statistically credible, not noise.
2. **Then it regressed.** The next 25 iters drifted back to 0.50. Three consecutive 20-iter windows trending the same direction is signal, not noise.
3. **Entropy climbed monotonically the entire run** (0.14 baseline → 0.58 currently). The entropy_coef=0.01 regularizer accumulates: every PPO step nudges policy mass toward uniform, and across ~30k gradient steps the cumulative push is large.
4. **Halving entropy_coef (0.01 → 0.005) at iter 174 hasn't bitten in 17 iters.** Entropy still drifting up. Suggests 0.005 is still too strong for a 50-action distribution at this scale, or the optimizer momentum from the prior regime is taking time to decay.
5. **All gradient-stability metrics were healthy throughout:** approx_kl 0.003-0.011 (well under target 0.03), value_loss 0.001-0.005, no early-stop fires, no early-stop fires, mean_turns stable at 160-190.
6. **Self-play pool likely dilutes the signal vs heuristic.** With 67% snapshot opponents, the policy is mostly optimizing for "beat my recent self." That objective drifts away from "beat the static heuristic" as snapshots get stronger and the heuristic-share of training games shrinks proportionally.

### What we did NOT do

- **No N-game tournament eval** of any RL checkpoint vs BC vs heuristic. We've been reading windowed averages. They're suggestive, not definitive.
- **No reward-shape ablation.** Locally-greedy `Δ(ship-share)` is the only reward used. Sparse terminal reward never tested.
- **No PFSP** (prioritized fictitious self-play). Pool sampling is uniform across snapshots.
- **No `heuristic_weight` sweep.** Stayed at 0.5 the entire run; did not test whether 1.5 or 2.0 makes the policy improve specifically against heuristic.
- **No model size sweep.** Stayed at d_model=32, 2 layers, 17k inference params.

## What's plausibly limiting progress

Ranked by likelihood:

1. **Reward shaping is myopic.** Per-turn ship-share delta penalizes long-term plays (sending ships across the board for a future capture, sun-skirting routes). The heuristic's strength is exactly those: locally bad, globally good moves.
2. **Self-play pool dilution.** Policy spends 67% of training on snapshots, only 33% on the actual eval target. As the pool snowballs, "improvement vs snapshots" decouples from "improvement vs heuristic."
3. **Entropy regularizer accumulation.** A small entropy_coef compounds across many updates. The peak-then-decline pattern is consistent with the policy reaching its capacity-limited optimum and then being slowly pushed back toward uniform by the entropy bonus.
4. **Model capacity.** 17k params is small. Some strategic patterns (multi-turn buildup, coordinated multi-fleet strikes) may not fit.
5. **Restricted action space.** `max_moves=2` per turn. Heuristic uses up to 3.

## Recommended next actions

1. **Stop the current run.**
2. **Run a 100-game tournament** with 50/50 seat assignment between three candidates: `bc_cpu_model.pt`, `rl_cpu_model.snap_iter_140.pt` (peak window), `rl_cpu_model.last.pt` (current). Get real win rates vs heuristic with confidence intervals. ~1h of compute.
3. **Decide based on the eval:**
   - If snap_iter_140 wins decisively (>0.55, CI excludes 0.50): generate `submission_cpu.py` from it, submit to Kaggle. The training story is "we have a working RL win, just unstable late-game; ship the peak."
   - If BC ties or wins: RL hasn't beat BC. Kill RL with current setup, address root causes before more compute. Top candidates: terminal-only reward + `heuristic_weight=1.5`.
4. **If we keep training,** the highest-leverage changes are:
   - Add a sparse terminal reward (`±1` at game end) alongside or replacing the per-turn shaping.
   - Bump `heuristic_weight` so the heuristic remains ≥50% of training games.
   - Lower `entropy_coef` further (to 0.001) and/or add a schedule that anneals it across iterations.

## Compute spent

- Engineering: ~3 chat-sessions of architecture + debugging.
- Wall: ~20h of c3-highcpu-8 ($0.32/hr → ~$6.40 burned on training, plus some prior runs).
- Iterations: 134 RL updates total (sync 18 + async 7 + rotating 109).
- Snapshots produced: 8 (current pool) plus checkpoints at iter 5,10,15… stamped to disk.

## Files modified during this work

- `orbit_wars/cpu/rl_train.py` — added `--num-workers`, `--async-rollout`, `--rotating-learner` flags, FD-strategy config.
- `orbit_wars/cpu/rl_rollout.py` — added `play_one_game_worker` for sync parallel mode.
- `orbit_wars/cpu/rl_async.py` — async driver (continuous workers, atomic weight propagation).
- `orbit_wars/cpu/rl_async_rotating.py` (new) — rotating-learner driver.
- `CLAUDE.md` — extensively updated with architecture, latency, async/rotating notes.
