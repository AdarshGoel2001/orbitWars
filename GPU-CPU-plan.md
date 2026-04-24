# GPU-train / CPU-infer plan

Context resumption doc for the CPU-focused model rebuild. Read this to
pick up where we left off.

## The problem

The current stack (`harness.py`, `model.py`) emits a padded 50×50 edge
grid = 2500 tokens. On Kaggle (CPU, ~1.6 cores, no GPU), numpy forward
over 2500 tokens times out at ~3.16 s first action. The passing
submission hacked it down with a hand-coded candidate-limit of 64, which
caps model agency and creates train/deploy mismatch.

**Root cause**: GPU-era design reflex — pad everything, mask later. CPU
can't cash in masked computation; padding is dead compute. Attention is
O(N²), so 2500 vs ~200 real legal edges is a ~150× pointless overhead.

## The plan, in one line

Parallel stack (`harness_cpu.py`, `model_cpu.py`, etc.) that emits
**variable-length tokens — only legal edges** — and uses an architecture
sized for CPU inference. Train on GPU, run inference in numpy on Kaggle
CPU. Old stack stays intact for A/B comparison.

## Architecture decisions (locked)

### Harness: `harness_cpu.py` — dynamic edge tokens

- One token per **action-mask-valid edge** (radar-validated, sun-safe,
  feasible under deterministic ship sizing). Typical N = 30–300 (p95
  ~310 observed in heuristic-vs-heuristic games).
- No padding, no fixed slot grid.
- **Comets dropped entirely in v1**. Planets in `obs.comet_planet_ids`
  are filtered out at ingest. Re-introduce later as a feature addition.
- **11 features**: dropped `tgt_expiry` from the old harness because comets
  are excluded, and added `src_can_fund` so the model can see targetable
  but currently underfunded attacks instead of inferring affordability from
  raw `src_ships` and `ships_needed`.

Feature order: `eta, ships_needed, kind_reinforce, kind_attack_enemy,
kind_attack_neutral, src_ships, src_net_threat, tgt_production,
tgt_will_fall, src_can_fund, turns_left`.

- Reuses `radar.py` and `targeting.py` unchanged.
- Output `TokenBundle`: `edges (N, 11)`, `src_ids (N,)`, `tgt_ids (N,)`,
  `ships (N,)`, `angles (N,)`, `planet_ids (P,)`.
- Incremental updates: dirty-flag + lazy rebuild (no strip patching).
  `apply_planned_move(token_idx)` and `update_from_obs(new_obs)` are
  the mutation entry points. Carry-over fleet predictions preserved
  across turns (`eta -= 1`).
- Action decode: token_idx → stored `(src_pid, angle, ships)`. O(1).

### Model: `model_cpu.py` — edge-set transformer with threshold-gated attention

- **Single head**, fused Q/K/V (`Linear(d, 3d)`) — one big matmul > many
  small, which matters on CPU BLAS.
- **Pre-LN**, GELU, 2 encoder blocks.
- **Attention pattern, per-sample**:
  - `N ≤ axial_threshold` → full self-attention among valid tokens
  - `N > axial_threshold` → axial: layer 1 attends within same `src_id`,
    layer 2 within same `tgt_id`
  - Default threshold = 256. Implementation uses `torch.where` on a
    per-sample boolean gate, so mixed-N batches work correctly. Fast
    path skips building axial masks if no sample crosses threshold.
- `d_model=32`, `d_ff=64`, ~**46K params total, ~18K without value head**.
- **Value head** is oversized on purpose (attention pool + 4-layer MLP,
  32→128→128→64→1, ~29K params). Gated by `compute_value=True`;
  inference passes `False` and skips it entirely. Rationale: a strong
  value head is the most common gate on PPO working.
- Stop token: separate `AttentionPool + Linear(32→1)` producing the
  `(N+1)`-th logit. Edge logits get `-1e9` on pad positions before
  concatenation.

Forward:
```python
logits, value = model(edges, src_ids, tgt_ids,
                      valid_mask=None, compute_value=True)
# logits: (B, N+1), last index = stop
# value:  (B,) or None
```

### Train on GPU, infer on CPU

- Same `nn.Module` code path. GPU-train uses padded batches +
  `valid_mask`; CPU-infer uses unpadded B=1.
- Parity verified (Step 5): padded vs unpadded outputs match to
  float32 precision on valid positions.
- Kaggle submission uses a numpy port of forward (see
  `bench_cpu.py::forward_numpy`). Tanh-GELU approx differs from torch's
  exact erf by ~1e-4 in logits — fine for runtime, will need the exact
  op or matched approx for production.

## Files shipped

| File | Purpose | Status |
|---|---|---|
| `harness_cpu.py` | `GameView_CPU`, `TokenBundle`, dynamic edges | ✅ Done |
| `model_cpu.py` | `OrbitWarsEdgeTransformer`, threshold-gated attention | ✅ Done |
| `agents_cpu.py` | CPU-token heuristic teacher | ✅ Done |
| `bc_data_cpu.py` | Ragged CPU BC shard capture | ✅ Smoke pass |
| `tests/test_model_cpu_parity.py` | Padded-batch vs unpadded-single parity test | ✅ Passes |
| `bench_cpu.py` | Numpy forward + latency benchmark | ✅ Passes |

Old stack (`harness.py`, `model.py`, `bc_data.py`, `bc_train.py`, etc.)
is untouched. Don't remove it until the new path has won on merit.

## Benchmark results

On M2 CPU, 156 forwards across a 200-turn heuristic-vs-heuristic game:

```
N tokens: mean=174  p95=310  max=324   ← some turns trigger axial mode

numpy forward_numpy() latency:
  mean=1.01 ms   p95=1.88 ms   max=2.36 ms

Projected Kaggle (×10 slowdown):
  p95 ≈ 19 ms   max ≈ 24 ms
  Budget per forward (MAX_MODEL_MOVES=3): ~333 ms
  Headroom: ~17×
```

For reference: old model forward on CPU is ~95 ms, numpy submission
timed out at ~3.16 s first action. New design is ~100× faster.

## What's left

### Step 6 — BC pipeline

**Decision**: re-capture BC data using the new harness rather than
converting old shards. Simpler, avoids comet-mismatch bookkeeping.

- `bc_data_cpu.py` — heuristic-vs-heuristic games through `GameView_CPU`,
  writes compressed ragged shards in the new format
- Write `bc_train_cpu.py` — pad-collate, masked cross-entropy,
  train on GPU (CUDA when available; MPS fallback)

**Proposed shard format**: one `.npz` per shard with ragged packing:
  - `edges_packed (total_N, 11)`, `src_ids_packed`, `tgt_ids_packed`
  - `offsets (K+1,)` — `edges_packed[offsets[i]:offsets[i+1]]` = example i
  - `action_idx (K,)` — new-space: position in packed edges or `N` for stop
  - `n_tokens (K,)` — convenience
  - metadata (game, step, player, submove, src_slot, tgt_slot, ships)

~10× smaller than one-file-per-example; trivial to mmap during training.

### Step 7 — Benchmark per-turn end-to-end

Already partly done (`bench_cpu.py` times numpy forward). Extend to
measure full `agent(obs)` including `GameView_CPU.update_from_obs`,
`tokens()` rebuild, forward, and action decode. Same ×10 Kaggle
projection applies.

### Step 8 — NumPy submission export

- Adapt/port `make_nn_submission.py` to `model_cpu`
- Weight export: state_dict → `.npz`
- Production numpy forward (use exact GELU via tanh-approx-matched
  torch training, OR switch torch training to tanh approx)
- Archive or single-file submission
- Verify parity vs torch on sampled obs before submit

### Step 9 — RL scaffolding adaptation

`rl_rollout.py`, `rl_opponent_pool.py`, `rl_ppo.py`, `rl_train.py` all
assume the old harness / model. Will need parallel `*_cpu.py` versions
or parameterize. Defer until BC baseline lands and wins vs heuristic.

## Decisions still open

1. **BC smoke-train** on how many games before scaling up? Suggest
   ~100 games, 5 epochs, see if accuracy > 50% on held-out shards.
2. **Training device**: CUDA migration was flagged in old CLAUDE.md as
   the MPS escape hatch. If we have CUDA access, use it. Otherwise
   MPS → CPU fallback is fine for a ~46K-param model at these N values.
3. **Safety margin**: baked into `GameView_CPU.__init__(safety_margin=1)`.
   Old harness took it as a per-call arg. Fine as-is; change if needed.
4. **Axial threshold**: default 256. If BC data shows N rarely exceeds
   256, consider raising (eliminates axial path entirely) or lowering
   (trains axial mode on more data). Observed p95 ≈ 310, so axial does
   fire — keep as-is for now.

## How to resume

Start here. Re-read this doc. If user references a decision, check the
relevant section. If a file is mentioned, it's already in the repo —
don't re-create. Next action from the user will typically be:

- "Start Step 6" → write `bc_data_cpu.py`
- "Benchmark full agent call" → extend `bench_cpu.py` (Step 7)
- "Submit to Kaggle" → Step 8 work

Old stack files (`harness.py`, `model.py`, `bc_data.py`, `bc_train.py`)
are **untouched**. The CPU rebuild lives in parallel `_cpu.py` files.
