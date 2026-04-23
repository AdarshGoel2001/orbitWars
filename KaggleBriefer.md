# Kaggle Briefer

## Competition Runtime

Orbit Wars is a Kaggle Environments competition. Submissions are Python agents
that receive one observation and return a list of moves:

```python
def agent(obs, config=None):
    return [[from_planet_id, angle_radians, num_ships], ...]
```

Observed environment configuration:

```text
episodeSteps: 500
actTimeout: 1
agentTimeout: 2
runTimeout: 1200
shipSpeed: 6.0
cometSpeed: 4.0
remainingOverageTime: 2
```

`actTimeout` is 1 second per call. Calls can exceed 1 second briefly, but the
agent is disqualified once `remainingOverageTime` drops below zero. In practice
we should target much less than 1 second per call.

Kaggle runtime hardware, per user-supplied staff/comment data:

```text
No GPU
~1.6 CPU cores
Slightly under 7 GB RAM
```

## Player Counts

The Orbit Wars spec supports `[2, 4]` agents.

Local 4-player validation works by passing four agents to `env.run()`:

```python
from kaggle_environments import make

env = make("orbit_wars", debug=True)
env.run([agent, agent, agent, agent])
```

Setting config keys like `{"agents": 4}`, `{"playerCount": 4}`, or
`{"agentCount": 4}` did not change the `reset()` state by itself. The local
environment infers 4-player mode from the number of agents passed to `run()`.

We validated short 4-player self-play locally using isolated imports of the
generated `submission.py`, which is important because separate seats should not
share module globals.

## Submission Paths

### CLI/API Submission

We installed Kaggle CLI into the repo venv:

```bash
.venv/bin/python -m pip install kaggle
```

Auth worked with:

```bash
KAGGLE_API_TOKEN=...
KAGGLE_CONFIG_DIR=/tmp/kaggle
```

Working CLI submission command:

```bash
env KAGGLE_API_TOKEN="$KAGGLE_API_TOKEN" KAGGLE_CONFIG_DIR=/tmp/kaggle \
  .venv/bin/kaggle competitions submit \
  -c orbit-wars \
  -f submission.py \
  -m "message"
```

`competitions submit` supports:

```text
-f FILE_NAME
-k KERNEL
-v VERSION
-m MESSAGE
```

It does not expose a `--model` attachment flag.

### Archive Submission

Kaggle staff comment found by the user:

> Submission size is limited to 100 MB. You can upload as a .tar.gz or .zip
> (support was added). Reference absolute model paths; in prod they live under
> `kaggle-environments/agent/weights.pkl`.

We added archive generation support:

```bash
.venv/bin/python make_nn_submission.py \
  --allow-bc-fallback \
  --backend numpy \
  --max-moves 1 \
  --candidate-limit 64 \
  --archive-out submission_nn.zip
```

Archive layout produced:

```text
submission.py
weights.npz
```

Generated code searches:

```text
/kaggle-environments/agent/weights.npz
agent/weights.npz
weights.npz
```

Local archive extraction and execution worked. However, CLI/API submissions of
both `.tar.gz` and `.zip` returned:

```text
400 Client Error: Bad Request
```

So archive upload may be supported in the web UI before it is accepted by the
public CLI/API endpoint, or the required archive format differs from our test.

## Current Builders

There are now two submission builders:

```text
make_submission.py      heuristic baseline builder
make_nn_submission.py   neural/NumPy builder
```

`make_submission.py` generates a single-file heuristic `submission.py`.

`make_nn_submission.py` can generate:

```text
single-file embedded NumPy submission.py
single-file torch submission.py
archive submission_nn.zip / .tar.gz with weights.npz
```

No RL checkpoint currently exists in `checkpoints/`. The NN submissions tested
used:

```text
checkpoints/bc_v1.pt
```

## Submitted Versions And Outcomes

Heuristic submission:

```text
Status: COMPLETE
Public score observed: ~606-630
```

Torch embedded NN submission:

```text
Status: ERROR
Reason: TIMEOUT
Agent logs showed ~12.5s duration with no traceback.
Likely cause: torch import/load/runtime too slow in Kaggle agent runner.
```

Full NumPy embedded NN submission:

```text
Status: ERROR
Reason: TIMEOUT
Agent logs showed first real action ~3.16s.
Cause: full 2500-token transformer forward + mask work too slow on Kaggle CPU.
```

Candidate-limited NumPy embedded NN submission:

```text
Status: passed validation / produced logs
Message: NN singlefile numpy bc_v1 candidate_limit_64 max_moves_1
```

This version uses a single embedded `submission.py` accepted by CLI/API.

## Runtime Measurements

### Failed Full NumPy

Kaggle first real action:

```text
~3.16s
```

Local first action for similar full NumPy:

```text
~0.3-0.45s
```

Approximate Kaggle/local multiplier:

```text
~7x to 10x
```

### Passing Candidate-Limited NumPy

For episode logs `75267597-*.json`, Kaggle durations:

```text
agent 0:
  calls: 499
  mean: 0.214s
  p95: 0.781s
  p99: 0.893s
  max: 1.008s
  >1s calls: 1
  overtime used: 0.008s

agent 1:
  calls: 358
  mean: 0.605s
  p95: 0.951s
  p99: 1.010s
  max: 1.053s
  >1s calls: 8
  overtime used: 0.169s

agent 2:
  calls: 499
  mean: 0.045s
  p95: 0.294s
  p99: 0.536s
  max: 0.692s
  >1s calls: 0
  overtime used: 0s
```

Local 4-player 120-step timings with current generated `submission.py`:

```text
seat 0 p95: 0.121s
seat 1 p95: 0.113s
seat 2 p95: 0.110s
seat 3 p95: 0.060s
```

Practical planning multiplier:

```text
Kaggle ≈ 5x-8x local Apple CPU
Use 10x for safety.
```

Shipping rule of thumb:

```text
local full agent(obs) p95 < 100 ms
local max preferably < 150 ms
```

Safer target:

```text
local p95 < 70 ms
```

Measure full `agent(obs)`, not raw model forward. Full call includes:

```text
GameView build/update
action_mask
candidate selection
model forward
action conversion/apply
```

## NumPy Model Runtime

The NumPy runtime was added because torch timed out in Kaggle.

It exports checkpoint tensors into `np.float32` arrays and implements:

```text
linear layers
layer norm
GELU
multi-head self-attention
softmax
policy head
stop head
```

The value head is not implemented because it is unused during action selection.

When `candidate_limit=None`, NumPy closely matched PyTorch on a sampled
observation:

```text
same argmax
max policy difference ≈ 2.7e-4
```

Expected small differences come from:

```text
GELU implementation
floating-point reduction order
NumPy vs PyTorch kernels
softmax/layernorm numeric details
```

## Candidate Limiting

The passing submission currently uses:

```text
candidate_limit = 64
max_moves = 1
```

This is not equivalent to the full PyTorch/MPS model.

Instead of all `2500` edge tokens, it:

1. Computes the normal legal/action mask.
2. Finds legal edge candidates.
3. Scores them with a fast hand-coded prior:

```python
score =
    2.0 * attack_kind * production / (ships_needed + eta + 1.0)
    + 0.25 * reinforce_kind
    + 0.02 * src_ships
    - 0.01 * eta
```

4. Keeps the top 64 candidates.
5. Runs transformer attention only over those candidates.
6. Sets all non-kept edge logits to `-1e9`.

This made the submission pass timing, but it changes the deployed policy class.
The model cannot choose an edge filtered out by the prior. This is a major
training/deployment mismatch and can cap performance.

## Architecture Implications

Full attention over `50 * 50 = 2500` edge tokens is not Kaggle-feasible with the
current transformer shape:

```text
d_model = 64
layers = 3
heads = 4
ff_dim = 384
```

The main bottleneck is token count:

```text
attention cost ≈ N^2
2500^2 = 6,250,000 attention pairs per layer
```

Candidate limiting is fast but undesirable because it uses a heuristic ranker.

Better long-term direction:

```text
full attention over actual legal owned-source edges only
```

Do not use padded `50x50` tokens. Use dynamic action tokens:

```text
tokens = every action_mask-valid edge from owned planets
```

Early game token counts can be tiny:

```text
1 owned source * 32 targets = ~32 tokens
5 owned sources * 32 targets = ~160 tokens
```

This preserves model agency while reducing cost structurally rather than with a
hand ranker.

Likely Kaggle-feasible model family:

```text
EdgeSetTransformer
tokens: all legal owned-source edges
d_model: 16-32
layers: 1-2
heads: 1-2
ff_dim: 64-128
stop token
```

The harness already encodes the hard mechanics:

```text
radar legality
ETA
ships needed
threat projection
lead targeting
deterministic ship count
```

Therefore the NN does not need to learn orbital physics. It primarily needs to
learn prioritization and timing.

## Local Validation Commands

2-player / 4-player local validation should use isolated imports where possible.

4-player self-play pattern:

```python
from kaggle_environments import make
import importlib.util, pathlib, sys

path = pathlib.Path("submission.py").resolve()

def load_agent(i):
    name = f"submission_seat_{i}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.agent

agents = [load_agent(i) for i in range(4)]
env = make("orbit_wars", configuration={"episodeSteps": 120}, debug=True)
env.run(agents)
print([s.status for s in env.steps[-1]])
print([s.reward for s in env.steps[-1]])
```

For latency calibration, wrap each agent call and record:

```text
min
mean
median
p90
p95
p99
max
sum
calls > 1s
```

## Open Issues

- Need a proper local latency benchmark script checked into the repo.
- Need to remove or replace heuristic candidate limiting before serious NN
  training, or train the deployed candidate-limited policy explicitly.
- Need to redesign model around dynamic legal edge tokens.
- Need to verify archive upload through Kaggle web UI, since CLI/API rejected
  `.zip` and `.tar.gz` with HTTP 400.
- Need to test with a real RL checkpoint once one exists.
- Need to decide whether final/public scoring uses 2-player only or includes
  4-player episodes.
