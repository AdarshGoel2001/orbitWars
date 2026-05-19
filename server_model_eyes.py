"""Orbit Wars human-vs-agent server with the **model's** vision/action space.

The human sees the same per-edge token table the CPU dynamic-edge model sees,
and acts in the model's action space: pick one of N valid edge tokens (or
stop), up to ``MAX_MODEL_MOVES`` times per turn. Ship counts and launch angles
are deterministic per token (set by the harness), exactly as the model
experiences them.

Run:
    .venv/bin/python server_model_eyes.py
    # or to play vs the trained CPU model instead of the heuristic:
    ORBIT_AGENT=cpu_model \\
    ORBIT_CPU_CHECKPOINT=checkpoints/rl_cpu_model.iter220.pt \\
    .venv/bin/python server_model_eyes.py
"""

import os
from flask import Flask, jsonify, request, render_template_string

from kaggle_environments import make

from action_space import MAX_MODEL_MOVES
from agents import nearest_planet_sniper, heuristic_agent
from orbit_wars.cpu.harness import GameView_CPU, FEATURE_NAMES

HUMAN = 0
AGENT = 1


def _build_agent():
    name = os.environ.get("ORBIT_AGENT", "heuristic")
    if name == "cpu_model":
        from agents_cpu import load_cpu_model_agent
        ckpt = os.environ.get("ORBIT_CPU_CHECKPOINT", "checkpoints/bc_cpu_model.pt")
        device = os.environ.get("ORBIT_CPU_DEVICE", "cpu")
        return load_cpu_model_agent(ckpt, device=device), f"cpu_model:{ckpt}"
    if name == "heuristic_cpu":
        from agents_cpu import heuristic_agent_cpu
        return heuristic_agent_cpu, "heuristic_cpu"
    if name == "sniper":
        return nearest_planet_sniper, "sniper"
    return heuristic_agent, "heuristic"


AGENT_FN, AGENT_NAME = _build_agent()

app = Flask(__name__)
_env = None
_human_view: GameView_CPU | None = None
_planned_actions: list = []


def new_env():
    global _env, _human_view, _planned_actions
    _env = make("orbit_wars", debug=False)
    _env.reset()
    _env.step([[], []])  # populate the world
    if hasattr(AGENT_FN, "reset"):
        AGENT_FN.reset()
    _human_view = None
    _planned_actions = []


def ensure_view() -> GameView_CPU:
    """Build or refresh the per-turn human-side view used to emit tokens."""
    global _human_view
    obs = _env.state[HUMAN].observation
    if _human_view is None:
        _human_view = GameView_CPU(obs)
    else:
        _human_view.update_from_obs(obs)
    return _human_view


def _kind_label(features) -> str:
    if float(features[2]) > 0.5:
        return "reinforce"
    if float(features[3]) > 0.5:
        return "attack_enemy"
    return "attack_neutral"


def token_payload(view: GameView_CPU):
    """Serialise the current TokenBundle into JSON-safe rows for the UI."""
    bundle = view.tokens()
    out = []
    for i in range(bundle.n):
        feats = bundle.edges[i]
        src_slot = int(bundle.src_ids[i])
        tgt_slot = int(bundle.tgt_ids[i])
        out.append({
            "idx": i,
            "src_pid": int(bundle.planet_ids[src_slot]),
            "tgt_pid": int(bundle.planet_ids[tgt_slot]),
            "kind": _kind_label(feats),
            "ships": int(bundle.ships[i]),
            "angle": float(bundle.angles[i]),
            # 11 raw features in canonical FEATURE_NAMES order:
            "features": [float(x) for x in feats],
        })
    return out


def scores(obs):
    s = {0: 0, 1: 0}
    for p in obs["planets"]:
        if p[1] in s:
            s[p[1]] += p[5]
    for f in obs["fleets"]:
        if f[1] in s:
            s[f[1]] += f[6]
    return s


def snapshot():
    obs = _env.state[HUMAN].observation
    view = ensure_view()
    return {
        "step": int(obs["step"]),
        "episode_steps": int(_env.configuration.episodeSteps),
        "planets": obs["planets"],
        "fleets": obs["fleets"],
        "human_player": HUMAN,
        "agent_name": AGENT_NAME,
        "scores": scores(obs),
        "done": bool(_env.done),
        "statuses": [_env.state[i].status for i in range(2)],
        "tokens": token_payload(view),
        "feature_names": FEATURE_NAMES,
        "planned_actions": list(_planned_actions),
        "max_moves": int(MAX_MODEL_MOVES),
    }


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/state")
def state():
    return jsonify(snapshot())


@app.route("/select", methods=["POST"])
def select_token():
    """Apply one model-style move (token index). Token list updates after."""
    if _env.done:
        return jsonify({"error": "game_done"}), 400
    if len(_planned_actions) >= MAX_MODEL_MOVES:
        return jsonify({"error": "max_moves_reached"}), 400

    data = request.get_json(silent=True) or {}
    try:
        idx = int(data.get("idx"))
    except Exception:
        return jsonify({"error": "bad_request"}), 400

    view = ensure_view()
    action = view.apply_planned_move(idx)
    if action is None:
        return jsonify({"error": "invalid_token"}), 400

    _planned_actions.append(action)
    return jsonify(snapshot())


@app.route("/commit", methods=["POST"])
def commit_turn():
    """Step the env using whatever moves were planned (possibly zero = stop)."""
    global _planned_actions, _human_view
    if _env.done:
        return jsonify(snapshot())
    agent_obs = _env.state[AGENT].observation
    agent_moves = AGENT_FN(agent_obs)
    _env.step([list(_planned_actions), agent_moves])
    _planned_actions = []
    _human_view = None  # force rebuild on next ensure_view()
    return jsonify(snapshot())


@app.route("/undo", methods=["POST"])
def undo_planned():
    """Discard planned moves for this turn and rebuild the view."""
    global _planned_actions, _human_view
    _planned_actions = []
    _human_view = None
    return jsonify(snapshot())


@app.route("/reset", methods=["POST"])
def reset():
    new_env()
    return jsonify(snapshot())


PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Orbit Wars — model-eyes</title>
<style>
  * { box-sizing: border-box; }
  body { font: 13px/1.4 system-ui, sans-serif; background: #0a0a14; color: #e6e6ee; margin: 0; padding: 16px; }
  h1 { margin: 0 0 12px; font-size: 16px; font-weight: 600; }
  h2 { margin: 14px 0 6px; font-size: 12px; font-weight: 600; color: #9cf; text-transform: uppercase; letter-spacing: 0.05em; }
  .header { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
  .chip { padding: 5px 10px; border-radius: 4px; font-weight: 600; font-size: 12px; }
  .chip.you { background: #1d4ed8; }
  .chip.them { background: #b91c1c; }
  .chip.step { background: #333; }
  .chip.agent { background: #444; color: #ccc; font-weight: 400; }
  button { background: #1f2937; color: #e6e6ee; border: 1px solid #374151; padding: 6px 12px; border-radius: 4px; cursor: pointer; font: inherit; font-size: 13px; }
  button:hover { background: #374151; }
  button.primary { background: #059669; border-color: #047857; }
  button.primary:hover { background: #047857; }
  .wrap { display: grid; grid-template-columns: 600px 1fr; gap: 18px; align-items: flex-start; }
  #canvas { background: #050510; border: 1px solid #222; display: block; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 4px 6px; text-align: left; border-bottom: 1px solid #1d1d35; }
  th { color: #9aa; font-weight: 500; background: #14142a; cursor: pointer; user-select: none; position: sticky; top: 0; }
  th:hover { color: #ccd; }
  tr.tok { cursor: pointer; }
  tr.tok:hover td { background: #1c1c38; }
  td.kind-r { color: #6cf; }
  td.kind-a { color: #f87171; }
  td.kind-n { color: #fcd34d; }
  td.f-flag { color: #fbbf24; }
  td.you { color: #93c5fd; }
  td.them { color: #fca5a5; }
  td.neutral { color: #aaa; }
  .token-table { max-height: 600px; overflow: auto; border: 1px solid #1d1d35; border-radius: 4px; }
  .planned { padding: 8px; background: #14142a; border: 1px solid #1d1d35; border-radius: 4px; min-height: 44px; }
  .move-row { padding: 4px 8px; background: #1c1c38; border-radius: 3px; margin: 3px 0; font-family: ui-monospace, monospace; font-size: 12px; }
  .hint { color: #888; font-size: 11px; margin: 4px 0; }
  .banner { padding: 10px; background: #422; border: 1px solid #833; border-radius: 4px; margin-bottom: 12px; display: none; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; margin-left: 4px; vertical-align: middle; }
  .badge.r { background: #1e3a8a; color: #cce; }
  .badge.a { background: #7f1d1d; color: #fcc; }
  .badge.n { background: #78350f; color: #fcc88c; }
  .feature-help { font-size: 10px; color: #777; margin-top: 6px; }
  .legend { font-size: 11px; color: #aaa; margin-top: 6px; }
  .legend span { display: inline-block; margin-right: 12px; }
  .legend .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; vertical-align: middle; margin-right: 4px; }
</style>
</head>
<body>
<h1>Orbit Wars — Model-eyes UI</h1>
<div class="banner" id="banner"></div>
<div class="header">
  <span class="chip you" id="score-you">You: 0</span>
  <span class="chip them" id="score-them">Agent: 0</span>
  <span class="chip step" id="step">Turn 0/500</span>
  <span class="chip agent" id="agent">agent: -</span>
  <button class="primary" id="commit-btn">Commit Turn / Stop</button>
  <button id="undo-btn">Undo Planned</button>
  <button id="reset-btn">New Game</button>
</div>

<div class="wrap">
  <div>
    <canvas id="canvas" width="600" height="600"></canvas>
    <div class="legend">
      <span><span class="dot" style="background:#3b82f6"></span>You</span>
      <span><span class="dot" style="background:#ef4444"></span>Agent</span>
      <span><span class="dot" style="background:#666"></span>Neutral</span>
      <span>Planet number = id · centre number = ships · "p#" = production</span>
    </div>
    <h2>Planned this turn (<span id="planned-count">0</span>/<span id="max-moves2">3</span>)</h2>
    <div class="planned" id="planned-list"></div>
  </div>

  <div>
    <h2>Tokens — agent's action space (<span id="token-count">0</span> valid)</h2>
    <p class="hint">Click a row to commit a move. Up to <span id="max-moves">3</span> per turn. Ship count + angle are deterministic per token, just like the agent. Hover a row to see the launch on the map.</p>
    <div class="token-table">
      <table id="tokens">
        <thead>
          <tr id="tok-head">
            <th data-k="kind">kind</th>
            <th data-k="src_pid">src</th>
            <th data-k="tgt_pid">→ tgt</th>
            <th data-k="ships">ships</th>
            <th data-k="0" data-feat>eta</th>
            <th data-k="1" data-feat>need</th>
            <th data-k="5" data-feat>src_ships</th>
            <th data-k="6" data-feat>src_threat</th>
            <th data-k="7" data-feat>tgt_prod</th>
            <th data-k="8" data-feat>fall?</th>
            <th data-k="9" data-feat>fund?</th>
            <th data-k="10" data-feat>turns_left</th>
          </tr>
        </thead>
        <tbody id="tokens-body"></tbody>
      </table>
    </div>
    <p class="feature-help">Features 2-4 are kind one-hot (collapsed into the kind column). "fall?" = tgt_will_fall; "fund?" = src_can_fund. Sortable headers. Press Enter to commit, U to undo.</p>
  </div>
</div>

<script>
let state = null;
let sortKey = "0";       // default: sort by eta ascending
let sortFeat = true;     // sortKey is a feature index (vs metadata key)
let sortDir = 1;
let hover = null;        // {src_pid, tgt_pid} | null — token currently hovered
let selectedSrc = null;  // pid of the source planet selected on the canvas, or null

const CANVAS = 600;
const BOARD = 100;
const SUN = { x: 50, y: 50, r: 10 };
const COLORS = {
  you: "#3b82f6",
  them: "#ef4444",
  neutral: "#666",
};
const cv = document.getElementById("canvas");
const ctx = cv.getContext("2d");

function u2p(u) { return u * (CANVAS / BOARD); }

function colorFor(owner) {
  if (state == null) return COLORS.neutral;
  if (owner === state.human_player) return COLORS.you;
  if (owner === 1 - state.human_player) return COLORS.them;
  return COLORS.neutral;
}

function setBanner(msg, on) {
  const b = document.getElementById("banner");
  if (on) { b.textContent = msg; b.style.display = "block"; }
  else { b.style.display = "none"; }
}

function drawCanvas() {
  ctx.fillStyle = "#050510";
  ctx.fillRect(0, 0, CANVAS, CANVAS);

  if (!state) return;

  // Sun
  const sx = u2p(SUN.x), sy = u2p(SUN.y), sr = u2p(SUN.r);
  const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, sr);
  g.addColorStop(0, "#fde68a");
  g.addColorStop(1, "#b45309");
  ctx.beginPath();
  ctx.arc(sx, sy, sr, 0, Math.PI * 2);
  ctx.fillStyle = g;
  ctx.fill();

  // Planets
  for (const p of state.planets) {
    const [id, owner, x, y, r, ships, prod] = p;
    ctx.beginPath();
    ctx.arc(u2p(x), u2p(y), u2p(r), 0, Math.PI * 2);
    ctx.fillStyle = colorFor(owner);
    ctx.fill();
    // Hover ring
    if (hover && (id === hover.src_pid || id === hover.tgt_pid)) {
      ctx.beginPath();
      ctx.arc(u2p(x), u2p(y), u2p(r) + 5, 0, Math.PI * 2);
      ctx.strokeStyle = id === hover.src_pid ? "#fde68a" : "#fca5a5";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    // Selected source ring
    if (selectedSrc !== null && id === selectedSrc) {
      ctx.beginPath();
      ctx.arc(u2p(x), u2p(y), u2p(r) + 8, 0, Math.PI * 2);
      ctx.strokeStyle = "#fde68a";
      ctx.lineWidth = 3;
      ctx.stroke();
    }
    // Valid-target ring when a source is selected
    if (selectedSrc !== null && id !== selectedSrc && state.tokens.some(t => t.src_pid === selectedSrc && t.tgt_pid === id)) {
      ctx.beginPath();
      ctx.arc(u2p(x), u2p(y), u2p(r) + 4, 0, Math.PI * 2);
      ctx.strokeStyle = "#34d399";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // Ship count in centre
    ctx.fillStyle = "#fff";
    ctx.font = "11px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`${ships}`, u2p(x), u2p(y));
    // Production badge below
    ctx.fillStyle = "#aaa";
    ctx.font = "9px system-ui";
    ctx.fillText(`p${prod}`, u2p(x), u2p(y) + u2p(r) + 8);
    // Planet id above
    ctx.fillStyle = "#778";
    ctx.font = "9px system-ui";
    ctx.fillText(`#${id}`, u2p(x), u2p(y) - u2p(r) - 6);
  }

  // Fleets
  for (const f of state.fleets) {
    const [fid, owner, x, y, angle, from, ships] = f;
    const len = Math.min(3 + Math.log10(Math.max(ships, 1)) * 3, 10);
    ctx.save();
    ctx.translate(u2p(x), u2p(y));
    ctx.rotate(angle);
    ctx.fillStyle = colorFor(owner);
    ctx.beginPath();
    ctx.moveTo(len, 0);
    ctx.lineTo(-len * 0.6, -len * 0.55);
    ctx.lineTo(-len * 0.6, len * 0.55);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    // Ship count next to fleet
    ctx.fillStyle = "#ddd";
    ctx.font = "9px system-ui";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`${ships}`, u2p(x) + len + 2, u2p(y));
  }
}

function sortedTokens(tokens) {
  const arr = tokens.slice();
  arr.sort((a, b) => {
    let va, vb;
    if (sortFeat) { va = a.features[+sortKey]; vb = b.features[+sortKey]; }
    else { va = a[sortKey]; vb = b[sortKey]; }
    if (typeof va === "string") return sortDir * va.localeCompare(vb);
    return sortDir * (va - vb);
  });
  return arr;
}

function ownerLabel(pid, planets, you, them) {
  const p = planets.find(p => p[0] === pid);
  if (!p) return ["?", "neutral"];
  const o = p[1];
  if (o === you) return ["YOU", "you"];
  if (o === them) return ["ENEMY", "them"];
  return ["NEUT", "neutral"];
}

function render() {
  if (!state) return;
  document.getElementById("score-you").textContent = "You: " + state.scores[state.human_player];
  document.getElementById("score-them").textContent = "Agent: " + state.scores[1 - state.human_player];
  document.getElementById("step").textContent = "Turn " + state.step + "/" + state.episode_steps;
  document.getElementById("agent").textContent = "agent: " + state.agent_name;
  document.getElementById("token-count").textContent = state.tokens.length;
  document.getElementById("max-moves").textContent = state.max_moves;
  document.getElementById("max-moves2").textContent = state.max_moves;
  document.getElementById("planned-count").textContent = state.planned_actions.length;

  const yourId = state.human_player, theirId = 1 - state.human_player;

  // Token table
  const tbody = document.getElementById("tokens-body");
  tbody.innerHTML = "";
  const remaining = state.max_moves - state.planned_actions.length;
  sortedTokens(state.tokens).forEach(t => {
    const tr = document.createElement("tr");
    tr.classList.add("tok");
    const kindClass = t.kind === "reinforce" ? "kind-r" : (t.kind === "attack_enemy" ? "kind-a" : "kind-n");
    const kindBadge = t.kind === "reinforce" ? `<span class="badge r">R</span>` : (t.kind === "attack_enemy" ? `<span class="badge a">A</span>` : `<span class="badge n">N</span>`);
    const [srcLabel, srcCls] = ownerLabel(t.src_pid, state.planets, yourId, theirId);
    const [tgtLabel, tgtCls] = ownerLabel(t.tgt_pid, state.planets, yourId, theirId);
    tr.innerHTML = `
      <td class="${kindClass}">${kindBadge}${t.kind}</td>
      <td>${t.src_pid} <span class="${srcCls}">${srcLabel}</span></td>
      <td>${t.tgt_pid} <span class="${tgtCls}">${tgtLabel}</span></td>
      <td><b>${t.ships}</b></td>
      <td>${t.features[0].toFixed(1)}</td>
      <td>${t.features[1].toFixed(0)}</td>
      <td>${t.features[5].toFixed(0)}</td>
      <td>${t.features[6].toFixed(0)}</td>
      <td>${t.features[7].toFixed(1)}</td>
      <td class="${t.features[8] > 0.5 ? 'f-flag' : ''}">${t.features[8].toFixed(0)}</td>
      <td>${t.features[9].toFixed(0)}</td>
      <td>${t.features[10].toFixed(0)}</td>
    `;
    if (remaining <= 0 || state.done) {
      tr.style.opacity = 0.4;
      tr.style.cursor = "not-allowed";
    } else {
      tr.addEventListener("click", () => selectToken(t.idx));
      tr.addEventListener("mouseenter", () => { hover = { src_pid: t.src_pid, tgt_pid: t.tgt_pid }; drawCanvas(); });
      tr.addEventListener("mouseleave", () => { hover = null; drawCanvas(); });
    }
    tbody.appendChild(tr);
  });

  drawCanvas();

  // Planned moves panel
  const plist = document.getElementById("planned-list");
  if (state.planned_actions.length === 0) {
    plist.innerHTML = `<span class="hint">No moves planned. Click a token to plan one, or click Commit/Stop to pass the turn.</span>`;
  } else {
    plist.innerHTML = state.planned_actions.map(m =>
      `<div class="move-row">launch ${m[2]} ships from planet ${m[0]} @ angle ${(+m[1]).toFixed(3)} rad</div>`
    ).join("");
  }

  if (state.done) {
    setBanner("Game over. Final scores: You " + state.scores[yourId] + " — Agent " + state.scores[theirId] + ". " + (state.scores[yourId] > state.scores[theirId] ? "YOU WIN." : (state.scores[yourId] === state.scores[theirId] ? "DRAW." : "AGENT WINS.")), true);
  } else {
    setBanner("", false);
  }
}

async function fetchState() {
  const r = await fetch("/state");
  state = await r.json();
  render();
}

async function selectToken(idx) {
  const r = await fetch("/select", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ idx }) });
  if (!r.ok) {
    const e = await r.json();
    setBanner("select error: " + (e.error || "unknown"), true);
    setTimeout(() => setBanner("", false), 2000);
    return;
  }
  state = await r.json();
  render();
}

async function commitTurn() {
  const r = await fetch("/commit", { method: "POST" });
  state = await r.json();
  render();
}

async function undoPlanned() {
  const r = await fetch("/undo", { method: "POST" });
  state = await r.json();
  render();
}

async function resetGame() {
  const r = await fetch("/reset", { method: "POST" });
  state = await r.json();
  render();
}

// --- Canvas click: pick source → pick target (must be a valid token edge) ---
function planetAtPixel(px, py) {
  if (!state) return null;
  // Hit-test in pixel space; small padding to make planet circles easy to click.
  const PAD = 4;
  let best = null;
  let bestDist = Infinity;
  for (const p of state.planets) {
    const cx = u2p(p[2]);
    const cy = u2p(p[3]);
    const radius = u2p(p[4]) + PAD;
    const d = Math.hypot(px - cx, py - cy);
    if (d <= radius && d < bestDist) {
      best = p;
      bestDist = d;
    }
  }
  return best;
}

cv.addEventListener("click", (e) => {
  if (!state || state.done) return;
  if (state.planned_actions.length >= state.max_moves) {
    setBanner("Max moves reached for this turn — commit or undo.", true);
    setTimeout(() => setBanner("", false), 1800);
    return;
  }
  const rect = cv.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  const planet = planetAtPixel(px, py);
  if (!planet) {
    selectedSrc = null;
    drawCanvas();
    return;
  }
  const pid = planet[0];
  const owner = planet[1];

  if (selectedSrc === null) {
    // Pick a source — only your own planets are valid sources.
    if (owner !== state.human_player) {
      setBanner("Select one of YOUR planets as source.", true);
      setTimeout(() => setBanner("", false), 1500);
      return;
    }
    // Source must have at least one valid outgoing token.
    if (!state.tokens.some(t => t.src_pid === pid)) {
      setBanner("That source has no valid moves this turn.", true);
      setTimeout(() => setBanner("", false), 1500);
      return;
    }
    selectedSrc = pid;
    drawCanvas();
    return;
  }

  // Source already selected.
  if (pid === selectedSrc) {
    // Clicking the source again deselects.
    selectedSrc = null;
    drawCanvas();
    return;
  }
  const tok = state.tokens.find(t => t.src_pid === selectedSrc && t.tgt_pid === pid);
  if (!tok) {
    setBanner(`No valid token for #${selectedSrc} → #${pid}. (out of radar / illegal trajectory / not enough ships)`, true);
    setTimeout(() => setBanner("", false), 2200);
    return;
  }
  // Valid: send the token index.
  const idx = tok.idx;
  selectedSrc = null;
  selectTokenWithRedraw(idx);
});

async function selectTokenWithRedraw(idx) {
  await selectToken(idx);
}

document.getElementById("commit-btn").addEventListener("click", () => {
  selectedSrc = null;
  commitTurn();
});
document.getElementById("undo-btn").addEventListener("click", () => {
  selectedSrc = null;
  undoPlanned();
});
document.getElementById("reset-btn").addEventListener("click", () => {
  selectedSrc = null;
  resetGame();
});

// Sortable header
document.querySelectorAll("#tok-head th").forEach(th => {
  th.addEventListener("click", () => {
    const k = th.getAttribute("data-k");
    const isFeat = th.hasAttribute("data-feat");
    if (sortKey === k && sortFeat === isFeat) sortDir = -sortDir;
    else { sortKey = k; sortFeat = isFeat; sortDir = 1; }
    render();
  });
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Enter") commitTurn();
  if (e.key === "u" || e.key === "U") undoPlanned();
});

fetchState();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    new_env()
    app.run(host="127.0.0.1", port=5001, debug=False)
