import math
from flask import Flask, jsonify, request, render_template_string

from kaggle_environments import make

from agents import nearest_planet_sniper, heuristic_agent
from harness import GameView
from radar import Radar
import targeting as T

HUMAN = 0
AGENT = 1
AGENT_FN = heuristic_agent

app = Flask(__name__)
_env = None


def new_env():
    global _env
    _env = make("orbit_wars", debug=False)
    _env.reset()
    # First step populates the world (planets are empty at step 0).
    _env.step([[], []])


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
    obs = _env.state[0].observation
    threats = T.threats_per_planet(
        obs["fleets"], obs["planets"], obs["initial_planets"],
        obs["angular_velocity"], obs["step"],
    )
    ring = T.ring_order(obs["planets"], obs["initial_planets"])
    return {
        "step": obs["step"],
        "episode_steps": _env.configuration.episodeSteps,
        "planets": obs["planets"],
        "initial_planets": obs["initial_planets"],
        "fleets": obs["fleets"],
        "angular_velocity": obs["angular_velocity"],
        "comet_planet_ids": obs["comet_planet_ids"],
        "human_player": HUMAN,
        "scores": scores(obs),
        "done": _env.done,
        "statuses": [_env.state[i].status for i in range(2)],
        "threats": threats,
        "ring": ring,
    }


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/state")
def state():
    return jsonify(snapshot())


@app.route("/aim", methods=["POST"])
def aim():
    """Server-side human aiming using the same targeting/radar as GameView."""
    if _env.done:
        return jsonify({"ok": False, "error": "game_done"})
    data = request.json or {}
    try:
        src_pid = int(data.get("src"))
        tgt_pid = int(data.get("tgt"))
        ships = int(data.get("ships"))
    except Exception:
        return jsonify({"ok": False, "error": "bad_request"})

    obs = _env.state[HUMAN].observation
    view = GameView(obs)
    src_slot = view.slot_of.get(src_pid)
    tgt_slot = view.slot_of.get(tgt_pid)
    if src_slot is None or tgt_slot is None:
        return jsonify({"ok": False, "error": "unknown_planet"})
    action = view.to_action(src_slot, tgt_slot, ships)
    if action is None:
        return jsonify({"ok": False, "error": "no_intercept"})

    src = view.planets_by_id.get(src_pid)
    hit = Radar(obs).simulate_launch(src, action[1], action[2]) if src else None
    ok = bool(hit and hit.hit_planet and hit.target_id == tgt_pid)
    intercept = view._lead_intercept((src[2], src[3]), tgt_pid, action[2], src_radius=src[4]) if src else None
    return jsonify({
        "ok": True,
        "legal": ok,
        "action": action,
        "angle": action[1],
        "ships": action[2],
        "eta": int(hit.eta) if hit and hit.eta is not None else (intercept["eta"] if intercept else None),
        "px": intercept["pred_x"] if intercept else None,
        "py": intercept["pred_y"] if intercept else None,
        "hit_kind": hit.kind if hit else None,
        "hit_target_id": hit.target_id if hit else None,
    })


@app.route("/action", methods=["POST"])
def action():
    if _env.done:
        return jsonify(snapshot())
    human_moves = request.json.get("moves", []) if request.is_json else []
    # Basic sanity filter — don't pass obvious garbage to env.
    cleaned = []
    obs0 = _env.state[0].observation
    owned_ids = {p[0] for p in obs0["planets"] if p[1] == HUMAN}
    garrison = {p[0]: p[5] for p in obs0["planets"]}
    for m in human_moves:
        try:
            pid, angle, ships = int(m[0]), float(m[1]), int(m[2])
        except Exception:
            continue
        if pid not in owned_ids or ships < 1:
            continue
        if ships > garrison.get(pid, 0):
            continue
        cleaned.append([pid, angle, ships])
    agent_obs = _env.state[AGENT].observation
    agent_moves = AGENT_FN(agent_obs)
    _env.step([cleaned, agent_moves])
    return jsonify(snapshot())


@app.route("/reset", methods=["POST"])
def reset():
    new_env()
    return jsonify(snapshot())


PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Orbit Wars — play vs. sniper</title>
<style>
  * { box-sizing: border-box; }
  body { font: 14px/1.4 system-ui, sans-serif; background: #0a0a14; color: #e6e6ee; margin: 0; padding: 20px; }
  h1 { margin: 0 0 12px; font-size: 18px; font-weight: 600; }
  .wrap { display: flex; gap: 16px; align-items: flex-start; }
  #canvas { background: #050510; border: 1px solid #222; cursor: crosshair; }
  .panel { width: 260px; flex-shrink: 0; }
  .ring-panel { width: 280px; flex-shrink: 0; font-size: 12px; }
  .row { margin: 10px 0; }
  .scores { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip { padding: 6px 10px; border-radius: 4px; font-weight: 600; font-size: 13px; }
  .chip.you { background: #1d4ed8; }
  .chip.them { background: #b91c1c; }
  .chip.step { background: #333; }
  button { background: #1f2937; color: #e6e6ee; border: 1px solid #374151; padding: 7px 10px; border-radius: 4px; cursor: pointer; font: inherit; font-size: 13px; }
  button:hover { background: #374151; }
  button.primary { background: #059669; border-color: #047857; }
  button.primary:hover { background: #047857; }
  button.active { background: #7c3aed; border-color: #6d28d9; }
  .presets, .aim-modes { display: flex; gap: 6px; }
  .presets button, .aim-modes button { flex: 1; }
  .hint { color: #888; font-size: 12px; }
  .moves { max-height: 160px; overflow: auto; border: 1px solid #222; padding: 6px; border-radius: 4px; }
  .move { padding: 4px 6px; display: flex; justify-content: space-between; font-size: 12px; }
  .move button { padding: 0 6px; font-size: 11px; }
  .banner { padding: 10px; background: #422; border: 1px solid #833; border-radius: 4px; display: none; }
  kbd { background: #222; padding: 1px 5px; border-radius: 3px; font-family: ui-monospace, monospace; font-size: 11px; }
  table.ring { width: 100%; border-collapse: collapse; }
  table.ring th { text-align: left; color: #888; font-weight: 500; padding: 3px 4px; font-size: 11px; border-bottom: 1px solid #222; }
  table.ring td { padding: 3px 4px; border-bottom: 1px solid #141428; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; vertical-align: middle; margin-right: 4px; }
  .dot.you { background: #3b82f6; }
  .dot.them { background: #ef4444; }
  .dot.neutral { background: #6b7280; }
  .pos { color: #34d399; }
  .neg { color: #f87171; }
  .dim { color: #666; }
</style>
</head>
<body>
<h1>Orbit Wars — you (blue) vs. Nearest-Planet Sniper (red)</h1>
<div class="wrap">
  <canvas id="canvas" width="720" height="720"></canvas>
  <div class="panel">
    <div class="row scores">
      <span class="chip you">You: <span id="score-you">0</span></span>
      <span class="chip them">Bot: <span id="score-them">0</span></span>
      <span class="chip step">T<span id="step">0</span>/<span id="max-step">500</span></span>
    </div>
    <div class="row">
      <div class="hint">Send size (of source garrison):</div>
      <div class="presets">
        <button data-preset="0.25">25%</button>
        <button data-preset="0.5" class="active">50%</button>
        <button data-preset="1.0">100%</button>
      </div>
    </div>
    <div class="row">
      <div class="hint">Aim mode:</div>
      <div class="aim-modes">
        <button data-aim="lead" class="active">Lead (predict)</button>
        <button data-aim="raw">Raw (current)</button>
      </div>
    </div>
    <div class="row">
      <label style="display:flex; align-items:center; gap:6px; cursor:pointer;">
        <input type="checkbox" id="realtime-toggle">
        <span>Realtime mode (agent's 1 s + 2 s overage)</span>
      </label>
      <div id="timer-wrap" style="margin-top:6px; display:none;">
        <div style="display:flex; justify-content:space-between; font-size:11px; color:#aaa;">
          <span>Turn: <span id="timer-turn">1.00</span>s</span>
          <span>Overage bank: <span id="timer-bank">2.00</span>s</span>
        </div>
        <div style="height:6px; background:#222; border-radius:3px; overflow:hidden; margin-top:3px;">
          <div id="timer-bar" style="height:100%; width:100%; background:#34d399; transition:width 0.05s linear;"></div>
        </div>
      </div>
    </div>
    <div class="row">
      <div class="hint">
        Click <b style="color:#60a5fa">owned</b> planet → any planet.
        <kbd>Esc</kbd> cancel · <kbd>Space</kbd> end turn · <kbd>R</kbd> reset
      </div>
    </div>
    <div class="row">
      <button id="end-turn" class="primary">End Turn ▶</button>
      <button id="reset">Reset</button>
    </div>
    <div class="row">
      <div class="hint">Pending moves:</div>
      <div id="moves" class="moves"></div>
    </div>
    <div class="row">
      <div id="banner" class="banner"></div>
    </div>
    <div class="row" id="hover-info" style="min-height: 60px; border: 1px solid #222; padding: 6px; border-radius: 4px; font-size: 12px;">
      <span class="hint">Hover over a target with a source selected to see lead-targeting info.</span>
    </div>
  </div>
  <div class="ring-panel">
    <div class="hint" style="margin-bottom:6px">Outer ring (static, CCW from +x):</div>
    <table class="ring" id="ring-table">
      <thead><tr><th>#</th><th>Own</th><th>Ships</th><th>p</th><th>In</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<script>
const BOARD = 100;
const CANVAS = 720;
const SCALE = CANVAS / BOARD;
const SUN = { x: 50, y: 50, r: 10 };
const MAX_SPEED = 6.0;

const COLORS = {
  you: "#3b82f6",
  them: "#ef4444",
  neutral: "#6b7280",
  sun: "#fbbf24",
  comet: "#f97316",
  preview: "#a78bfa",
  rawAim: "#f87171",
  leadAim: "#34d399",
};

let state = null;
let preset = 0.5;
let aimMode = "lead";
let selectedSource = null;
let mouse = null;
let hoverPlanetId = null;
let pending = [];
let aimSeq = 0;
let aimCache = null;
let cometSet = new Set();
let initById = new Map();

// Realtime mode
const PER_TURN_SEC = 1.0;
const OVERAGE_TOTAL_SEC = 2.0;
let realtimeOn = false;
let overageBank = OVERAGE_TOTAL_SEC;
let turnStartMs = null;
let timerRaf = null;
let turnInFlight = false;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

function u2p(u) { return u * SCALE; }

// ---------- Targeting math (port of targeting.py) ----------
function fleetSpeed(ships) {
  if (ships <= 1) return 1.0;
  return 1.0 + (MAX_SPEED - 1.0) * Math.pow(Math.log(ships) / Math.log(1000), 1.5);
}
function isOrbiting(ip) {
  const [_pid, _o, ix, iy, r] = ip;
  return (Math.hypot(ix - SUN.x, iy - SUN.y) + r) < 50.0;
}
function orbitParams(ip) {
  const [_pid, _o, ix, iy] = ip;
  const dx = ix - SUN.x, dy = iy - SUN.y;
  return { r: Math.hypot(dx, dy), phase0: Math.atan2(dy, dx) };
}
function futurePos(ip, currentStep, tAhead, av) {
  if (!isOrbiting(ip)) return { x: ip[2], y: ip[3] };
  const { r, phase0 } = orbitParams(ip);
  const phi = phase0 + (currentStep + tAhead - 1) * av;
  return { x: SUN.x + r * Math.cos(phi), y: SUN.y + r * Math.sin(phi) };
}
function leadIntercept(src, tgtInit, ships, av, step) {
  const speed = fleetSpeed(ships);
  let tx = futurePos(tgtInit, step, 0, av).x;
  let ty = futurePos(tgtInit, step, 0, av).y;
  let eta = Math.max(1.0, Math.hypot(tx - src.x, ty - src.y) / speed);
  let ok = false;
  for (let i = 0; i < 30; i++) {
    const fp = futurePos(tgtInit, step, eta, av);
    const newEta = Math.hypot(fp.x - src.x, fp.y - src.y) / speed;
    if (Math.abs(newEta - eta) < 0.05) { eta = newEta; ok = true; break; }
    eta = newEta;
  }
  if (!ok) return null;
  const etaI = Math.max(1, Math.ceil(eta));
  const fp = futurePos(tgtInit, step, etaI, av);
  return {
    angle: Math.atan2(fp.y - src.y, fp.x - src.x),
    eta: etaI, px: fp.x, py: fp.y, speed,
  };
}
function requiredShips(target, eta) {
  const ships = target[5], prod = target[6];
  return Math.floor(ships + prod * eta + 1);
}

function selectedShipCount(src) {
  const pledged = pending.filter(m => m.from === src[0]).reduce((a, m) => a + m.ships, 0);
  const avail = Math.max(0, src[5] - pledged);
  return { avail, ships: Math.max(1, Math.floor(avail * preset)) };
}

async function requestAim(srcId, tgtId, ships) {
  const seq = ++aimSeq;
  const res = await fetch("/aim", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ src: srcId, tgt: tgtId, ships })
  });
  const data = await res.json();
  if (seq === aimSeq) {
    aimCache = { srcId, tgtId, ships, data };
    draw();
    updateHoverInfo();
  }
  return data;
}

function cachedAim(srcId, tgtId, ships) {
  if (aimCache && aimCache.srcId === srcId && aimCache.tgtId === tgtId && aimCache.ships === ships) {
    return aimCache.data;
  }
  return null;
}

async function refresh(newState) {
  if (newState) state = newState;
  else state = await (await fetch("/state")).json();
  aimCache = null;
  cometSet = new Set(state.comet_planet_ids || []);
  initById = new Map((state.initial_planets || []).map(p => [p[0], p]));
  // Validate carried-over selection: drop if planet is no longer ours / gone.
  if (selectedSource != null) {
    const src = state.planets.find(p => p[0] === selectedSource);
    if (!src || src[1] !== state.human_player || src[5] < 1) {
      selectedSource = null;
    }
  }
  document.getElementById("score-you").textContent = state.scores[0] ?? 0;
  document.getElementById("score-them").textContent = state.scores[1] ?? 0;
  document.getElementById("step").textContent = state.step;
  document.getElementById("max-step").textContent = state.episode_steps;
  const banner = document.getElementById("banner");
  if (state.done) {
    const you = state.scores[0] ?? 0, them = state.scores[1] ?? 0;
    const msg = you > them ? "You win!" : (you < them ? "Bot wins." : "Tie.");
    banner.textContent = `Game over — ${msg} (you ${you} / bot ${them})`;
    banner.style.display = "block";
  } else {
    banner.style.display = "none";
  }
  draw();
  renderPending();
  renderRing();
  updateHoverInfo();
}

function threatSummary(pid) {
  const list = (state.threats && state.threats[pid]) || [];
  let mine = 0, theirs = 0, minEta = Infinity;
  for (const t of list) {
    if (t.owner === state.human_player) mine += t.ships;
    else theirs += t.ships;
    if (t.eta < minEta) minEta = t.eta;
  }
  return { mine, theirs, net: mine - theirs, nextEta: minEta === Infinity ? null : minEta, list };
}

function planetAt(u_x, u_y) {
  for (const p of state.planets) {
    const [id, owner, x, y, r] = p;
    if (Math.hypot(x - u_x, y - u_y) <= r + 0.5) return p;
  }
  return null;
}

function colorFor(owner) {
  if (owner === 0) return COLORS.you;
  if (owner === 1) return COLORS.them;
  return COLORS.neutral;
}

function draw() {
  ctx.fillStyle = "#050510";
  ctx.fillRect(0, 0, CANVAS, CANVAS);

  // Sun
  ctx.beginPath();
  ctx.arc(u2p(SUN.x), u2p(SUN.y), u2p(SUN.r), 0, Math.PI * 2);
  const g = ctx.createRadialGradient(u2p(SUN.x), u2p(SUN.y), 0, u2p(SUN.x), u2p(SUN.y), u2p(SUN.r));
  g.addColorStop(0, "#fde68a");
  g.addColorStop(1, "#b45309");
  ctx.fillStyle = g;
  ctx.fill();

  // Planets
  for (const p of state.planets) {
    const [id, owner, x, y, r, ships, prod] = p;
    const isComet = cometSet.has(id);
    ctx.beginPath();
    ctx.arc(u2p(x), u2p(y), u2p(r), 0, Math.PI * 2);
    ctx.fillStyle = colorFor(owner);
    ctx.fill();
    if (isComet) {
      ctx.strokeStyle = COLORS.comet;
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (selectedSource !== null && id === selectedSource) {
      ctx.beginPath();
      ctx.arc(u2p(x), u2p(y), u2p(r) + 5, 0, Math.PI * 2);
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    // ship count
    ctx.fillStyle = "#fff";
    ctx.font = "11px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`${ships}`, u2p(x), u2p(y));
    // production badge
    ctx.fillStyle = "#aaa";
    ctx.font = "9px system-ui";
    ctx.fillText(`p${prod}`, u2p(x), u2p(y) + u2p(r) + 8);
    // incoming-threat badge
    const th = threatSummary(id);
    if (th.list.length) {
      const labelRed = th.theirs > 0 ? `-${th.theirs}` : "";
      const labelGrn = th.mine > 0 ? `+${th.mine}` : "";
      const etaStr = th.nextEta != null ? ` in ${th.nextEta}t` : "";
      ctx.font = "10px system-ui";
      ctx.textAlign = "left";
      const bx = u2p(x) + u2p(r) + 3;
      let by = u2p(y) - u2p(r) - 2;
      if (th.theirs > 0) {
        ctx.fillStyle = "#f87171";
        ctx.fillText(labelRed + etaStr, bx, by);
        by += 11;
      }
      if (th.mine > 0) {
        ctx.fillStyle = "#34d399";
        ctx.fillText(labelGrn + etaStr, bx, by);
      }
    }
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
    ctx.lineTo(-len * 0.6, -len * 0.5);
    ctx.lineTo(-len * 0.6, len * 0.5);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    ctx.fillStyle = "#fff";
    ctx.font = "10px system-ui";
    ctx.textAlign = "left";
    ctx.fillText(`${ships}`, u2p(x) + 8, u2p(y) - 6);
  }

  // Hover preview (lead intercept)
  if (selectedSource !== null) {
    const src = state.planets.find(p => p[0] === selectedSource);
    if (src) {
      const hovered = hoverPlanetId != null ? state.planets.find(p => p[0] === hoverPlanetId) : null;
      if (hovered) {
        const { ships: nShips } = selectedShipCount(src);
        // Raw aim line (red)
        ctx.strokeStyle = COLORS.rawAim;
        ctx.setLineDash([2, 3]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(u2p(src[2]), u2p(src[3]));
        ctx.lineTo(u2p(hovered[2]), u2p(hovered[3]));
        ctx.stroke();
        // Lead aim from the same server-side GameView/Radar path used by agents.
        const intercept = nShips >= 1 ? cachedAim(src[0], hovered[0], nShips) : null;
        if (intercept && intercept.ok && intercept.px != null && intercept.py != null) {
          ctx.strokeStyle = intercept.legal ? COLORS.leadAim : COLORS.rawAim;
          ctx.setLineDash([5, 3]);
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(u2p(src[2]), u2p(src[3]));
          ctx.lineTo(u2p(intercept.px), u2p(intercept.py));
          ctx.stroke();
          // Predicted-pos marker
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.arc(u2p(intercept.px), u2p(intercept.py), 5, 0, Math.PI * 2);
          ctx.strokeStyle = intercept.legal ? COLORS.leadAim : COLORS.rawAim;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        ctx.setLineDash([]);
      } else if (mouse) {
        // Empty-space aim preview
        ctx.beginPath();
        ctx.moveTo(u2p(src[2]), u2p(src[3]));
        ctx.lineTo(mouse.x, mouse.y);
        ctx.strokeStyle = COLORS.preview;
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  // Pending move arrows (purple) with predicted impact dot
  for (const m of pending) {
    const src = state.planets.find(p => p[0] === m.from);
    if (!src) continue;
    const len = 100;
    ctx.beginPath();
    ctx.moveTo(u2p(src[2]), u2p(src[3]));
    ctx.lineTo(u2p(src[2]) + Math.cos(m.angle) * len, u2p(src[3]) + Math.sin(m.angle) * len);
    ctx.strokeStyle = "#a78bfa";
    ctx.lineWidth = 2;
    ctx.stroke();
    if (m.px != null) {
      ctx.beginPath();
      ctx.arc(u2p(m.px), u2p(m.py), 3, 0, Math.PI * 2);
      ctx.fillStyle = "#a78bfa";
      ctx.fill();
    }
  }
}

function renderPending() {
  const el = document.getElementById("moves");
  if (!pending.length) { el.innerHTML = '<span class="hint">(none)</span>'; return; }
  el.innerHTML = "";
  pending.forEach((m, i) => {
    const row = document.createElement("div");
    row.className = "move";
    row.innerHTML = `<span>#${m.from} → #${m.targetId ?? '?'} · ${m.ships} ships</span>`;
    const btn = document.createElement("button");
    btn.textContent = "×";
    btn.onclick = () => { pending.splice(i, 1); renderPending(); draw(); };
    row.appendChild(btn);
    el.appendChild(row);
  });
}

canvas.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  mouse = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  const p = planetAt(mouse.x / SCALE, mouse.y / SCALE);
  const nextHover = p ? p[0] : null;
  hoverPlanetId = nextHover;
  if (selectedSource != null && hoverPlanetId != null) {
    const src = state.planets.find(pl => pl[0] === selectedSource);
    if (src) {
      const { ships, avail } = selectedShipCount(src);
      if (avail >= 1 && !cachedAim(src[0], hoverPlanetId, ships)) {
        requestAim(src[0], hoverPlanetId, ships).catch(() => {});
      }
    }
  }
  draw();
  updateHoverInfo();
});
canvas.addEventListener("mouseleave", () => { mouse = null; hoverPlanetId = null; aimCache = null; draw(); updateHoverInfo(); });

canvas.addEventListener("click", async (e) => {
  const rect = canvas.getBoundingClientRect();
  const ux = (e.clientX - rect.left) / SCALE;
  const uy = (e.clientY - rect.top) / SCALE;
  const p = planetAt(ux, uy);
  if (!p) { selectedSource = null; draw(); return; }
  const [id, owner, x, y, r, ships] = p;
  if (selectedSource === null) {
    if (owner !== state.human_player) return;
    if (ships < 1) return;
    selectedSource = id;
    draw();
    updateHoverInfo();
    return;
  }
  const src = state.planets.find(pl => pl[0] === selectedSource);
  if (!src) { selectedSource = null; return; }
  if (src[0] === id) { selectedSource = null; draw(); return; }
  const { avail, ships: nShips } = selectedShipCount(src);
  if (nShips < 1 || avail < 1) { selectedSource = null; draw(); return; }
  // Lead vs raw
  let angle, px = null, py = null, eta = null;
  const serverAim = aimMode === "lead" ? await requestAim(src[0], id, nShips).catch(() => null) : null;
  if (serverAim && serverAim.ok) {
    angle = serverAim.angle; px = serverAim.px; py = serverAim.py; eta = serverAim.eta;
  } else {
    angle = Math.atan2(y - src[3], x - src[2]);
    px = x; py = y;
  }
  pending.push({ from: src[0], angle, ships: nShips, targetId: id, px, py, eta });
  selectedSource = null;
  draw();
  renderPending();
});

function updateHoverInfo() {
  const el = document.getElementById("hover-info");
  if (!state) { el.innerHTML = ""; return; }
  if (selectedSource == null || hoverPlanetId == null) {
    el.innerHTML = '<span class="hint">Hover over a target with a source selected to see lead-targeting info.</span>';
    return;
  }
  const src = state.planets.find(p => p[0] === selectedSource);
  const tgt = state.planets.find(p => p[0] === hoverPlanetId);
  if (!src || !tgt) { el.innerHTML = ""; return; }
  const { ships: nShips } = selectedShipCount(src);
  const intercept = cachedAim(src[0], tgt[0], nShips);
  const need = intercept && intercept.eta != null ? requiredShips(tgt, intercept.eta) : requiredShips(tgt, 0);
  const dist = Math.hypot(tgt[2]-src[2], tgt[3]-src[3]);
  let html = `<div><b>#${src[0]} → #${tgt[0]}</b> · dist ${dist.toFixed(1)}</div>`;
  html += `<div>sending <b>${nShips}</b> ships · speed ${fleetSpeed(nShips).toFixed(2)}/turn</div>`;
  if (intercept && intercept.ok && intercept.eta != null) {
    html += `<div>ETA: <b>${intercept.eta}</b> turns</div>`;
    html += `<div>need to capture: <b>${need}</b> (garrison ${tgt[5]} + prod ${tgt[6]}×${intercept.eta} + 1)</div>`;
    const verdict = nShips >= need ? `<span class="pos">likely capture</span>` : `<span class="neg">under-force by ${need - nShips}</span>`;
    html += `<div>${verdict}</div>`;
    if (!intercept.legal) {
      const hit = intercept.hit_target_id == null ? "nothing" : `#${intercept.hit_target_id}`;
      html += `<div class="neg">radar warning: first hit is ${hit} (${intercept.hit_kind ?? "unknown"})</div>`;
    }
  } else if (intercept && !intercept.ok) {
    html += `<div class="neg">radar: ${intercept.error ?? "no aim solution"}</div>`;
  } else {
    html += `<div class="dim">server radar pending...</div>`;
  }
  el.innerHTML = html;
}

function renderRing() {
  const tbody = document.querySelector("#ring-table tbody");
  tbody.innerHTML = "";
  const ring = state.ring || [];
  for (const pid of ring) {
    const p = state.planets.find(pp => pp[0] === pid);
    if (!p) continue;
    const [id, owner, x, y, r, ships, prod] = p;
    const th = threatSummary(id);
    const ownerClass = owner === 0 ? "you" : (owner === 1 ? "them" : "neutral");
    let inc = "";
    if (th.theirs > 0) inc += `<span class="neg">-${th.theirs}</span>`;
    if (th.theirs > 0 && th.mine > 0) inc += " ";
    if (th.mine > 0) inc += `<span class="pos">+${th.mine}</span>`;
    if (!inc) inc = '<span class="dim">—</span>';
    const etaStr = th.nextEta != null ? ` <span class="dim">(${th.nextEta}t)</span>` : "";
    tbody.insertAdjacentHTML("beforeend",
      `<tr><td>${id}</td><td><span class="dot ${ownerClass}"></span></td><td>${ships}</td><td>${prod}</td><td>${inc}${etaStr}</td></tr>`);
  }
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { selectedSource = null; draw(); }
  else if (e.key === " ") { e.preventDefault(); endTurn(); }
  else if (e.key === "r" || e.key === "R") { resetGame(); }
});

document.querySelectorAll(".presets button").forEach(b => {
  b.onclick = () => {
    preset = parseFloat(b.dataset.preset);
    document.querySelectorAll(".presets button").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    draw(); updateHoverInfo();
  };
});
document.querySelectorAll(".aim-modes button").forEach(b => {
  b.onclick = () => {
    aimMode = b.dataset.aim;
    document.querySelectorAll(".aim-modes button").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    draw();
  };
});

async function endTurn() {
  if (turnInFlight) return;
  turnInFlight = true;
  stopTimer();
  const moves = pending.map(m => [m.from, m.angle, m.ships]);
  pending = [];
  // NB: don't clear selectedSource here — let a mid-click flow carry into the
  // next turn. refresh() will validate the selection against the new state.
  try {
    const res = await fetch("/action", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ moves })
    });
    await refresh(await res.json());
  } finally {
    turnInFlight = false;
    if (realtimeOn && state && !state.done) startTimer();
  }
}

async function resetGame() {
  pending = [];
  selectedSource = null;
  stopTimer();
  overageBank = OVERAGE_TOTAL_SEC;
  const res = await fetch("/reset", { method: "POST" });
  await refresh(await res.json());
  if (realtimeOn && !state.done) startTimer();
}

// ---------- Realtime timer ----------
function startTimer() {
  stopTimer();
  document.getElementById("timer-wrap").style.display = "block";
  turnStartMs = performance.now();
  const tick = () => {
    const elapsed = (performance.now() - turnStartMs) / 1000;
    const turnBudget = PER_TURN_SEC;
    const perTurnLeft = Math.max(0, turnBudget - elapsed);
    const overPerTurn = Math.max(0, elapsed - turnBudget);
    const bankLeft = Math.max(0, overageBank - overPerTurn);
    const totalLeft = perTurnLeft + bankLeft;
    const totalBudget = turnBudget + overageBank;
    const frac = totalBudget > 0 ? totalLeft / totalBudget : 0;
    const bar = document.getElementById("timer-bar");
    bar.style.width = (frac * 100).toFixed(1) + "%";
    bar.style.background = perTurnLeft > 0 ? "#34d399" : (bankLeft > 0 ? "#fbbf24" : "#ef4444");
    document.getElementById("timer-turn").textContent = perTurnLeft.toFixed(2);
    document.getElementById("timer-bank").textContent = bankLeft.toFixed(2);
    if (totalLeft <= 0) {
      overageBank = 0;
      stopTimer();
      endTurn();
      return;
    }
    timerRaf = requestAnimationFrame(tick);
  };
  timerRaf = requestAnimationFrame(tick);
}
function stopTimer() {
  if (timerRaf != null) {
    cancelAnimationFrame(timerRaf);
    timerRaf = null;
  }
  // Commit elapsed overage when we stop mid-turn (voluntary End Turn).
  if (turnStartMs != null) {
    const elapsed = (performance.now() - turnStartMs) / 1000;
    const overPerTurn = Math.max(0, elapsed - PER_TURN_SEC);
    overageBank = Math.max(0, overageBank - overPerTurn);
    turnStartMs = null;
  }
}

document.getElementById("realtime-toggle").addEventListener("change", (e) => {
  realtimeOn = e.target.checked;
  if (realtimeOn) {
    if (state && !state.done) startTimer();
  } else {
    stopTimer();
    document.getElementById("timer-wrap").style.display = "none";
  }
});

document.getElementById("end-turn").onclick = endTurn;
document.getElementById("reset").onclick = resetGame;

refresh().then(() => { if (realtimeOn && !state.done) startTimer(); });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    new_env()
    app.run(host="127.0.0.1", port=5000, debug=False)
