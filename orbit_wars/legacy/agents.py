import math
import random
from copy import deepcopy

from harness import (
    GameView,
    FEATURE_ETA,
    FEATURE_SHIPS_NEEDED,
    FEATURE_TGT_PRODUCTION,
    FEATURE_TGT_EXPIRY,
)
from action_space import MAX_MODEL_MOVES


HOARD_WINDOW = 20       # turns_left < this → skip attacks
SAFETY_MARGIN = 1       # extra ships on top of harness ships_needed


def _get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _mutable_obs(obs):
    """Copy the observation fields GameView needs into plain mutable lists."""
    out = {}
    for key in (
        "step",
        "player",
        "planets",
        "fleets",
        "angular_velocity",
        "initial_planets",
        "comets",
        "comet_planet_ids",
        "next_fleet_id",
        "remainingOverageTime",
    ):
        value = _get(obs, key, None)
        if value is not None:
            out[key] = deepcopy(value)
    return out


def random_agent(obs):
    """Uniform random legal edge policy for end-to-end pipeline smoke tests."""
    view = GameView(obs)
    legal_edges = list(zip(*view.legal_mask.nonzero()))
    if not legal_edges:
        return []

    random.shuffle(legal_edges)
    for src_slot, tgt_slot in legal_edges:
        src_pid = int(view.planet_ids[src_slot])
        src = view.planets_by_id.get(src_pid)
        if src is None:
            continue
        ships = max(1, int(src[5]) // 2)
        action = view.to_action(int(src_slot), int(tgt_slot), ships)
        if action is not None:
            return [action]
    return []


def random_model_space_agent(obs, max_moves=MAX_MODEL_MOVES):
    """Random smoke-test agent for the model's edge+stop action space."""
    view = GameView(_mutable_obs(obs))
    moves = []
    for _ in range(max_moves):
        action_mask = view.action_mask(SAFETY_MARGIN)
        choices = list(zip(*action_mask.nonzero()))
        if not choices:
            break
        src_slot, tgt_slot = random.choice(choices)
        ships = view.deterministic_ship_count(int(src_slot), int(tgt_slot), SAFETY_MARGIN)
        action = view.apply_planned_move(int(src_slot), int(tgt_slot), ships)
        if action is None:
            break
        moves.append(action)
    return moves


def model_agent_actions(model, obs, max_moves=MAX_MODEL_MOVES, deterministic=False,
                        view: "GameView | None" = None):
    """Decode a model policy into up to `max_moves` game actions.

    Uses a single persistent `GameView` per turn — sub-moves mutate it via
    `apply_planned_move`, avoiding full rebuilds. Pass `view=` to reuse a
    view already updated to the current obs (skip the cold-build cost).
    The stop action is the final action-probability slot.
    """
    import torch

    if view is None:
        view = GameView(_mutable_obs(obs))
    moves = []
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_moves):
            action_mask = view.action_mask(SAFETY_MARGIN)
            if not action_mask.any():
                break

            edge_features = torch.as_tensor(view.edge_features).unsqueeze(0)
            legal_mask = torch.as_tensor(view.legal_mask).unsqueeze(0)
            edge_mask = torch.as_tensor(action_mask).unsqueeze(0)
            with torch.no_grad():
                out = model(edge_features, legal_mask, edge_mask)
            policy = out["action_policy"][0]

            if deterministic:
                action_idx = int(torch.argmax(policy).item())
            else:
                action_idx = int(torch.multinomial(policy, 1).item())

            stop_idx = view.n_max * view.n_max
            if action_idx == stop_idx:
                break

            src_slot = action_idx // view.n_max
            tgt_slot = action_idx % view.n_max
            ships = view.deterministic_ship_count(src_slot, tgt_slot, SAFETY_MARGIN)
            action = view.apply_planned_move(src_slot, tgt_slot, ships)
            if action is None:
                break
            moves.append(action)
    finally:
        model.train(was_training)
    return moves


class StatefulModelAgent:
    """Persistent-view wrapper for Kaggle's stateless agent(obs) interface.

    Holds a `GameView` across turns and updates it via `update_from_obs`,
    so the expensive cold rebuild (radar-simulating every in-flight fleet)
    only happens on the first turn. Call the instance as `agent(obs)`.
    """

    def __init__(self, model, max_moves: int = MAX_MODEL_MOVES, deterministic: bool = False):
        self.model = model
        self.max_moves = max_moves
        self.deterministic = deterministic
        self._view: "GameView | None" = None

    def __call__(self, obs):
        mut = _mutable_obs(obs)
        if self._view is None:
            self._view = GameView(mut)
        else:
            self._view.update_from_obs(mut)
        return model_agent_actions(
            self.model, mut, max_moves=self.max_moves,
            deterministic=self.deterministic, view=self._view,
        )

    def reset(self):
        self._view = None


# Backwards-compatible name for older smoke commands.
random_bucket_agent = random_model_space_agent


def nearest_planet_sniper(obs):
    """Baseline agent from the Kaggle tutorial.

    Each owned planet fires max(target.ships + 1, 20) ships at its nearest
    non-owned planet.
    """
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

    my_planets = [p for p in raw_planets if p[1] == player]
    targets = [p for p in raw_planets if p[1] != player]
    if not targets:
        return moves

    for mine in my_planets:
        mid, _mowner, mx, my_, _mr, mships, _mprod = mine
        nearest = None
        min_dist = float("inf")
        for t in targets:
            tx, ty = t[2], t[3]
            dist = math.hypot(mx - tx, my_ - ty)
            if dist < min_dist:
                min_dist = dist
                nearest = t
        if nearest is None:
            continue
        tships = nearest[5]
        ships_needed = max(tships + 1, 20)
        if mships >= ships_needed:
            angle = math.atan2(nearest[3] - my_, nearest[2] - mx)
            moves.append([mid, angle, ships_needed])
    return moves


def _first_flip_eta(view, tgt_pid, player):
    """Step at which this (my) planet flips to non-me under current threats, or None."""
    planet = view.planets_by_id[tgt_pid]
    if planet[1] != player:
        return None
    ships = int(planet[5])
    prod = int(planet[6])
    owner = planet[1]
    last_t = 0
    for ar in sorted(view.threats.get(tgt_pid, []), key=lambda t: t["eta"]):
        eta = ar["eta"]
        ships += prod * max(0, eta - last_t)
        if ar["owner"] == owner:
            ships += ar["ships"]
        else:
            ships -= ar["ships"]
            if ships < 0:
                return eta
        last_t = eta
    return None


def heuristic_agent(obs, max_moves=MAX_MODEL_MOVES):
    """Rule-based policy acting in (src, tgt, ships) space via the harness.

    Priority: defend falling planets → expand by ROI → hoard endgame.

    The teacher is capped to the same turn shape as the model: at most
    `MAX_MODEL_MOVES` edge decisions. Candidate edges use `action_mask`, not
    the cheaper `legal_mask`, so BC labels line up with the model's
    radar-validated action space.
    """
    view = GameView(obs)
    edges = view.edge_features
    action_mask = view.action_mask(SAFETY_MARGIN)
    n = view.n_max
    turns_left = view.turns_left
    player = view.player

    my_slots = []
    for slot in range(n):
        pid = int(view.planet_ids[slot])
        if pid >= 0 and view.planets_by_id[pid][1] == player:
            my_slots.append(slot)
    my_set = set(my_slots)
    if not my_slots:
        return []

    available = {
        s: float(view.planets_by_id[int(view.planet_ids[s])][5]) for s in my_slots
    }
    defended = set()
    claimed_attacks = set()
    moves = []

    # --- Phase 1: DEFEND ---
    # For each of my planets projected to flip, pick one reinforcer that
    # arrives before the flip and carries enough ships to flip the outcome.
    for tgt_slot in my_slots:
        tgt_pid = int(view.planet_ids[tgt_slot])
        flip_eta = _first_flip_eta(view, tgt_pid, player)
        if flip_eta is None:
            continue
        deficit = -view._project_garrison(tgt_pid, flip_eta)
        if deficit <= 0:
            continue
        need = int(math.ceil(deficit)) + SAFETY_MARGIN

        best = None
        for src_slot in my_slots:
            if src_slot == tgt_slot:
                continue
            if not action_mask[src_slot, tgt_slot]:
                continue
            eta = float(edges[src_slot, tgt_slot, FEATURE_ETA])
            if eta <= 0 or eta > flip_eta:
                continue
            if available[src_slot] < need:
                continue
            if best is None or eta < best[0]:
                best = (eta, src_slot)
        if best is None:
            continue
        _, src_slot = best
        action = view.to_action(src_slot, tgt_slot, need)
        if action is None:
            continue
        moves.append(action)
        if len(moves) >= max_moves:
            return moves
        available[src_slot] -= need
        defended.add(tgt_slot)

    # --- Phase 3 (guard): HOARD — skip expansion in the final stretch ---
    if turns_left < HOARD_WINDOW:
        return moves[:max_moves]

    # --- Phase 2: EXPAND ---
    # Refuse to launch offense from a doomed planet we couldn't defend.
    doomed = set()
    for slot in my_slots:
        if slot in defended:
            continue
        if _first_flip_eta(view, int(view.planet_ids[slot]), player) is not None:
            doomed.add(slot)

    candidates = []
    for src_slot in my_slots:
        if src_slot in doomed:
            continue
        if available[src_slot] <= 1:
            continue
        for tgt_slot in range(n):
            if tgt_slot == src_slot or tgt_slot in my_set:
                continue
            if int(view.planet_ids[tgt_slot]) < 0:
                continue
            if not action_mask[src_slot, tgt_slot]:
                continue
            e = edges[src_slot, tgt_slot]
            eta = float(e[FEATURE_ETA])
            ships_needed = float(e[FEATURE_SHIPS_NEEDED])
            prod = float(e[FEATURE_TGT_PRODUCTION])
            expiry = float(e[FEATURE_TGT_EXPIRY])
            if eta <= 0:
                continue
            hold_time = min(expiry, turns_left - eta)
            if hold_time <= 0:
                continue
            if ships_needed + SAFETY_MARGIN > available[src_slot]:
                continue
            score = prod * hold_time / (ships_needed + eta + 1.0)
            candidates.append((score, src_slot, tgt_slot, ships_needed))

    candidates.sort(reverse=True, key=lambda c: c[0])

    for _score, src_slot, tgt_slot, ships_needed in candidates:
        if tgt_slot in claimed_attacks:
            continue
        need = int(math.ceil(ships_needed)) + SAFETY_MARGIN
        if available[src_slot] < need:
            continue
        action = view.to_action(src_slot, tgt_slot, need)
        if action is None:
            continue
        moves.append(action)
        if len(moves) >= max_moves:
            return moves
        available[src_slot] -= need
        claimed_attacks.add(tgt_slot)

    return moves[:max_moves]
