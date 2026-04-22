"""Edge-centric harness for Orbit Wars.

Produces per-(source, target) feature tensors for a transformer policy.
One turn in → one tensor + mask out. The model reasons over edges; this
module turns observations into edges, and turns the model's choice back
into a legal game action.

Output contract:
    view.edge_features: (N_max, N_max, F) float32
    view.legal_mask:    (N_max, N_max) bool
    view.planet_ids:    (N_max,) int32  (-1 for padding slots)

Mask terminology:
  * legal_mask is an internal cheap candidate mask built with edge_features.
    It means "this src->tgt edge has a solvable intercept and passes the
    cheap geometry checks." It is deliberately not the final action contract.
  * action_mask(...) is the authoritative model/action mask. It starts from
    legal_mask, chooses the deterministic ship count for each edge, then uses
    Radar.simulate_launch to prove the fleet's first hit is the intended
    target. Training and inference should mask policy logits with action_mask.

Feature layout (indices are FEATURE_* constants below):
    0  eta                     turns for a fleet from src to intercept tgt
    1  ships_needed            attackers required to flip tgt at that eta
    2  kind_reinforce          bool: src mine, tgt mine
    3  kind_attack_enemy       bool: src mine, tgt enemy
    4  kind_attack_neutral     bool: src mine, tgt neutral
    5  src_ships               current garrison at src
    6  src_net_threat          enemy_inbound − friendly_inbound at src
    7  tgt_production          tgt production rate
    8  tgt_will_fall           bool: tgt (mine) is projected to flip
    9  tgt_expiry              comet turns-until-gone (999 sentinel otherwise)
    10 turns_left              episode turns remaining (broadcast scalar)

Design notes:
  * eta assumes fleet = src.ships (maximum feasible speed). to_action()
    recomputes the actual angle from whatever ship count the model picks.
  * legal_mask is intentionally only a cheap prefilter. The radar-backed
    action_mask is what prevents planet-clipping / wrong-first-hit launches
    from being sampled.
  * Non-mine rows of edge_features are left zero and mask=False since
    only owned planets can launch.
  * Comet orbit data is consumed minimally (for expiry and position).
    Full comet handling is out of scope for v1.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

import targeting as T
from radar import Radar, RadarHit


BOARD = 100.0
SUN = (50.0, 50.0)
SUN_R = 10.0
EPISODE_STEPS = 500
N_MAX_DEFAULT = 50

# Feature indices — stable, use these never the raw ints.
FEATURE_ETA = 0
FEATURE_SHIPS_NEEDED = 1
FEATURE_KIND_REINFORCE = 2
FEATURE_KIND_ATTACK_ENEMY = 3
FEATURE_KIND_ATTACK_NEUTRAL = 4
FEATURE_SRC_SHIPS = 5
FEATURE_SRC_NET_THREAT = 6
FEATURE_TGT_PRODUCTION = 7
FEATURE_TGT_WILL_FALL = 8
FEATURE_TGT_EXPIRY = 9
FEATURE_TURNS_LEFT = 10
FEATURE_DIM = 11

FEATURE_NAMES = [
    "eta", "ships_needed",
    "kind_reinforce", "kind_attack_enemy", "kind_attack_neutral",
    "src_ships", "src_net_threat",
    "tgt_production", "tgt_will_fall", "tgt_expiry",
    "turns_left",
]

# Suggested normalizers for the model's input layer (divide raw by these).
FEATURE_SCALES = np.array([
    50.0,   # eta
    100.0,  # ships_needed
    1.0, 1.0, 1.0,
    100.0,  # src_ships
    50.0,   # src_net_threat
    5.0,    # tgt_production
    1.0,
    100.0,  # tgt_expiry
    500.0,  # turns_left
], dtype=np.float32)

COMET_EXPIRY_SENTINEL = 999.0


def _get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _point_to_segment_distance(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)
    u = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    cx = ax + u * dx
    cy = ay + u * dy
    return math.hypot(px - cx, py - cy)


class GameView:
    """A single observation, wrapped into model-ready edge tensors."""

    def __init__(self, obs, n_max: int = N_MAX_DEFAULT):
        self.obs = obs
        self.n_max = n_max
        self.player = _get(obs, "player", 0)
        self.step = _get(obs, "step", 0)
        self.angular_velocity = _get(obs, "angular_velocity", 0.0)
        self.planets = [list(p) for p in _get(obs, "planets", []) or []]
        self.fleets = [list(f) for f in _get(obs, "fleets", []) or []]
        self.initial_planets = [list(p) for p in _get(obs, "initial_planets", []) or []]
        self.comet_planet_ids = set(_get(obs, "comet_planet_ids", []) or [])
        self.comets = list(_get(obs, "comets", []) or [])
        self.next_fleet_id = int(_get(obs, "next_fleet_id", 0) or 0)

        self.planets_by_id = {p[0]: p for p in self.planets}
        self.initial_by_id = {p[0]: p for p in self.initial_planets}
        self._comet_lookup = self._build_comet_lookup()

        # Slot assignment: first n_max planets in the obs order.
        self.planet_ids = np.full(n_max, -1, dtype=np.int32)
        self.slot_of: dict[int, int] = {}
        for slot, planet in enumerate(self.planets[:n_max]):
            self.planet_ids[slot] = planet[0]
            self.slot_of[planet[0]] = slot

        # Lazy caches for the radar, the action mask, and per-planet scalars.
        # Must be assigned before anything that might read them (e.g.
        # _populate_all_fleet_predictions calls _get_radar).
        self._radar: Optional[Radar] = None
        self._cached_action_mask: Optional[np.ndarray] = None
        self._cached_safety_margin: int = 1
        self._tgt_info: dict[int, dict] = {}
        self._src_info: dict[int, dict] = {}

        # Per-fleet radar predictions — cached across turns. For each fleet
        # we store the full RadarHit; carry-over fleets get their eta
        # decremented rather than re-simulated next turn.
        self._fleet_predictions: dict[int, RadarHit] = {}
        self._populate_all_fleet_predictions()
        self.threats = self._rebuild_threats_from_predictions()

        self.turns_left = max(0, EPISODE_STEPS - self.step)

        # Per-planet scalar caches used by the edge build. Kept as instance
        # attributes so incremental updates can patch individual slots without
        # rebuilding the whole 50×50 tensor.
        for slot, p in enumerate(self.planets[:self.n_max]):
            self._tgt_info[slot] = self._compute_tgt_info(slot)
            self._src_info[slot] = self._compute_src_info(slot)

        self.edge_features, self.legal_mask = self._build_edges()

    # ------------------------------------------------------------------
    # Public convenience
    # ------------------------------------------------------------------

    @property
    def my_planets(self):
        return [p for p in self.planets if p[1] == self.player]

    def to_action(self, src_slot: int, tgt_slot: int, ships: int) -> Optional[list]:
        """Turn the model's choice into `[from_planet_id, angle, ships]`.

        Recomputes the angle using the *actual* ship count (speed depends on
        ship count, so the intercept angle does too).  Returns None if the
        choice is invalid — caller should have masked these out already.
        """
        if not (0 <= src_slot < self.n_max and 0 <= tgt_slot < self.n_max):
            return None
        src_pid = int(self.planet_ids[src_slot])
        tgt_pid = int(self.planet_ids[tgt_slot])
        if src_pid < 0 or tgt_pid < 0 or src_pid == tgt_pid:
            return None
        src = self.planets_by_id.get(src_pid)
        if src is None or src[1] != self.player:
            return None
        ships = int(min(ships, src[5]))
        if ships <= 0:
            return None
        intercept = self._lead_intercept((src[2], src[3]), tgt_pid, ships, src_radius=src[4])
        if intercept is None:
            return None
        return [src_pid, float(intercept["angle"]), ships]

    def deterministic_ship_count(self, src_slot: int, tgt_slot: int, safety_margin: int = 1) -> int:
        """Default ship sizing for an edge: defense deficit or attack need."""
        if not (0 <= src_slot < self.n_max and 0 <= tgt_slot < self.n_max):
            return 0
        src_pid = int(self.planet_ids[src_slot])
        src = self.planets_by_id.get(src_pid)
        if src is None:
            return 0
        available = int(src[5])
        if available <= 0:
            return 0

        edge = self.edge_features[src_slot, tgt_slot]
        if edge[FEATURE_KIND_REINFORCE] > 0.5:
            tgt_pid = int(self.planet_ids[tgt_slot])
            flip_eta = self.first_flip_eta(tgt_pid)
            if flip_eta is None:
                need = 1
            else:
                deficit = -self._project_garrison(tgt_pid, flip_eta)
                need = max(1, int(math.ceil(deficit)) + safety_margin)
        else:
            need = max(1, int(math.ceil(float(edge[FEATURE_SHIPS_NEEDED]))) + safety_margin)
        return min(available, need)

    def action_mask(self, safety_margin: int = 1):
        """Authoritative mechanical mask for deterministic `(src, tgt)` actions.

        This is the mask the model should train/sample against.  `legal_mask`
        is only the cheap candidate mask created during feature construction;
        this method re-checks each candidate using deterministic ship sizing
        plus Radar.simulate_launch, and only keeps edges whose first hit is
        the intended target.

        Cached — first call does the full radar validation; subsequent calls
        return the cached mask (or an incrementally-patched version if
        `apply_planned_move` has run). Pass a different `safety_margin` to
        force a rebuild.
        """
        if (self._cached_action_mask is not None
                and self._cached_safety_margin == safety_margin):
            return self._cached_action_mask

        mask = np.zeros((self.n_max, self.n_max), dtype=bool)
        for src_slot, tgt_slot in zip(*self.legal_mask.nonzero()):
            if self._validate_mask_cell(int(src_slot), int(tgt_slot), safety_margin):
                mask[int(src_slot), int(tgt_slot)] = True
        self._cached_action_mask = mask
        self._cached_safety_margin = safety_margin
        return mask

    # ------------------------------------------------------------------
    # Edge-tensor construction
    # ------------------------------------------------------------------

    def _compute_tgt_info(self, slot: int) -> dict:
        pid = int(self.planet_ids[slot]) if slot < len(self.planet_ids) else -1
        if pid < 0:
            return {"production": 0.0, "expiry": COMET_EXPIRY_SENTINEL, "will_fall": False}
        planet = self.planets_by_id[pid]
        return {
            "production": float(planet[6]),
            "expiry": self._expiry_for(pid),
            "will_fall": self._will_fall_if_ignored(pid),
        }

    def _compute_src_info(self, slot: int) -> dict:
        pid = int(self.planet_ids[slot]) if slot < len(self.planet_ids) else -1
        if pid < 0:
            return {"ships": 0.0, "net_threat": 0.0}
        planet = self.planets_by_id[pid]
        return {
            "ships": float(planet[5]),
            "net_threat": self._net_threat_at(pid),
        }

    def _build_edges(self):
        features = np.zeros((self.n_max, self.n_max, FEATURE_DIM), dtype=np.float32)
        legal = np.zeros((self.n_max, self.n_max), dtype=bool)

        for src_slot, src in enumerate(self.planets[:self.n_max]):
            _, src_owner, _, _, _, src_ships, _ = src
            if src_owner != self.player or src_ships <= 0:
                continue
            for tgt_slot, tgt in enumerate(self.planets[:self.n_max]):
                if src_slot == tgt_slot:
                    continue
                self._fill_edge_cell(src_slot, src, tgt_slot, tgt, features, legal)

        return features, legal

    def _fill_edge_cell(self, src_slot, src, tgt_slot, tgt, features, legal):
        """Populate one (src,tgt) cell in edge_features + legal_mask."""
        pair = self._compute_pair(src, tgt)
        if pair is None:
            features[src_slot, tgt_slot] = 0.0
            legal[src_slot, tgt_slot] = False
            return

        f = features[src_slot, tgt_slot]
        f[:] = 0.0
        f[FEATURE_ETA] = pair["eta"]
        f[FEATURE_SHIPS_NEEDED] = pair["ships_needed"]

        tgt_owner = tgt[1]
        if tgt_owner == self.player:
            f[FEATURE_KIND_REINFORCE] = 1.0
        elif tgt_owner == -1:
            f[FEATURE_KIND_ATTACK_NEUTRAL] = 1.0
        else:
            f[FEATURE_KIND_ATTACK_ENEMY] = 1.0

        f[FEATURE_SRC_SHIPS] = self._src_info[src_slot]["ships"]
        f[FEATURE_SRC_NET_THREAT] = self._src_info[src_slot]["net_threat"]
        f[FEATURE_TGT_PRODUCTION] = self._tgt_info[tgt_slot]["production"]
        f[FEATURE_TGT_WILL_FALL] = 1.0 if self._tgt_info[tgt_slot]["will_fall"] else 0.0
        f[FEATURE_TGT_EXPIRY] = self._tgt_info[tgt_slot]["expiry"]
        f[FEATURE_TURNS_LEFT] = float(self.turns_left)

        legal[src_slot, tgt_slot] = pair["legal"]

    def _compute_pair(self, src, tgt):
        """Compute (eta, ships_needed, legal) for a single src→tgt edge."""
        _, _, sx, sy, _, src_ships, _ = src
        tgt_pid, tgt_owner, _, _, _, _, _ = tgt

        ships_for_speed = max(1, int(src_ships))
        intercept = self._lead_intercept((sx, sy), tgt_pid, ships_for_speed, src_radius=src[4])
        if intercept is None:
            return None
        eta = intercept["eta"]

        future_garrison = self._project_garrison(tgt_pid, eta)
        if tgt_owner == self.player:
            # Reinforcement: ships to patch a projected shortfall.
            ships_needed = max(1, -future_garrison) if future_garrison < 0 else 1
        else:
            # Attack: future_garrison is signed; for non-mine it is ≤ 0
            # from self's frame, so negate to get the defender's count.
            defender_count = -future_garrison if future_garrison < 0 else future_garrison
            ships_needed = max(1, int(defender_count) + 1)

        legal = self._sun_crossing_clear((sx, sy), intercept)
        return {"eta": float(eta), "ships_needed": float(ships_needed), "legal": legal}

    # ------------------------------------------------------------------
    # Fleet-prediction cache (populates the threats dict)
    # ------------------------------------------------------------------

    def _populate_all_fleet_predictions(self):
        """Simulate every fleet with the radar, store predictions by id."""
        radar = self._get_radar()
        self._fleet_predictions.clear()
        for f in self.fleets:
            self._fleet_predictions[int(f[0])] = radar.simulate_fleet(f)

    def _rebuild_threats_from_predictions(self) -> dict:
        """Assemble the threats dict from the cached fleet predictions."""
        threats: dict[int, list[dict]] = {}
        for f in self.fleets:
            fid = int(f[0])
            hit = self._fleet_predictions.get(fid)
            if hit is None or not hit.hit_planet:
                continue
            threats.setdefault(int(hit.target_id), []).append({
                "eta": int(hit.eta),
                "owner": int(f[1]),
                "ships": int(f[6]),
                "fleet_id": fid,
            })
        for entries in threats.values():
            entries.sort(key=lambda t: t["eta"])
        return threats

    # ------------------------------------------------------------------
    # Cross-turn update
    # ------------------------------------------------------------------

    def update_from_obs(self, new_obs):
        """Advance this view to the next turn's observation.

        Uses the cached fleet predictions: carry-over fleets get their eta
        decremented; only newly-appeared fleets are re-simulated. Planet
        positions, ownership, garrisons are taken from the new obs.
        """
        old_step = self.step
        new_step = int(_get(new_obs, "step", self.step + 1))
        steps_advanced = max(1, new_step - old_step)
        old_planet_id_set = set(self.planets_by_id.keys())

        # --- Absorb scalar & list state from the new obs ---
        self.obs = new_obs
        self.step = new_step
        self.turns_left = max(0, EPISODE_STEPS - self.step)
        self.angular_velocity = _get(new_obs, "angular_velocity", self.angular_velocity)
        self.planets = [list(p) for p in _get(new_obs, "planets", []) or []]
        self.planets_by_id = {p[0]: p for p in self.planets}

        # If the planet set changed (comet spawn/despawn), every carried-over
        # fleet prediction is suspect — a new comet may lie in a fleet's path,
        # or an expired one may no longer block it. Re-simulate all fleets.
        planets_changed = set(self.planets_by_id.keys()) != old_planet_id_set
        if planets_changed:
            self._fleet_predictions.clear()
        self.comet_planet_ids = set(_get(new_obs, "comet_planet_ids", []) or [])
        self.comets = list(_get(new_obs, "comets", []) or [])
        self._comet_lookup = self._build_comet_lookup()
        self.next_fleet_id = int(_get(new_obs, "next_fleet_id", self.next_fleet_id))

        # --- Reassign slots (comets can spawn/despawn, shifting the list) ---
        self.planet_ids[:] = -1
        self.slot_of.clear()
        for slot, planet in enumerate(self.planets[:self.n_max]):
            self.planet_ids[slot] = planet[0]
            self.slot_of[planet[0]] = slot

        # --- Invalidate radar cache — planets have rotated ---
        self._radar = None

        # --- Diff fleets; carry-over, add new, drop departed ---
        new_fleets = [list(f) for f in _get(new_obs, "fleets", []) or []]
        new_ids = {int(f[0]) for f in new_fleets}
        for fid in list(self._fleet_predictions.keys()):
            if fid not in new_ids:
                del self._fleet_predictions[fid]

        radar = self._get_radar()
        for f in new_fleets:
            fid = int(f[0])
            old_hit = self._fleet_predictions.get(fid)
            if old_hit is None:
                # Newly-appeared fleet — simulate via radar.
                self._fleet_predictions[fid] = radar.simulate_fleet(f)
                continue
            # Carry-over: decrement eta. If it's expired or wasn't a planet hit,
            # re-simulate to be safe (fleet might still be alive if previous
            # prediction was a board/sun exit but obs still contains it).
            if old_hit.eta is None or (old_hit.eta - steps_advanced) < 1:
                self._fleet_predictions[fid] = radar.simulate_fleet(f)
            else:
                self._fleet_predictions[fid] = RadarHit(
                    kind=old_hit.kind,
                    eta=old_hit.eta - steps_advanced,
                    target_id=old_hit.target_id,
                )

        self.fleets = new_fleets
        self.threats = self._rebuild_threats_from_predictions()

        # --- Refresh per-planet scalar caches; ownership / ships / threats all moved ---
        for slot in range(self.n_max):
            self._tgt_info[slot] = self._compute_tgt_info(slot)
            self._src_info[slot] = self._compute_src_info(slot)

        # --- Rebuild edges & legal mask from scratch ---
        # (Orbiter positions changed, so most geometry is invalid.)
        self.edge_features, self.legal_mask = self._build_edges()

        # --- Invalidate cached action mask ---
        self._cached_action_mask = None

    # ------------------------------------------------------------------
    # Incremental updates (sub-move deltas within a single turn)
    # ------------------------------------------------------------------

    def _get_radar(self) -> Radar:
        """Lazy-build and cache a Radar instance valid for the current step.

        The radar only depends on planet positions + comet paths, which are
        stable across sub-moves (the turn hasn't advanced). So one Radar
        serves all sub-moves — its position cache is reused.
        """
        if self._radar is None:
            self._radar = Radar(self.obs)
        return self._radar

    def _validate_mask_cell(self, src_slot: int, tgt_slot: int, safety_margin: int) -> bool:
        """Is this edge radar-legal under deterministic sizing?"""
        if not self.legal_mask[src_slot, tgt_slot]:
            return False
        ships = self.deterministic_ship_count(src_slot, tgt_slot, safety_margin)
        action = self.to_action(src_slot, tgt_slot, ships)
        if action is None:
            return False
        src_pid = int(self.planet_ids[src_slot])
        tgt_pid = int(self.planet_ids[tgt_slot])
        src = self.planets_by_id.get(src_pid)
        if src is None:
            return False
        hit = self._get_radar().simulate_launch(src, action[1], action[2])
        return bool(hit.hit_planet and hit.target_id == tgt_pid)

    def _rebuild_row(self, src_slot: int):
        """Recompute edge_features[src_slot, :] and legal_mask[src_slot, :].

        Caller is responsible for ensuring `_src_info` and `_tgt_info` are
        up-to-date for all slots before calling — this method does not
        refresh the caches itself.
        """
        src_pid = int(self.planet_ids[src_slot])
        src = self.planets_by_id.get(src_pid) if src_pid >= 0 else None
        self.edge_features[src_slot, :, :] = 0.0
        self.legal_mask[src_slot, :] = False
        if src is None or src[1] != self.player or src[5] <= 0:
            return
        for tgt_slot, tgt in enumerate(self.planets[:self.n_max]):
            if tgt_slot == src_slot:
                continue
            self._fill_edge_cell(src_slot, src, tgt_slot, tgt,
                                 self.edge_features, self.legal_mask)

    def _rebuild_col(self, tgt_slot: int):
        """Recompute edge_features[:, tgt_slot] and legal_mask[:, tgt_slot]."""
        tgt_pid = int(self.planet_ids[tgt_slot])
        tgt = self.planets_by_id.get(tgt_pid) if tgt_pid >= 0 else None
        self.edge_features[:, tgt_slot, :] = 0.0
        self.legal_mask[:, tgt_slot] = False
        if tgt is None:
            return
        for src_slot, src in enumerate(self.planets[:self.n_max]):
            if src_slot == tgt_slot:
                continue
            if src[1] != self.player or src[5] <= 0:
                continue
            self._fill_edge_cell(src_slot, src, tgt_slot, tgt,
                                 self.edge_features, self.legal_mask)

    def _update_action_mask_for_slots(self, slots):
        """Re-validate the cached action mask for the rows and cols of `slots`."""
        if self._cached_action_mask is None:
            return
        sm = self._cached_safety_margin
        mask = self._cached_action_mask
        touched = set()
        for slot in slots:
            for j in range(self.n_max):
                touched.add((slot, j))
                touched.add((j, slot))
        for src_slot, tgt_slot in touched:
            if self.legal_mask[src_slot, tgt_slot] and self._validate_mask_cell(
                src_slot, tgt_slot, sm
            ):
                mask[src_slot, tgt_slot] = True
            else:
                mask[src_slot, tgt_slot] = False

    def apply_planned_move(self, src_slot: int, tgt_slot: int, ships: int) -> Optional[list]:
        """Mutate this view in place to reflect a same-turn planned launch.

        Returns the resolved `[src_pid, angle, ships]` action if applied, or
        None if the launch was invalid (shouldn't happen if the caller used
        `action_mask()` to pre-filter).

        Side effects:
          * `self.planets` / `planets_by_id`: src's garrison drops by `ships`
          * `self.fleets`: planned fleet appended
          * `self.threats[tgt_pid]`: planned fleet's predicted landing inserted
          * `edge_features[src_slot, :]` and `[:, tgt_slot]` recomputed
          * `legal_mask[src_slot, :]` and `[:, tgt_slot]` recomputed
          * cached action mask (if built) patched for the same strips
        """
        action = self.to_action(src_slot, tgt_slot, ships)
        if action is None:
            return None
        src_pid, angle, real_ships = int(action[0]), float(action[1]), int(action[2])
        src = self.planets_by_id[src_pid]
        tgt_pid = int(self.planet_ids[tgt_slot])

        radar = self._get_radar()
        hit = radar.simulate_launch(src, angle, real_ships)
        if not (hit.hit_planet and hit.target_id == tgt_pid):
            return None

        # Planned fleet bookkeeping.
        launch_xy = radar.launch_position(src, angle)
        planned_id = self.next_fleet_id
        self.next_fleet_id += 1
        planned_fleet = [planned_id, int(self.player), float(launch_xy[0]),
                         float(launch_xy[1]), float(angle), src_pid, real_ships]
        self.fleets.append(planned_fleet)

        # Apply the ship-count drop on the source planet (in our mutable copy).
        src[5] = int(src[5]) - real_ships

        # Insert into threats at the intended target.
        self.threats.setdefault(tgt_pid, []).append({
            "eta": int(hit.eta),
            "owner": int(self.player),
            "ships": int(real_ships),
            "fleet_id": int(planned_id),
        })
        self.threats[tgt_pid].sort(key=lambda t: t["eta"])

        # Refresh per-planet caches for both affected slots. A launch changes:
        #   src: ships dropped → _src_info[src], _tgt_info[src] (will_fall risk ↑)
        #   tgt: threats gained a friendly → _src_info[tgt] (net_threat shift),
        #                                    _tgt_info[tgt] (will_fall may flip)
        for slot in (src_slot, tgt_slot):
            self._src_info[slot] = self._compute_src_info(slot)
            self._tgt_info[slot] = self._compute_tgt_info(slot)

        # Rebuild both planets' rows and columns in edge_features / legal_mask.
        for slot in (src_slot, tgt_slot):
            self._rebuild_row(slot)
            self._rebuild_col(slot)

        # Patch the radar-backed action mask for the same four strips.
        self._update_action_mask_for_slots((src_slot, tgt_slot))

        return action

    # ------------------------------------------------------------------
    # Debug self-check — compare incremental state to a cold rebuild
    # ------------------------------------------------------------------

    def _reconstructed_obs(self) -> dict:
        """Emit a dict that a fresh GameView can be built from."""
        return {
            "step": self.step,
            "player": self.player,
            "planets": [list(p) for p in self.planets],
            "fleets": [list(f) for f in self.fleets],
            "angular_velocity": self.angular_velocity,
            "initial_planets": [list(p) for p in self.initial_planets],
            "comets": list(self.comets),
            "comet_planet_ids": list(self.comet_planet_ids),
            "next_fleet_id": self.next_fleet_id,
        }

    def assert_equals_fresh_rebuild(self, atol: float = 1e-4):
        """Assert this view matches a cold rebuild from the same internal state.

        Call after `apply_planned_move` in debug mode to catch drift.
        """
        fresh = GameView(self._reconstructed_obs(), n_max=self.n_max)
        if not np.allclose(self.edge_features, fresh.edge_features, atol=atol):
            diff = np.abs(self.edge_features - fresh.edge_features)
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            raise AssertionError(
                f"edge_features drift at {idx}: "
                f"incremental={self.edge_features[idx]} fresh={fresh.edge_features[idx]}"
            )
        if not np.array_equal(self.legal_mask, fresh.legal_mask):
            idx = np.argwhere(self.legal_mask != fresh.legal_mask)[0]
            raise AssertionError(
                f"legal_mask drift at {tuple(idx)}: "
                f"incremental={self.legal_mask[tuple(idx)]} fresh={fresh.legal_mask[tuple(idx)]}"
            )
        inc_mask = self.action_mask(self._cached_safety_margin)
        fresh_mask = fresh.action_mask(self._cached_safety_margin)
        if not np.array_equal(inc_mask, fresh_mask):
            idx = np.argwhere(inc_mask != fresh_mask)[0]
            raise AssertionError(
                f"action_mask drift at {tuple(idx)}: "
                f"incremental={inc_mask[tuple(idx)]} fresh={fresh_mask[tuple(idx)]}"
            )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _lead_intercept(self, src_xy, tgt_pid, ships, src_radius=None, max_iter=30, tol=0.05):
        """Fixed-point solve for (angle, eta) to intercept a moving target."""
        speed = T.fleet_speed(ships)
        sx, sy = src_xy
        pos = self._position_at(tgt_pid, 0)
        if pos is None:
            return None
        angle = math.atan2(pos[1] - sy, pos[0] - sx)
        launch = self._launch_point((sx, sy), src_radius, angle)
        eta = max(1.0, math.hypot(pos[0] - launch[0], pos[1] - launch[1]) / speed)
        converged = False
        for _ in range(max_iter):
            pos = self._position_at(tgt_pid, int(math.ceil(eta)))
            if pos is None:
                return None
            launch = self._launch_point((sx, sy), src_radius, angle)
            new_angle = math.atan2(pos[1] - launch[1], pos[0] - launch[0])
            new_launch = self._launch_point((sx, sy), src_radius, new_angle)
            new_eta = math.hypot(pos[0] - new_launch[0], pos[1] - new_launch[1]) / speed
            if abs(new_eta - eta) < tol:
                eta = new_eta
                angle = new_angle
                converged = True
                break
            eta = new_eta
            angle = new_angle
        if not converged:
            return None
        eta_i = max(1, int(math.ceil(eta)))
        pos = self._position_at(tgt_pid, eta_i)
        if pos is None:
            return None
        launch = self._launch_point((sx, sy), src_radius, angle)
        angle = math.atan2(pos[1] - launch[1], pos[0] - launch[0])
        return {"angle": angle, "eta": eta_i,
                "pred_x": pos[0], "pred_y": pos[1], "speed": speed}

    def _launch_point(self, src_xy, src_radius, angle):
        if src_radius is None:
            return src_xy
        return (
            src_xy[0] + math.cos(angle) * (float(src_radius) + 0.1),
            src_xy[1] + math.sin(angle) * (float(src_radius) + 0.1),
        )

    def _position_at(self, planet_id, t_ahead):
        planet = self.planets_by_id.get(planet_id)
        if planet is None:
            return None
        if planet_id in self.comet_planet_ids:
            return self._comet_position_at(planet_id, t_ahead)
        initial = self.initial_by_id.get(planet_id)
        if initial is None:
            return (planet[2], planet[3])
        return T.future_position(initial, self.step, t_ahead, self.angular_velocity)

    def _build_comet_lookup(self):
        out = {}
        for group in self.comets:
            ids = group.get("planet_ids", []) if isinstance(group, dict) else []
            paths = group.get("paths", []) if isinstance(group, dict) else []
            path_index = group.get("path_index", -1) if isinstance(group, dict) else -1
            for i, pid in enumerate(ids):
                if i < len(paths):
                    out[pid] = (paths[i], path_index)
        return out

    def _comet_position_at(self, planet_id, t_ahead):
        planet = self.planets_by_id.get(planet_id)
        entry = self._comet_lookup.get(planet_id)
        if entry is None:
            return (planet[2], planet[3]) if t_ahead == 0 else None
        path, path_index = entry
        idx = path_index + int(t_ahead)
        if idx < 0 or idx >= len(path):
            return (planet[2], planet[3]) if t_ahead == 0 else None
        return (path[idx][0], path[idx][1])

    def _sun_crossing_clear(self, src_xy, intercept):
        """Does the straight-line path pass outside the sun?"""
        end = (intercept["pred_x"], intercept["pred_y"])
        return _point_to_segment_distance(SUN, src_xy, end) >= SUN_R

    # ------------------------------------------------------------------
    # Threat projections
    # ------------------------------------------------------------------

    def _project_garrison(self, planet_id, horizon):
        """Signed projection at step+horizon given known inbound fleets.

        Positive → I own with that many ships. Negative → someone else
        owns with |value| ships.  Includes production and resolves
        arrivals in eta order with the simple "subtract, flip if
        overflow" combat proxy.
        """
        planet = self.planets_by_id[planet_id]
        owner = planet[1]
        ships = int(planet[5])
        prod = int(planet[6])
        arrivals = sorted(self.threats.get(planet_id, []), key=lambda t: t["eta"])
        last_t = 0
        for ar in arrivals:
            eta = ar["eta"]
            if eta > horizon:
                break
            if owner != -1:
                ships += prod * max(0, eta - last_t)
            if ar["owner"] == owner:
                ships += ar["ships"]
            else:
                ships -= ar["ships"]
                if ships < 0:
                    owner = ar["owner"]
                    ships = abs(ships)
            last_t = eta
        if owner != -1:
            ships += prod * max(0, horizon - last_t)
        return ships if owner == self.player else -ships

    def _will_fall_if_ignored(self, planet_id):
        """True iff this (my) planet is projected to flip to non-me."""
        planet = self.planets_by_id[planet_id]
        if planet[1] != self.player:
            return False
        ships = int(planet[5])
        prod = int(planet[6])
        owner = planet[1]
        last_t = 0
        for ar in sorted(self.threats.get(planet_id, []), key=lambda t: t["eta"]):
            eta = ar["eta"]
            ships += prod * max(0, eta - last_t)
            if ar["owner"] == owner:
                ships += ar["ships"]
            else:
                ships -= ar["ships"]
                if ships < 0:
                    return True
            last_t = eta
        return False

    def first_flip_eta(self, planet_id):
        """Step at which this owned planet flips to non-me, or None."""
        planet = self.planets_by_id[planet_id]
        if planet[1] != self.player:
            return None
        ships = int(planet[5])
        prod = int(planet[6])
        owner = planet[1]
        last_t = 0
        for ar in sorted(self.threats.get(planet_id, []), key=lambda t: t["eta"]):
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

    def _net_threat_at(self, planet_id):
        total = 0
        for ar in self.threats.get(planet_id, []):
            if ar["owner"] == self.player:
                total -= ar["ships"]
            else:
                total += ar["ships"]
        return float(total)

    def _expiry_for(self, planet_id):
        if planet_id not in self.comet_planet_ids:
            return COMET_EXPIRY_SENTINEL
        entry = self._comet_lookup.get(planet_id)
        if entry is None:
            return COMET_EXPIRY_SENTINEL
        path, path_index = entry
        return float(max(0, len(path) - path_index - 1))
