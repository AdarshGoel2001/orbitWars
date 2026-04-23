"""Dynamic-edge harness for Orbit Wars — CPU inference focus.

Emits variable-length edge tokens (only action-mask-valid edges) instead of
a padded N_max × N_max grid.  One observation in → one ``TokenBundle`` out,
where each token is a single src→tgt launch that has already passed radar
validation under deterministic ship sizing.

Reuses ``radar.py`` and ``targeting.py`` unchanged.

Comets are excluded from consideration in this version — planets listed in
``obs.comet_planet_ids`` are filtered out of the internal planet table.
Re-introduce when we're ready to pay for the extra edges.

Output contract (see ``TokenBundle``):
    edges      (N, 10) float32    per-edge features
    src_ids    (N,) int32         source slot (0..num_planets-1)
    tgt_ids    (N,) int32         target slot
    ships      (N,) int32         deterministic ship count for this edge
    angles     (N,) float32       lead-intercept angle (radians)
    planet_ids (P,) int32         slot → env planet_id

Feature layout (indices are FEATURE_* constants):
    0 eta                  lead-intercept turns to arrival
    1 ships_needed         attackers required to flip tgt at that eta
    2 kind_reinforce       bool: src mine, tgt mine
    3 kind_attack_enemy    bool: src mine, tgt enemy
    4 kind_attack_neutral  bool: src mine, tgt neutral
    5 src_ships            current garrison at src
    6 src_net_threat       enemy_inbound − friendly_inbound at src
    7 tgt_production       tgt production rate
    8 tgt_will_fall        bool: tgt (mine) is projected to flip
    9 turns_left           episode turns remaining (broadcast scalar)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

import targeting as T
from radar import Radar, RadarHit


BOARD = 100.0
SUN = (50.0, 50.0)
SUN_R = 10.0
EPISODE_STEPS = 500

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
FEATURE_TURNS_LEFT = 9
FEATURE_DIM = 10

FEATURE_NAMES = [
    "eta", "ships_needed",
    "kind_reinforce", "kind_attack_enemy", "kind_attack_neutral",
    "src_ships", "src_net_threat",
    "tgt_production", "tgt_will_fall",
    "turns_left",
]

FEATURE_SCALES = np.array([
    50.0,   # eta
    100.0,  # ships_needed
    1.0, 1.0, 1.0,
    100.0,  # src_ships
    50.0,   # src_net_threat
    5.0,    # tgt_production
    1.0,
    500.0,  # turns_left
], dtype=np.float32)


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


@dataclass
class TokenBundle:
    """Packed edge tokens for a single observation."""

    edges: np.ndarray        # (N, FEATURE_DIM) float32
    src_ids: np.ndarray      # (N,) int32 — slot index into planet_ids
    tgt_ids: np.ndarray      # (N,) int32 — slot index into planet_ids
    ships: np.ndarray        # (N,) int32
    angles: np.ndarray       # (N,) float32
    planet_ids: np.ndarray   # (P,) int32 — slot → env planet_id

    @property
    def n(self) -> int:
        return int(self.edges.shape[0])

    @property
    def num_planets(self) -> int:
        return int(self.planet_ids.shape[0])


class GameView_CPU:
    """Observation wrapped as variable-length edge tokens.

    Comets are dropped — this view only considers permanent planets.
    """

    def __init__(self, obs, safety_margin: int = 1):
        self.safety_margin = int(safety_margin)
        self._radar: Optional[Radar] = None
        self._fleet_predictions: dict[int, RadarHit] = {}
        self._tokens: Optional[TokenBundle] = None
        self._tokens_dirty = True
        self._absorb_obs(obs)
        self._populate_all_fleet_predictions()
        self.threats = self._rebuild_threats_from_predictions()

    # ------------------------------------------------------------------
    # State ingestion
    # ------------------------------------------------------------------

    def _absorb_obs(self, obs):
        """Copy planets / fleets / scalars from obs, filter out comets."""
        self.obs = obs
        self.player = int(_get(obs, "player", 0))
        self.step = int(_get(obs, "step", 0))
        self.turns_left = max(0, EPISODE_STEPS - self.step)
        self.angular_velocity = float(_get(obs, "angular_velocity", 0.0))

        comet_ids = set(_get(obs, "comet_planet_ids", []) or [])
        raw_planets = _get(obs, "planets", []) or []
        self.planets = [list(p) for p in raw_planets if p[0] not in comet_ids]
        self.planets_by_id = {p[0]: p for p in self.planets}

        raw_initial = _get(obs, "initial_planets", []) or []
        self.initial_planets = [list(p) for p in raw_initial if p[0] not in comet_ids]
        self.initial_by_id = {p[0]: p for p in self.initial_planets}

        self.fleets = [list(f) for f in _get(obs, "fleets", []) or []]
        self.next_fleet_id = int(_get(obs, "next_fleet_id", 0) or 0)

        self._num_planets = len(self.planets)
        self._planet_ids_array = np.array(
            [p[0] for p in self.planets], dtype=np.int32
        )
        self._slot_of: dict[int, int] = {p[0]: i for i, p in enumerate(self.planets)}

        # New obs → planets may have rotated → radar cache is stale.
        self._radar = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokens(self) -> TokenBundle:
        if self._tokens_dirty or self._tokens is None:
            self._tokens = self._build_tokens()
            self._tokens_dirty = False
        return self._tokens

    def decode_action(self, token_idx: int) -> Optional[list]:
        """Convert a token index to ``[from_planet_id, angle, num_ships]``."""
        bundle = self.tokens()
        if not (0 <= token_idx < bundle.n):
            return None
        src_slot = int(bundle.src_ids[token_idx])
        src_pid = int(bundle.planet_ids[src_slot])
        return [src_pid, float(bundle.angles[token_idx]), int(bundle.ships[token_idx])]

    def apply_planned_move(self, token_idx: int) -> Optional[list]:
        """Mutate the view to reflect a same-turn launch of the given token.

        Returns the resolved ``[src_pid, angle, ships]`` action, or None if
        the token index is invalid / radar rejects the launch.  On success:

          * ``self.planets``: src garrison drops by ``ships``
          * ``self.fleets``: planned fleet appended (so future projections see it)
          * ``self.threats[tgt_pid]``: planned fleet's predicted landing inserted
          * ``self._fleet_predictions[planned_id]``: stored for cross-turn reuse
          * tokens are marked dirty; next ``tokens()`` call rebuilds
        """
        bundle = self.tokens()
        if not (0 <= token_idx < bundle.n):
            return None
        src_slot = int(bundle.src_ids[token_idx])
        tgt_slot = int(bundle.tgt_ids[token_idx])
        src_pid = int(bundle.planet_ids[src_slot])
        tgt_pid = int(bundle.planet_ids[tgt_slot])
        angle = float(bundle.angles[token_idx])
        ships = int(bundle.ships[token_idx])

        src = self.planets_by_id.get(src_pid)
        if src is None or src[5] < ships or ships <= 0:
            return None

        radar = self._get_radar()
        hit = radar.simulate_launch(src, angle, ships)
        if not (hit.hit_planet and hit.target_id == tgt_pid):
            return None
        launch_xy = radar.launch_position(src, angle)

        # Apply state deltas.
        src[5] = int(src[5]) - ships
        planned_id = self.next_fleet_id
        self.next_fleet_id += 1
        planned_fleet = [
            planned_id, int(self.player),
            float(launch_xy[0]), float(launch_xy[1]),
            float(angle), src_pid, ships,
        ]
        self.fleets.append(planned_fleet)
        self._fleet_predictions[planned_id] = hit
        self.threats.setdefault(tgt_pid, []).append({
            "eta": int(hit.eta),
            "owner": int(self.player),
            "ships": int(ships),
            "fleet_id": int(planned_id),
        })
        self.threats[tgt_pid].sort(key=lambda t: t["eta"])

        self._tokens_dirty = True
        return [src_pid, angle, ships]

    def update_from_obs(self, new_obs):
        """Advance to next turn; reuse carry-over fleet predictions."""
        old_step = self.step
        new_step = int(_get(new_obs, "step", old_step + 1))
        steps_advanced = max(1, new_step - old_step)
        old_predictions = dict(self._fleet_predictions)

        self._absorb_obs(new_obs)

        # Diff fleets: carry-overs reuse prediction with eta decremented;
        # new ones get a fresh simulate_fleet.
        radar = self._get_radar()
        self._fleet_predictions.clear()
        for f in self.fleets:
            fid = int(f[0])
            old_hit = old_predictions.get(fid)
            if (old_hit is not None
                    and old_hit.hit_planet
                    and old_hit.eta is not None
                    and (old_hit.eta - steps_advanced) >= 1):
                self._fleet_predictions[fid] = RadarHit(
                    kind=old_hit.kind,
                    eta=old_hit.eta - steps_advanced,
                    target_id=old_hit.target_id,
                )
            else:
                self._fleet_predictions[fid] = radar.simulate_fleet(f)

        self.threats = self._rebuild_threats_from_predictions()
        self._tokens_dirty = True

    # ------------------------------------------------------------------
    # Token construction
    # ------------------------------------------------------------------

    def _build_tokens(self) -> TokenBundle:
        # Per-planet scalar caches.
        tgt_info = [self._compute_tgt_info(i) for i in range(self._num_planets)]
        src_info = [self._compute_src_info(i) for i in range(self._num_planets)]

        edges_list: list[np.ndarray] = []
        src_ids_list: list[int] = []
        tgt_ids_list: list[int] = []
        ships_list: list[int] = []
        angles_list: list[float] = []

        radar = self._get_radar()
        for src_slot, src in enumerate(self.planets):
            if src[1] != self.player or src[5] <= 0:
                continue
            for tgt_slot, tgt in enumerate(self.planets):
                if src_slot == tgt_slot:
                    continue
                tok = self._try_build_token(
                    src_slot, src, tgt_slot, tgt,
                    src_info[src_slot], tgt_info[tgt_slot], radar,
                )
                if tok is None:
                    continue
                features, ships, angle = tok
                edges_list.append(features)
                src_ids_list.append(src_slot)
                tgt_ids_list.append(tgt_slot)
                ships_list.append(ships)
                angles_list.append(angle)

        if edges_list:
            edges = np.stack(edges_list, axis=0).astype(np.float32, copy=False)
        else:
            edges = np.zeros((0, FEATURE_DIM), dtype=np.float32)
        return TokenBundle(
            edges=edges,
            src_ids=np.asarray(src_ids_list, dtype=np.int32),
            tgt_ids=np.asarray(tgt_ids_list, dtype=np.int32),
            ships=np.asarray(ships_list, dtype=np.int32),
            angles=np.asarray(angles_list, dtype=np.float32),
            planet_ids=self._planet_ids_array,
        )

    def _try_build_token(self, src_slot, src, tgt_slot, tgt,
                         src_info, tgt_info, radar):
        """Return (features, ships, angle) if the edge is mask-valid, else None."""
        _, _, sx, sy, src_r, src_ships_avail, _ = src
        tgt_pid = tgt[0]
        tgt_owner = tgt[1]

        # Fast lead-intercept at max-speed estimate (used for eta/ships_needed).
        ships_for_speed = max(1, int(src_ships_avail))
        intercept = self._lead_intercept(
            (sx, sy), tgt_pid, ships_for_speed, src_radius=src_r,
        )
        if intercept is None:
            return None
        eta = intercept["eta"]

        if not self._sun_crossing_clear((sx, sy), intercept):
            return None

        # Future-garrison projection at arrival → ships_needed feature.
        future_garrison = self._project_garrison(tgt_pid, eta)
        if tgt_owner == self.player:
            ships_needed_f = (
                max(1.0, float(-future_garrison)) if future_garrison < 0 else 1.0
            )
        else:
            defender = -future_garrison if future_garrison < 0 else future_garrison
            ships_needed_f = max(1.0, float(defender) + 1.0)

        # Deterministic ship count (defense deficit vs attack need).
        if tgt_owner == self.player:
            flip_eta = self.first_flip_eta(tgt_pid)
            if flip_eta is None:
                need = 1
            else:
                deficit = -self._project_garrison(tgt_pid, flip_eta)
                need = max(1, int(math.ceil(deficit)) + self.safety_margin)
        else:
            need = max(1, int(math.ceil(ships_needed_f)) + self.safety_margin)
        ships = min(int(src_ships_avail), need)
        if ships <= 0:
            return None

        # Re-solve angle with the actual ship count (speed depends on ships).
        # Skip if ship count matches the speed estimate we already used.
        if ships == ships_for_speed:
            final_angle = float(intercept["angle"])
        else:
            final_intercept = self._lead_intercept(
                (sx, sy), tgt_pid, ships, src_radius=src_r,
            )
            if final_intercept is None:
                return None
            final_angle = float(final_intercept["angle"])

        # Radar-validate: the launched fleet's first hit must be tgt_pid.
        hit = radar.simulate_launch(src, final_angle, ships)
        if not (hit.hit_planet and hit.target_id == tgt_pid):
            return None

        # Feature vector.
        f = np.zeros(FEATURE_DIM, dtype=np.float32)
        f[FEATURE_ETA] = eta
        f[FEATURE_SHIPS_NEEDED] = ships_needed_f
        if tgt_owner == self.player:
            f[FEATURE_KIND_REINFORCE] = 1.0
        elif tgt_owner == -1:
            f[FEATURE_KIND_ATTACK_NEUTRAL] = 1.0
        else:
            f[FEATURE_KIND_ATTACK_ENEMY] = 1.0
        f[FEATURE_SRC_SHIPS] = src_info["ships"]
        f[FEATURE_SRC_NET_THREAT] = src_info["net_threat"]
        f[FEATURE_TGT_PRODUCTION] = tgt_info["production"]
        f[FEATURE_TGT_WILL_FALL] = 1.0 if tgt_info["will_fall"] else 0.0
        f[FEATURE_TURNS_LEFT] = float(self.turns_left)
        return f, int(ships), final_angle

    def _compute_tgt_info(self, slot: int) -> dict:
        planet = self.planets[slot]
        return {
            "production": float(planet[6]),
            "will_fall": self._will_fall_if_ignored(planet[0]),
        }

    def _compute_src_info(self, slot: int) -> dict:
        planet = self.planets[slot]
        return {
            "ships": float(planet[5]),
            "net_threat": self._net_threat_at(planet[0]),
        }

    # ------------------------------------------------------------------
    # Radar / threat bookkeeping
    # ------------------------------------------------------------------

    def _get_radar(self) -> Radar:
        if self._radar is None:
            self._radar = Radar(self.obs)
        return self._radar

    def _populate_all_fleet_predictions(self):
        radar = self._get_radar()
        self._fleet_predictions.clear()
        for f in self.fleets:
            self._fleet_predictions[int(f[0])] = radar.simulate_fleet(f)

    def _rebuild_threats_from_predictions(self) -> dict:
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
    # Geometry
    # ------------------------------------------------------------------

    def _lead_intercept(self, src_xy, tgt_pid, ships, src_radius=None,
                        max_iter=30, tol=0.05):
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
        initial = self.initial_by_id.get(planet_id)
        if initial is None:
            return (planet[2], planet[3])
        return T.future_position(initial, self.step, t_ahead, self.angular_velocity)

    def _sun_crossing_clear(self, src_xy, intercept):
        end = (intercept["pred_x"], intercept["pred_y"])
        return _point_to_segment_distance(SUN, src_xy, end) >= SUN_R

    # ------------------------------------------------------------------
    # Threat projections
    # ------------------------------------------------------------------

    def _project_garrison(self, planet_id, horizon):
        """Signed garrison at step + horizon under known inbound fleets."""
        planet = self.planets_by_id.get(planet_id)
        if planet is None:
            return 0
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
        planet = self.planets_by_id.get(planet_id)
        if planet is None or planet[1] != self.player:
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
        planet = self.planets_by_id.get(planet_id)
        if planet is None or planet[1] != self.player:
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
