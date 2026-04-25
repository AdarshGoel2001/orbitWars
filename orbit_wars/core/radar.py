"""Env-faithful trajectory radar for Orbit Wars.

This module predicts where existing fleets and candidate launches will end up.
It mirrors the environment's per-turn fleet movement checks: board bounds, sun
segment crossing, planet segment collision, then moving-planet sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

BOARD_SIZE = 100.0
CENTER = 50.0
SUN = (CENTER, CENTER)
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
SPAWN_OFFSET = 0.1
DEFAULT_HORIZON = 500
MAX_SPEED = 6.0


@dataclass(frozen=True)
class RadarHit:
    kind: str
    eta: Optional[int] = None
    target_id: Optional[int] = None

    @property
    def hit_planet(self) -> bool:
        return self.kind in ("hit_planet", "swept_planet")


def _get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def point_to_segment_distance(point, a, b):
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


def fleet_speed(ships, max_speed=MAX_SPEED):
    if ships <= 1:
        return 1.0
    return 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def _is_orbiting(initial_planet):
    _pid, _o, ix, iy, r = initial_planet[:5]
    return math.hypot(ix - CENTER, iy - CENTER) + r < ROTATION_RADIUS_LIMIT


def _future_position(initial_planet, current_step, t_ahead, angular_velocity):
    if not _is_orbiting(initial_planet):
        return (initial_planet[2], initial_planet[3])
    _pid, _o, ix, iy = initial_planet[:4]
    dx, dy = ix - CENTER, iy - CENTER
    orbital_radius = math.hypot(dx, dy)
    phase0 = math.atan2(dy, dx)
    phi = phase0 + (current_step + t_ahead - 1) * angular_velocity
    return (
        CENTER + orbital_radius * math.cos(phi),
        CENTER + orbital_radius * math.sin(phi),
    )


class Radar:
    """Trajectory simulator over one observation."""

    def __init__(self, obs, horizon: int = DEFAULT_HORIZON):
        self.obs = obs
        self.horizon = horizon
        self.step = int(_get(obs, "step", 0))
        self.angular_velocity = float(_get(obs, "angular_velocity", 0.0))
        self.planets = [list(p) for p in _get(obs, "planets", []) or []]
        self.initial_planets = [list(p) for p in _get(obs, "initial_planets", []) or []]
        self.comets = [dict(c) for c in _get(obs, "comets", []) or []]
        self.comet_planet_ids = set(_get(obs, "comet_planet_ids", []) or [])

        self.planets_by_id = {p[0]: p for p in self.planets}
        self.initial_by_id = {p[0]: p for p in self.initial_planets}
        self._comet_lookup = self._build_comet_lookup()
        self._pos_cache: dict[tuple[int, int], Optional[tuple[float, float]]] = {}

    def simulate_fleet(self, fleet, horizon: Optional[int] = None) -> RadarHit:
        """Predict the first event for an existing fleet record."""
        return self._simulate(
            x=float(fleet[2]),
            y=float(fleet[3]),
            angle=float(fleet[4]),
            from_planet_id=int(fleet[5]),
            ships=int(fleet[6]),
            horizon=horizon,
        )

    def simulate_launch(
        self,
        src_planet,
        angle: float,
        ships: int,
        horizon: Optional[int] = None,
    ) -> RadarHit:
        """Predict the first event for a newly launched fleet."""
        sx = float(src_planet[2])
        sy = float(src_planet[3])
        sr = float(src_planet[4])
        x = sx + math.cos(angle) * (sr + SPAWN_OFFSET)
        y = sy + math.sin(angle) * (sr + SPAWN_OFFSET)
        return self._simulate(
            x=x,
            y=y,
            angle=float(angle),
            from_planet_id=int(src_planet[0]),
            ships=int(ships),
            horizon=horizon,
        )

    def launch_position(self, src_planet, angle: float) -> tuple[float, float]:
        sx = float(src_planet[2])
        sy = float(src_planet[3])
        sr = float(src_planet[4])
        return (
            sx + math.cos(angle) * (sr + SPAWN_OFFSET),
            sy + math.sin(angle) * (sr + SPAWN_OFFSET),
        )

    def _simulate(
        self,
        x: float,
        y: float,
        angle: float,
        from_planet_id: int,
        ships: int,
        horizon: Optional[int],
    ) -> RadarHit:
        limit = self.horizon if horizon is None else horizon
        limit = max(0, min(int(limit), max(0, 500 - self.step)))
        speed = fleet_speed(ships)
        cx, cy = float(x), float(y)

        for eta in range(1, limit + 1):
            old_pos = (cx, cy)
            cx += math.cos(angle) * speed
            cy += math.sin(angle) * speed
            new_pos = (cx, cy)

            if not (0 <= cx <= BOARD_SIZE and 0 <= cy <= BOARD_SIZE):
                return RadarHit("board", eta=eta)

            if point_to_segment_distance(SUN, old_pos, new_pos) < SUN_RADIUS:
                return RadarHit("sun", eta=eta)

            # Env checks collisions against planet positions before rotation
            # for this turn. Since this Radar starts from an observation at
            # `step`, eta=1 sees planet positions at t_ahead=0.
            for planet in self.planets:
                ppos = self.position_at(planet[0], eta - 1)
                if ppos is None:
                    continue
                if point_to_segment_distance(ppos, old_pos, new_pos) < planet[4]:
                    return RadarHit("hit_planet", eta=eta, target_id=int(planet[0]))

            # Then the env rotates/moves planets and sweeps fleets.
            for planet in self.planets:
                if int(planet[0]) == from_planet_id and eta <= 2:
                    continue
                old_planet_pos = self.position_at(planet[0], eta - 1)
                new_planet_pos = self.position_at(planet[0], eta)
                if old_planet_pos is None or new_planet_pos is None:
                    continue
                if old_planet_pos == new_planet_pos:
                    continue
                if point_to_segment_distance(new_pos, old_planet_pos, new_planet_pos) < planet[4]:
                    return RadarHit("swept_planet", eta=eta, target_id=int(planet[0]))

        return RadarHit("timeout")

    def position_at(self, planet_id: int, t_ahead: int) -> Optional[tuple[float, float]]:
        key = (int(planet_id), int(t_ahead))
        if key in self._pos_cache:
            return self._pos_cache[key]

        planet = self.planets_by_id.get(planet_id)
        if planet is None:
            self._pos_cache[key] = None
            return None

        if planet_id in self.comet_planet_ids:
            pos = self._comet_position_at(planet_id, t_ahead)
        else:
            initial = self.initial_by_id.get(planet_id)
            if initial is None:
                pos = (float(planet[2]), float(planet[3]))
            else:
                pos = _future_position(initial, self.step, t_ahead, self.angular_velocity)

        self._pos_cache[key] = pos
        return pos

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

    def _comet_position_at(self, planet_id: int, t_ahead: int) -> Optional[tuple[float, float]]:
        planet = self.planets_by_id.get(planet_id)
        entry = self._comet_lookup.get(planet_id)
        if entry is None:
            return (float(planet[2]), float(planet[3])) if t_ahead == 0 else None
        path, path_index = entry
        idx = path_index + int(t_ahead)
        if idx < 0 or idx >= len(path):
            return (float(planet[2]), float(planet[3])) if t_ahead == 0 else None
        return (float(path[idx][0]), float(path[idx][1]))


def simulate_fleet(obs, fleet, horizon: int = DEFAULT_HORIZON) -> RadarHit:
    return Radar(obs, horizon=horizon).simulate_fleet(fleet)


def simulate_launch(obs, src_planet, angle: float, ships: int, horizon: int = DEFAULT_HORIZON) -> RadarHit:
    return Radar(obs, horizon=horizon).simulate_launch(src_planet, angle, ships)
