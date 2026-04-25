"""Pure-math helpers for aiming, threat detection, and ring analysis.

No obstruction checks (sun/planet blockers) yet — kept intentionally simple.
Shared by the human UI (via server endpoints) and by future agents.
"""

import math

from radar import Radar

SUN = (50.0, 50.0)
SUN_R = 10.0
BOARD = 100.0
MAX_SPEED = 6.0


def fleet_speed(ships, max_speed=MAX_SPEED):
    """Speed per turn for a fleet of N ships (per game's log formula)."""
    if ships <= 1:
        return 1.0
    return 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def is_orbiting(initial_planet):
    _pid, _o, ix, iy, r = initial_planet[:5]
    return (math.hypot(ix - SUN[0], iy - SUN[1]) + r) < 50.0


def orbit_params(initial_planet):
    """(orbital_radius, phase_at_step_1) for an orbiting planet."""
    _pid, _o, ix, iy = initial_planet[:4]
    dx, dy = ix - SUN[0], iy - SUN[1]
    return math.hypot(dx, dy), math.atan2(dy, dx)


def future_position(initial_planet, current_step, t_ahead, angular_velocity):
    """Where a planet will be at step (current_step + t_ahead).

    Rotation formula (verified empirically): phase = phase0 + (step - 1) * ω.
    """
    if not is_orbiting(initial_planet):
        return (initial_planet[2], initial_planet[3])
    orb_r, phase0 = orbit_params(initial_planet)
    phi = phase0 + (current_step + t_ahead - 1) * angular_velocity
    return (SUN[0] + orb_r * math.cos(phi), SUN[1] + orb_r * math.sin(phi))


def lead_intercept(src_xy, target_initial, ships, angular_velocity, current_step,
                   max_iter=30, tol=0.05):
    """Fixed-point solve for (angle, eta, predicted_pos).

    Source is treated as a stationary point (its current xy). We iterate:
      eta_{n+1} = distance(src, target_pos_at(current_step + eta_n)) / speed.
    Returns dict or None if it fails to converge.
    """
    speed = fleet_speed(ships)
    sx, sy = src_xy
    tx, ty = future_position(target_initial, current_step, 0, angular_velocity)
    eta = max(1.0, math.hypot(tx - sx, ty - sy) / speed)
    converged = False
    for _ in range(max_iter):
        tx, ty = future_position(target_initial, current_step, eta, angular_velocity)
        new_eta = math.hypot(tx - sx, ty - sy) / speed
        if abs(new_eta - eta) < tol:
            eta = new_eta
            converged = True
            break
        eta = new_eta
    if not converged:
        return None
    eta_i = max(1, int(math.ceil(eta)))
    tx, ty = future_position(target_initial, current_step, eta_i, angular_velocity)
    angle = math.atan2(ty - sy, tx - sx)
    return {
        "angle": angle,
        "eta": eta_i,
        "pred_x": tx,
        "pred_y": ty,
        "speed": speed,
    }


def required_ships_to_capture(target_planet, eta):
    """Garrison the target will have at arrival + 1 (ignores other fleets)."""
    ships = target_planet[5]
    prod = target_planet[6]
    return int(ships + prod * eta + 1)


def predict_fleet_landing(fleet, planets, initial_planets, angular_velocity,
                          current_step, max_t=400):
    """Simulate a fleet's straight-line path; return first planet-hit as
    {target_id, eta}, or None if it exits the board or hits the sun.

    Approximation: checks every integer turn. Fast enough for ~50 fleets.
    """
    _fid, _owner, x, y, angle, from_pid, ships = fleet
    speed = fleet_speed(ships)
    vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
    init_by_id = {p[0]: p for p in initial_planets}
    for t in range(1, max_t + 1):
        cx = x + vx * t
        cy = y + vy * t
        if cx < 0 or cx > BOARD or cy < 0 or cy > BOARD:
            return None
        if math.hypot(cx - SUN[0], cy - SUN[1]) < SUN_R:
            return None
        for p in planets:
            pid, _, _, _, pr = p[:5]
            if pid == from_pid and t <= 2:
                continue  # don't re-match source immediately after spawn
            ip = init_by_id.get(pid)
            if ip is None:
                continue
            fx, fy = future_position(ip, current_step, t, angular_velocity)
            if math.hypot(cx - fx, cy - fy) <= pr:
                return {"target_id": pid, "eta": t}
    return None


def threats_per_planet(fleets, planets, initial_planets, angular_velocity, current_step):
    """Map planet_id -> sorted list of {eta, owner, ships, fleet_id} hitting it."""
    obs = {
        "step": current_step,
        "planets": planets,
        "fleets": fleets,
        "initial_planets": initial_planets,
        "angular_velocity": angular_velocity,
        "comets": [],
        "comet_planet_ids": [],
    }
    radar = Radar(obs)
    out = {}
    for f in fleets:
        pred = radar.simulate_fleet(f)
        if not pred.hit_planet:
            continue
        out.setdefault(pred.target_id, []).append({
            "eta": pred.eta,
            "owner": f[1],
            "ships": f[6],
            "fleet_id": f[0],
        })
    for k in out:
        out[k].sort(key=lambda x: x["eta"])
    return out


def ring_order(planets, initial_planets, include_orbiting=False):
    """Planet IDs sorted by polar angle around the sun (CCW from +x axis).

    By default only static outer planets are included. Orbiting planets
    move so their ring position is time-varying; pass include_orbiting=True
    to include them at their current angle.
    """
    init_by_id = {p[0]: p for p in initial_planets}
    entries = []
    for p in planets:
        pid = p[0]
        ip = init_by_id.get(pid)
        if ip is None:
            continue
        if not include_orbiting and is_orbiting(ip):
            continue
        phi = math.atan2(p[3] - SUN[1], p[2] - SUN[0])
        entries.append((phi, pid))
    entries.sort()
    return [pid for _, pid in entries]
