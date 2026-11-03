"""Program adapter: per-stage limits from `limit` terms, clipped to the family safety floors.

Floors come from configs/v2/robots/<family>.yaml and no program can change them: a requested fraction above a
floor is clipped down, a clearance margin below the floor is raised, and a stage with no `limit` term runs at
the floor.
"""
from __future__ import annotations

import dataclasses

from spec2mpc.core import RobotSpec, Stage, Term

# limit(quantity) -> the floor key that caps its fraction
_FLOOR_OF = {"speed": "max_speed_frac", "effort": "max_force_frac", "force": "max_force_frac"}


def physical_limits(robot: RobotSpec, nominal: dict[str, float] | None = None) -> dict[str, float]:
    # absolute limits the fractions refer to: robot.limits when it states them, else the family nominal
    # (configs/v2/mpc/sampling.yaml: nominal_limits; RobotSpec.limits only lists actuator ranges)
    if nominal is None:
        from spec2mpc.mpc.sampling.solver import load_cfg
        nominal = load_cfg(robot.family)["nominal_limits"]
    return {k: float(robot.limits.get(k, v)) for k, v in nominal.items()}


def clip_to_floors(params: dict, robot: RobotSpec) -> dict:
    # params: {speed, accel, jerk, effort, force, tilt: fraction of the robot limit; margin_m: clearance, m}
    # -> the same keys with floors applied; speed, effort, force default to their floor
    f = robot.floors
    out = dict(params)
    for q, fk in _FLOOR_OF.items():
        if fk in f:
            out[q] = min(float(out.get(q, f[fk])), float(f[fk]))
    if "max_tilt_rad" in f:
        cap = float(f["max_tilt_rad"]) / physical_limits(robot)["tilt_rad"]
        out["tilt"] = min(float(out.get("tilt", cap)), cap)
    if "min_clearance_m" in f and "margin_m" in out:
        out["margin_m"] = max(float(out["margin_m"]), float(f["min_clearance_m"]))
    return out


def stage_limits(stage: Stage, robot: RobotSpec) -> tuple[dict, bool]:
    # -> (limits after floors, True if any requested value was clipped)
    req: dict = {}
    for t in stage.terms:
        if t.type == "limit" and t.args:
            req[t.args[0]] = float(t.params.get("value", 1.0))
    out = clip_to_floors(req, robot)
    clipped = any(k in out and out[k] != v for k, v in req.items())
    return out, clipped


def clip_term(t: Term, robot: RobotSpec) -> tuple[Term, bool]:
    # clearance margins below the floor are raised; every other term passes unchanged
    if t.type != "clearance" or "min_clearance_m" not in robot.floors:
        return t, False
    m = float(t.params.get("margin_m", 0.0))
    lo = float(robot.floors["min_clearance_m"])
    if m >= lo:
        return t, False
    return dataclasses.replace(t, params={**t.params, "margin_m": lo}), True
