"""Graded terms and penalties over an episode trace (trace format: predicates.py docstring).

TaskTest.graded entries use the predicate dict format: {"term": <name>, "args": [...], <param>: <value>}.
Graded terms return a value in [0, 1] through ramp(x, good, bad): 1 at or past `good`, 0 at or past `bad`,
linear between; good < bad means lower is better, good > bad means higher is better.
Penalty terms (names in PENALTIES) return a raw magnitude >= 0; score() subtracts weight * magnitude.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from spec2mpc.rubric.v2.predicates import completion_index, contacts, match, not_fallen, pose, tilt_rad, touching

Graded = Callable[[dict, list[str], dict, list[dict]], float]  # (trace, args, params, predicates) -> value


def ramp(x: float, good: float, bad: float) -> float:
    good, bad = float(good), float(bad)
    if good == bad:
        return float(x <= good)  # degenerate: a step, lower is better
    return float(np.clip((x - bad) / (good - bad), 0.0, 1.0))


def _derivs(tr: dict, id: str, n: int) -> np.ndarray:
    # n-th time derivative of an entity position, [T - n, 3]
    t = np.asarray(tr["t_s"], np.float64)
    x = pose(tr, id)[:, :3]
    for _ in range(n):
        if len(t) < 2:
            return np.zeros((0, 3))
        x = np.diff(x, axis=0) / np.maximum(np.diff(t), 1e-9)[:, None]
        t = 0.5 * (t[1:] + t[:-1])
    return x


def _peak(v: np.ndarray) -> float:
    return float(np.linalg.norm(v, axis=1).max()) if len(v) else 0.0


def time(tr: dict, args: list[str], p: dict, preds: list[dict]) -> float:
    # time at which the predicates became true for good; 0 if they never did
    i = completion_index(preds, tr)
    return 0.0 if i is None else ramp(float(tr["t_s"][i]), p["good"], p["bad"])


def speed(tr: dict, args: list[str], p: dict, preds: list[dict]) -> float:
    return ramp(_peak(_derivs(tr, args[0], 1)), p["good"], p["bad"])


def accel(tr: dict, args: list[str], p: dict, preds: list[dict]) -> float:
    return ramp(_peak(_derivs(tr, args[0], 2)), p["good"], p["bad"])


def jerk(tr: dict, args: list[str], p: dict, preds: list[dict]) -> float:
    j = _derivs(tr, args[0], 3)
    rms = float(np.sqrt(np.mean(np.sum(j * j, axis=1)))) if len(j) else 0.0
    return ramp(rms, p["good"], p["bad"])


def tilt(tr: dict, args: list[str], p: dict, preds: list[dict]) -> float:
    return ramp(float(tilt_rad(pose(tr, args[0])[:, 3:]).max()), p["good"], p["bad"])


def force(tr: dict, args: list[str], p: dict, preds: list[dict]) -> float:
    # peak contact force between a and b (contacts without a force count as 0 N)
    a, b = args
    f = [float(c[2]) for cs in contacts(tr) for c in cs if len(c) > 2
         and ((match(c[0], a) and match(c[1], b)) or (match(c[0], b) and match(c[1], a)))]
    return ramp(max(f, default=0.0), p["good"], p["bad"])


def clearance(tr: dict, args: list[str], p: dict, preds: list[dict]) -> float:
    return ramp(float(np.min(tr["min_clearance_m"])), p["good"], p["bad"])


def energy(tr: dict, args: list[str], p: dict, preds: list[dict]) -> float:
    return ramp(float(np.asarray(tr["energy_J"])[-1]), p["good"], p["bad"])


GRADED: dict[str, Graded] = {
    "time": time, "speed": speed, "accel": accel, "jerk": jerk, "tilt": tilt, "force": force,
    "clearance": clearance, "energy": energy,
}


# ---- penalties: raw magnitudes >= 0 ----

def contact_events(tr: dict, args: list[str], p: dict, preds: list[dict]) -> float:
    # number of contact onsets between a and b
    c = touching(tr, *args).astype(int)
    return float(np.sum(np.diff(np.concatenate([[0], c])) == 1))


def fell(tr: dict, args: list[str], p: dict, preds: list[dict]) -> float:
    return float(not not_fallen(tr, args, p)[-1])


def count(tr: dict, args: list[str], p: dict, preds: list[dict]) -> float:
    # final value of a cumulative counter key, e.g. solver_failures
    return float(max(np.asarray(tr[args[0]], np.float64)[-1], 0.0))


PENALTIES: dict[str, Graded] = {"contact_events": contact_events, "fell": fell, "count": count}
