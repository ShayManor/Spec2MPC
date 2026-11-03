"""Success predicates over an episode trace (rubric side: hidden from executors).

Trace format (Episode.trace, produced by the v2 SimEnv; T = number of logged steps):
  t_s                  [T]     float, sim time of each step, s
  entity_pose/<id>     [T, 7]  float, xyz (m) + quaternion wxyz, world frame; every object, fixture,
                               and robot frame a test names (robot frames use RobotSpec.frames ids, e.g. "ee")
  entity_extent/<id>   [3]     float, box half-sizes in the entity frame, m (static); needed by on, inside,
                               and box regions
  entity_points/<id>   [N, 3]  float, world-frame polygon vertices of a region (static); in_region uses xy
  open_frac/<id>       [T]     float in [0, 1], opening fraction of an articulated entity (drawer, door, lid)
  contacts             [T]     list (or object ndarray) of per-step lists of (id_a, id_b) or (id_a, id_b, force_N)
                               tuples; ids are entity ids, robot links are "robot:<link>" ("robot" matches all)
  robot_base_height    [T]     float, base height above ground, m
  robot_base_pose      [T, 7]  float, optional, base pose; not_fallen uses it when max_tilt_rad is given
  min_clearance_m      [T]     float, optional, robot-to-scene minimum distance (graded "clearance")
  energy_J             [T]     float, optional, cumulative actuator energy (graded "energy")
  <counter>            [T]     float, optional, cumulative event counters, e.g. solver_failures, floor_clips
                               (penalty "count")

Predicate dict format (TaskTest.predicates entries), flat:
  {"pred": <name>, "args": [<entity ids>], <param>: <value>, ...}
The nested form {"type": <name>, "args": [...], "params": {...}} is accepted too.

Each predicate maps a trace to a per-step bool array [T]. State predicates (on, inside, open_frac, upright,
at_pose, in_region, held) are judged at the last step. Invariants (not_fallen, no_contact) are cumulative:
step t is true only if the condition held at every step up to t. time_below(t_s) is true iff the other state
predicates became true and stayed true from a step whose time is below t_s (alone: the episode ended before t_s).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

Pred = Callable[[dict, list[str], dict], np.ndarray]  # (trace, args, params) -> [T] bool

DEFAULT_TOL_M = 0.02


def normalize(d: dict) -> tuple[str, list[str], dict[str, Any]]:
    # accept {"pred"|"term"|"type": name, "args": [...], params flat or under "params"}
    name = d.get("pred", d.get("term", d.get("type")))
    if name is None:
        raise ValueError(f"no predicate name in {d!r}")
    params = dict(d["params"]) if "params" in d else {k: v for k, v in d.items()
                                                      if k not in ("pred", "term", "type", "args")}
    return str(name), list(d.get("args", [])), params


# ---- geometry helpers ----

def pose(tr: dict, id: str) -> np.ndarray:
    k = f"entity_pose/{id}"
    if k not in tr:
        raise KeyError(f"trace has no {k!r}")
    return np.asarray(tr[k], np.float64).reshape(-1, 7)  # [T, 7]


def extent(tr: dict, id: str) -> np.ndarray:
    k = f"entity_extent/{id}"
    if k not in tr:
        raise KeyError(f"trace has no {k!r}")
    return np.asarray(tr[k], np.float64).reshape(3)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    # q: [..., 4] wxyz -> [..., 3, 3]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def tilt_rad(q: np.ndarray) -> np.ndarray:
    # angle between the body z axis and world z; q: [T, 4] -> [T]
    return np.arccos(np.clip(quat_to_mat(q)[..., 2, 2], -1.0, 1.0))


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # rotation angle between orientations; [T, 4], [T, 4] or [4] -> [T]
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    return 2 * np.arccos(np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0))


def to_local(p: np.ndarray, frame: np.ndarray) -> np.ndarray:
    # world points p [T, 3] into frame poses [T, 7] -> [T, 3]
    R = quat_to_mat(frame[:, 3:])  # [T, 3, 3]
    return np.einsum("tji,tj->ti", R, p - frame[:, :3])


def half_height(q: np.ndarray, ext: np.ndarray) -> np.ndarray:
    # world-z half-extent of a box with half-sizes ext [3] at orientation q [T, 4] -> [T]
    return np.abs(quat_to_mat(q)[:, 2, :]) @ ext


def steps(tr: dict) -> int:
    return len(tr["t_s"])


def contacts(tr: dict) -> list[list[tuple]]:
    return [list(c) for c in tr["contacts"]]


def match(cid: str, id: str) -> bool:
    # "robot" matches "robot:<link>"; ids otherwise match exactly
    return cid == id or cid.startswith(id + ":")


def touching(tr: dict, a: str, b: str) -> np.ndarray:
    # [T] bool: any contact between a and b at each step
    return np.array([any((match(c[0], a) and match(c[1], b)) or (match(c[0], b) and match(c[1], a))
                         for c in cs) for cs in contacts(tr)], bool)


def _always(x: np.ndarray) -> np.ndarray:
    # cumulative "held at every step so far"
    return np.logical_and.accumulate(x)


# ---- predicates ----

def on(tr: dict, args: list[str], p: dict) -> np.ndarray:
    a, b = args
    pa, pb, ea, eb = pose(tr, a), pose(tr, b), extent(tr, a), extent(tr, b)
    loc = to_local(pa[:, :3], pb)
    in_xy = (np.abs(loc[:, 0]) <= eb[0]) & (np.abs(loc[:, 1]) <= eb[1])
    gap = (pa[:, 2] - half_height(pa[:, 3:], ea)) - (pb[:, 2] + half_height(pb[:, 3:], eb))
    return in_xy & (np.abs(gap) <= float(p.get("tol_m", DEFAULT_TOL_M)))


def inside(tr: dict, args: list[str], p: dict) -> np.ndarray:
    a, b = args
    loc = to_local(pose(tr, a)[:, :3], pose(tr, b))
    return np.all(np.abs(loc) <= extent(tr, b) + float(p.get("tol_m", 0.0)), axis=1)


def open_frac(tr: dict, args: list[str], p: dict) -> np.ndarray:
    (a,) = args
    k = f"open_frac/{a}"
    if k not in tr:
        raise KeyError(f"trace has no {k!r}")
    return np.asarray(tr[k], np.float64) >= float(p["min"])


def upright(tr: dict, args: list[str], p: dict) -> np.ndarray:
    (a,) = args
    return tilt_rad(pose(tr, a)[:, 3:]) <= float(p["tol_rad"])


def at_pose(tr: dict, args: list[str], p: dict) -> np.ndarray:
    # target from params "pose" ([3] or [7]) or from a second entity id
    pa = pose(tr, args[0])
    tgt = pose(tr, args[1]) if len(args) > 1 else np.asarray(p["pose"], np.float64)
    tgt = np.broadcast_to(tgt, (len(pa), tgt.shape[-1]))
    ok = np.linalg.norm(pa[:, :3] - tgt[:, :3], axis=1) <= float(p.get("tol_m", DEFAULT_TOL_M))
    if "tol_rad" in p and tgt.shape[-1] == 7:
        ok &= quat_angle(pa[:, 3:], tgt[:, 3:]) <= float(p["tol_rad"])
    return ok


def _in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    # even-odd ray casting; xy [T, 2], poly [N, 2] -> [T]
    x, y = xy[:, 0:1], xy[:, 1:2]
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / np.where(y1 == y0, 1e-12, y1 - y0) + x0)
    return np.sum(crosses, axis=1) % 2 == 1


def in_region(tr: dict, args: list[str], p: dict) -> np.ndarray:
    # region: polygon in entity_points/<r> (xy only), else an oriented box from pose + extent
    a, r = args
    pa = pose(tr, a)[:, :3]
    if f"entity_points/{r}" in tr:
        return _in_polygon(pa[:, :2], np.asarray(tr[f"entity_points/{r}"], np.float64)[:, :2])
    return inside(tr, [a, r], p)


def not_fallen(tr: dict, args: list[str], p: dict) -> np.ndarray:
    h = np.asarray(tr["robot_base_height"], np.float64)
    ok = h >= float(p.get("min_height_m", 0.5 * h[0]))
    if "max_tilt_rad" in p:
        ok &= tilt_rad(np.asarray(tr["robot_base_pose"], np.float64)[:, 3:]) <= float(p["max_tilt_rad"])
    return _always(ok)


def time_below(tr: dict, args: list[str], p: dict) -> np.ndarray:
    # standalone meaning: the clock is still below t_s; evaluate() replaces it with completion time
    return np.asarray(tr["t_s"], np.float64) < float(p["t_s"])


def no_contact(tr: dict, args: list[str], p: dict) -> np.ndarray:
    a, b = args
    return _always(~touching(tr, a, b))


def held(tr: dict, args: list[str], p: dict) -> np.ndarray:
    # touched by the robot and by nothing else (lifted off every support)
    (a,) = args
    out = np.zeros(steps(tr), bool)
    for t, cs in enumerate(contacts(tr)):
        others = [c[1] if match(c[0], a) else c[0] for c in cs if match(c[0], a) or match(c[1], a)]
        out[t] = bool(others) and all(match(o, "robot") for o in others)
    return out


PREDICATES: dict[str, Pred] = {
    "on": on, "inside": inside, "open_frac": open_frac, "upright": upright, "at_pose": at_pose,
    "in_region": in_region, "not_fallen": not_fallen, "time_below": time_below, "no_contact": no_contact,
    "held": held,
}
INVARIANTS = {"not_fallen", "no_contact"}


def run(d: dict, tr: dict) -> np.ndarray:
    name, args, params = normalize(d)
    if name not in PREDICATES:
        raise ValueError(f"unknown predicate {name!r}")
    return np.asarray(PREDICATES[name](tr, args, params), bool)  # [T]


def completion_index(predicates: list[dict], tr: dict) -> int | None:
    # first step from which every state predicate holds through the end; None if they fail at the end
    st = [run(d, tr) for d in predicates if normalize(d)[0] not in INVARIANTS | {"time_below"}]
    if not st:
        return steps(tr) - 1
    allv = np.logical_and.reduce(st)
    if not allv[-1]:
        return None
    bad = np.flatnonzero(~allv)
    return int(bad[-1] + 1) if len(bad) else 0


def evaluate(predicates: list[dict], tr: dict) -> list[bool]:
    """End-of-episode truth value of each predicate, in order."""
    out: list[bool] = []
    done: Any = ...
    for d in predicates:
        name, _, params = normalize(d)
        if name == "time_below" and len(predicates) > 1:
            if done is ...:
                done = completion_index(predicates, tr)
            out.append(done is not None and float(tr["t_s"][done]) < float(params["t_s"]))
        else:
            out.append(bool(run(d, tr)[-1]))
    return out
