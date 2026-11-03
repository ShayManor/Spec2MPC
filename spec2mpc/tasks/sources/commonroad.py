"""CommonRoad motion-planning scenarios for road vehicles (kart family, the only car-like family).

Assets: <ASSET_ROOT>/commonroad/<repo>/... scenario XMLs (2018b and 2020a formats) from the public CommonRoad
GitHub repos (route planner, reactive planner, CriMe). The XML is parsed here with the standard library; no
commonroad-io. Each planning problem is one task: the robot starts at the initial state, must be inside the goal
region at the end of the goal time interval (the episode horizon), and must never touch another traffic participant.
Lanelets give the route (start lanelet to goal lanelet over successors and same-direction neighbours). Coordinates
are recentred on the initial position. Moving obstacles are mocap bodies; `trajectories()` gives their states.
"""
from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np

from spec2mpc.core import Entity, Scene, TaskCard, TaskPublic
from spec2mpc.tasks.sources import Translation, pred, write_coverage
from spec2mpc.tasks.sources import _nav as N

SOURCE = "commonroad"
ROOT = N.ASSET_ROOT / "commonroad"
OBST_HALF_H_M = 0.75
AT_POSE_Z_SLACK_M = 0.5  # goal pose sits on the ground, the base frame above it
GOAL_SNAP_M = 3.0  # goal center may sit this far from a lanelet
EGO_DUP_M = 0.5  # an obstacle starting this close to the ego start is the ego itself
MAX_HORIZON_S = 600.0  # longest `timeout` a program can express
KART_VMAX_MPS = 25.0  # top of the v1 kart v_max range; longer routes cannot be driven within the horizon
FAMILY, MEMBER = "kart", "kart"

STYLES = ["Drive to the destination and do not hit any other road user.",
          "Get to the goal area in traffic without a collision.",
          "Follow the road to the destination, keep clear of the other vehicles and pedestrians."]


class Skip(Exception):
    pass


def available() -> bool:
    return bool(scenario_files())


@functools.lru_cache(maxsize=None)
def scenario_files() -> dict[str, str]:
    # benchmark id -> xml path, deduplicated across repos
    out: dict[str, str] = {}
    for p in sorted(ROOT.glob("*/*/*.xml")) + sorted(ROOT.glob("*/*.xml")):
        if ".git" in p.parts:
            continue
        head = p.read_bytes()[:400]
        if b"<commonRoad" not in head:
            continue  # git-lfs pointer or another xml
        out.setdefault(p.stem, str(p))
    return out


def _num(el: ET.Element | None, tag: str) -> float | None:
    if el is None:
        return None
    e = el.find(tag)
    if e is None:
        return None
    ex = e.find("exact")
    return float((ex if ex is not None else e).text)


def _interval(el: ET.Element | None, tag: str) -> tuple[float, float] | None:
    e = None if el is None else el.find(tag)
    if e is None:
        return None
    if e.find("exact") is not None:
        v = float(e.find("exact").text)
        return v, v
    return float(e.find("intervalStart").text), float(e.find("intervalEnd").text)


def _point(el: ET.Element) -> np.ndarray:
    return np.array([float(el.find("x").text), float(el.find("y").text)])


def _shape_poly(el: ET.Element) -> np.ndarray:
    # first rectangle, circle, or polygon under el -> [K, 2] polygon
    for s in el:
        if s.tag == "rectangle":
            c = s.find("center")
            ctr = _point(c) if c is not None else np.zeros(2)
            yaw = float(s.find("orientation").text) if s.find("orientation") is not None else 0.0
            return N.rect(ctr, float(s.find("length").text) / 2, float(s.find("width").text) / 2, yaw)[:, :2]
        if s.tag == "circle":
            c = s.find("center")
            return N.circle(_point(c) if c is not None else np.zeros(2), float(s.find("radius").text))[:, :2]
        if s.tag == "polygon":
            return np.array([_point(p) for p in s.findall("point")])
    raise Skip("shape")


def _state_xy(st: ET.Element) -> np.ndarray:
    pos = st.find("position")
    if pos.find("point") is not None:
        return _point(pos.find("point"))
    raise Skip("uncertain_state")


@functools.lru_cache(maxsize=None)
def parse(bench: str) -> dict:
    """Lanelets, obstacles, planning problems, and dt of one scenario (raw, scenario coordinates)."""
    root = ET.parse(scenario_files()[bench]).getroot()
    dt = float(root.get("timeStepSize", "0.1"))
    lanelets = {}
    for ll in root.findall("lanelet"):
        left = np.array([_point(p) for p in ll.find("leftBound").findall("point")])
        right = np.array([_point(p) for p in ll.find("rightBound").findall("point")])
        adj = [a.get("ref") for t in ("adjacentLeft", "adjacentRight") for a in ll.findall(t)
               if a.get("drivingDir") == "same"]
        lanelets[ll.get("id")] = {"left": left, "right": right, "succ": [s.get("ref") for s in ll.findall("successor")],
                                  "adj": adj}
    obstacles = []
    for ob in list(root):
        role = {"staticObstacle": "static", "dynamicObstacle": "dynamic"}.get(ob.tag)
        if ob.tag == "obstacle":
            role = ob.find("role").text.strip()
        if role not in ("static", "dynamic"):
            continue
        if ob.find("setBasedPrediction") is not None or ob.find("occupancySet") is not None:
            obstacles.append({"skip": "set_based_prediction"})
            continue
        ini = ob.find("initialState")
        traj = []
        try:
            traj.append((_num(ini, "time") or 0.0, *_state_xy(ini), _num(ini, "orientation") or 0.0))
            for st in (ob.find("trajectory") if ob.find("trajectory") is not None else []):
                traj.append((_num(st, "time"), *_state_xy(st), _num(st, "orientation") or 0.0))
            poly = _shape_poly(ob.find("shape"))
        except Skip as e:
            obstacles.append({"skip": str(e)})
            continue
        tr = np.array(traj, np.float64)
        tr[:, 0] *= dt  # steps -> s
        obstacles.append({"id": ob.get("id"), "type": ob.find("type").text.strip(), "role": role, "traj": tr,
                          "half": np.r_[np.abs(poly).max(0), OBST_HALF_H_M]})
    problems = []
    for pp in root.findall("planningProblem"):
        ini = pp.find("initialState")
        problems.append({"id": pp.get("id"), "start": _state_xy(ini), "yaw": _num(ini, "orientation") or 0.0,
                         "v0": _num(ini, "velocity") or 0.0, "goals": pp.findall("goalState")})
    return {"dt": dt, "lanelets": lanelets, "obstacles": obstacles, "problems": problems}


def _lanelet_poly(ll: dict) -> np.ndarray:
    return np.vstack([ll["left"], ll["right"][::-1]])


def _inside(xy: np.ndarray, poly: np.ndarray) -> bool:
    x, y = xy
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    c = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / np.where(y1 == y0, 1e-12, y1 - y0) + x0)
    return bool(np.sum(c) % 2)


def _centerline(ll: dict) -> np.ndarray:
    n = max(len(ll["left"]), len(ll["right"]))
    s = np.linspace(0, 1, n)

    def rs(p: np.ndarray) -> np.ndarray:
        t = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
        t = t / max(t[-1], 1e-9)
        return np.stack([np.interp(s, t, p[:, 0]), np.interp(s, t, p[:, 1])], 1)

    return 0.5 * (rs(ll["left"]) + rs(ll["right"]))


def _containing(lls: dict, xy: np.ndarray, snap: float = 0.0) -> list[str]:
    ids = [k for k, ll in lls.items() if _inside(xy, _lanelet_poly(ll))]
    if not ids and snap > 0:
        d = {k: float(np.min(np.linalg.norm(_centerline(ll) - xy, axis=1))) for k, ll in lls.items()}
        ids = [k for k, v in d.items() if v <= snap]
    return ids


def _chain(lls: dict, ids: list[str]) -> list[str] | None:
    # order lanelet ids into one successor chain, or None
    if len(ids) == 1:
        return ids
    heads = [i for i in ids if not any(i in lls[j]["succ"] for j in ids if j != i)]
    if len(heads) != 1:
        return None
    out = heads
    while len(out) < len(ids):
        nxt = [s for s in lls[out[-1]]["succ"] if s in ids and s not in out]
        if not nxt:
            return None
        out.append(nxt[0])
    return out


def _route(lls: dict, starts: list[str], goals: set[str]) -> list[str] | None:
    # BFS over successors (preferred) and same-direction neighbours
    prev: dict[str, str | None] = {s: None for s in starts}
    q = deque(starts)
    while q:
        u = q.popleft()
        if u in goals:
            out = [u]
            while prev[out[-1]] is not None:
                out.append(prev[out[-1]])
            return out[::-1]
        for v in lls[u]["succ"] + lls[u]["adj"]:
            if v in lls and v not in prev:
                prev[v] = u
                q.append(v)
    return None


def _route_line(lls: dict, ids: list[str]) -> np.ndarray:
    # centerlines in order; a same-direction lane change switches lanes at mid-lanelet instead of backtracking
    segs = [_centerline(lls[ids[0]])]
    for u, v in zip(ids[:-1], ids[1:]):
        cl = _centerline(lls[v])
        if v in lls[u]["succ"]:
            segs.append(cl)
        else:
            segs[-1] = segs[-1][:max(len(segs[-1]) // 2, 1)]
            segs.append(cl[len(cl) // 2:])
    return np.vstack(segs)


def _goal(sc: dict, g: ET.Element) -> tuple[np.ndarray, tuple[float, float] | None, tuple[float, float], list[str]]:
    # (goal polygon [K, 2], orientation interval, time interval in s, goal lanelet ids)
    lls = sc["lanelets"]
    pos = g.find("position")
    if pos is None:
        raise Skip("no_goal_position")
    refs = [e.get("ref") for e in pos.findall("lanelet")]
    if refs:
        ch = _chain(lls, refs)
        if ch is None:
            raise Skip("goal_lanelet_union")
        poly = np.vstack([np.vstack([lls[i]["left"] for i in ch]), np.vstack([lls[i]["right"] for i in ch])[::-1]])
        goal_ids = ch
    else:
        shapes = [s for s in pos if s.tag in ("rectangle", "circle", "polygon")]
        if len(shapes) != 1:
            raise Skip("goal_shape_union")
        poly = _shape_poly(pos)
        goal_ids = _containing(lls, poly.mean(0), GOAL_SNAP_M)
    t = _interval(g, "time")
    if t is None:
        raise Skip("no_goal_time")
    return poly, _interval(g, "orientation"), (t[0] * sc["dt"], t[1] * sc["dt"]), goal_ids


@functools.lru_cache(maxsize=None)
def _problem(bench: str, ppid: str) -> dict:
    """Everything one task needs, recentred on the start; raises Skip when untranslatable."""
    sc = parse(bench)
    pp = next(p for p in sc["problems"] if p["id"] == ppid)
    if any("skip" in o for o in sc["obstacles"]):
        raise Skip(next(o["skip"] for o in sc["obstacles"] if "skip" in o))
    if len(pp["goals"]) != 1:
        raise Skip("goal_disjunction" if pp["goals"] else "no_goal_state")
    poly, ori, (t0, t1), goal_ids = _goal(sc, pp["goals"][0])
    if t1 <= 0:
        raise Skip("goal_time_zero")
    lls = sc["lanelets"]
    starts = _containing(lls, pp["start"], GOAL_SNAP_M)
    if not starts:
        raise Skip("start_off_road")
    rt_ids = _route(lls, starts, set(goal_ids)) if goal_ids else None
    if rt_ids is None:
        raise Skip("no_route")
    if t1 > MAX_HORIZON_S:
        raise Skip("horizon_over_program_timeout")
    org = pp["start"]
    rt = _route_line(lls, rt_ids)
    i0 = int(np.argmin(np.linalg.norm(rt - org, axis=1)))
    inside = [j for j in range(i0 + 1, len(rt)) if _inside(rt[j], poly)]
    i1 = inside[0] if inside else i0 + 1 + int(np.argmin(np.linalg.norm(rt[i0 + 1:] - poly.mean(0), axis=1)))
    rt = rt[i0:i1 + 1]
    if not inside:
        rt = np.vstack([rt, poly.mean(0)])  # lead the route into an off-centerline goal
    if N.arclen(rt) > KART_VMAX_MPS * t1:
        raise Skip("route_too_long_for_horizon")
    # the scenario's ego vehicle is sometimes listed as an obstacle too
    obstacles = [o for o in sc["obstacles"] if np.linalg.norm(o["traj"][0, 1:3] - org) > EGO_DUP_M]
    dropped = ["road_boundary", "speed_limit"]
    if t0 > 0:
        dropped.append("goal_time_lower_bound")
    if t1 > t0:
        dropped.append("goal_time_interval: judged at its end")
    if _interval(pp["goals"][0], "velocity") is not None:
        dropped.append("goal_velocity")
    if pp["v0"] > 0.1:
        dropped.append("initial_velocity")
    return {"org": org, "yaw": pp["yaw"], "goal": poly - org, "ori": ori, "t_end_s": t1, "route": rt - org,
            "obstacles": obstacles, "dropped": dropped}


def tasks() -> tuple[dict[str, dict], dict[str, str]]:
    # (translated task -> problem, skipped task -> reason) over every planning problem
    ok, skipped = {}, {}
    for bench in scenario_files():
        try:
            pps = [p["id"] for p in parse(bench)["problems"]]
        except Skip as e:
            skipped[bench] = str(e)
            continue
        for ppid in pps:
            name = f"{bench}:{ppid}"
            try:
                ok[name] = _problem(bench, ppid)
            except Skip as e:
                skipped[name] = str(e)
    return ok, skipped


def _ent_id(o: dict) -> str:
    return f"{o['type']}_{o['id']}"


def _build(pub: TaskPublic) -> tuple[str, Scene, dict]:
    _, task, _ = N.parse_task_id(pub.task_id)
    bench, ppid = task.split(":")
    pr = _problem(bench, ppid)
    org = pr["org"]
    w = N.World(f"commonroad_{bench}", floor_half_m=float(np.abs(pr["route"]).max() + 200))
    ents = [N.start([0, 0, 0], pr["yaw"]),
            N.region("goal", np.c_[pr["goal"], np.zeros(len(pr["goal"]))],
                     yaw=0.0 if pr["ori"] is None else 0.5 * sum(pr["ori"])),
            N.path("route", np.c_[pr["route"], np.zeros(len(pr["route"]))])]
    for o in pr["obstacles"]:
        tr = o["traj"]
        xyz = np.c_[tr[:, 1:3] - org, np.full(len(tr), OBST_HALF_H_M)]
        eid = _ent_id(o)
        if o["role"] == "dynamic":
            w.mocap_box(eid, list(o["half"]), list(xyz[0]), float(tr[0, 3]))
        else:
            w.geom(eid, "box", list(o["half"]), list(xyz[0]), float(tr[0, 3]), "0.5 0.5 0.6 1")
        ents.append(Entity(eid, o["type"], "object", N.pose7(xyz[0], float(tr[0, 3])), o["half"].copy(),
                           xyz if o["role"] == "dynamic" else None, None))
    return w.xml(), N.scene_of(ents), pr


def scene(card: TaskCard | TaskPublic) -> tuple[str, Scene]:
    """MJCF world xml (no robot) and the Scene at t = 0 for a commonroad card."""
    xml, sc, _ = _build(card.public if isinstance(card, TaskCard) else card)
    return xml, sc


def trajectories(card: TaskCard | TaskPublic) -> dict[str, np.ndarray]:
    """Moving obstacles: entity id -> [T, 4] (t_s, x, y, yaw) in the recentred frame, for the sim's mocap bodies."""
    pub = card.public if isinstance(card, TaskCard) else card
    bench, ppid = N.parse_task_id(pub.task_id)[1].split(":")
    pr = _problem(bench, ppid)
    out = {}
    for o in pr["obstacles"]:
        if o["role"] == "dynamic":
            tr = o["traj"].copy()
            tr[:, 1:3] -= pr["org"]
            out[_ent_id(o)] = tr
    return out


def _test(pr: dict) -> tuple[list[dict], list[dict], float]:
    preds = [pred("in_region", "base", "goal")]
    if pr["ori"] is not None:
        r = float(np.max(np.linalg.norm(pr["goal"] - pr["goal"].mean(0), axis=1)))
        preds.append(pred("at_pose", "base", "goal", tol_m=round(r + AT_POSE_Z_SLACK_M, 3),
                          tol_rad=round(0.5 * (pr["ori"][1] - pr["ori"][0]), 4)))
    preds += [pred("no_contact", "robot", _ent_id(o)) for o in pr["obstacles"]]
    graded = [{"term": "accel", "args": ["base"], "good": 2.0, "bad": 6.0}]
    return preds, graded, pr["t_end_s"]


def coverage() -> dict:
    ok, skipped = tasks()
    tr = [Translation(k, _test(pr)[0], ["base", "goal"], STYLES[0], "drive_to", pr["t_end_s"], pr["dropped"], "kart")
          for k, pr in ok.items()]
    return write_coverage(SOURCE, sorted(ok) + sorted(skipped), tr, skipped,
                          {"scenario_files": len(scenario_files())})


def cards(n: int, seed: int) -> list[TaskCard]:
    """n cards cycling through the translated planning problems (seeded order, randomized kart)."""
    if not available():
        raise RuntimeError(f"CommonRoad scenarios missing under {ROOT}")
    ok, _ = tasks()
    names = sorted(ok)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    coverage()
    out = []
    for j in range(n):
        task = names[int(order[j % len(names)])]
        sseed = int(rng.integers(0, 2**31 - 1))
        rb = N.robot(FAMILY, MEMBER, sseed)
        preds, graded, horizon = _test(ok[task])
        out.append(N.card(SOURCE, task, sseed, rb, N.pick(rng, STYLES), "drive_to", preds, graded, horizon))
    return out
