"""Goal types: each builds a hidden TaskTest (Task 4 vocabulary) for one scene, robot, and style.

Metadata (families, scene kinds, templates, style phrases, motion scales) comes from configs/v2/goals.yaml; the
builders below pick the entities a goal is about and compose predicates from spec2mpc.rubric.v2. Robot frames are
referenced only by the canonical names of Task 3 ("base", "ee", ...); "robot" means every robot link.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from spec2mpc.core import RobotSpec, Scene, TaskTest
from spec2mpc.rubric.v2 import GRADED, PENALTIES, PREDICATES
from spec2mpc.tasks.scenes import DRAWER_TRAVEL_M, layout

GOALS_CFG = Path(__file__).parent.parent.parent / "configs" / "v2" / "goals.yaml"
CFG = yaml.safe_load(GOALS_CFG.read_text())
STYLES: dict[str, list[str]] = CFG["styles"]
LEGGED = ("quadruped", "humanoid")


@dataclass
class Built:
    """What a goal builder decided: hidden predicates plus what the style terms and the command need."""
    predicates: list[dict]
    subject: str  # entity id or robot frame whose motion the style terms grade
    dist_m: float  # nominal travel of the subject, m
    slots: dict[str, str]  # command template slots -> entity labels
    avoid: list[str] = field(default_factory=list)  # ids the robot should not touch (careful style)
    touch: str | None = None  # manipulated object (gentle style grades its contact force)
    graded: list[dict] = field(default_factory=list)  # goal-specific graded terms, every style


Builder = Callable[[Scene, np.random.Generator, RobotSpec], Built]


@dataclass(frozen=True)
class GoalSpec:
    name: str
    families: tuple[str, ...]
    scene_kinds: tuple[str, ...]
    build: Builder
    styles: tuple[str, ...]
    templates: tuple[str, ...]
    overhead_s: float
    horizon_s: float

    def scene_kind(self, scene_seed: int) -> str:
        return self.scene_kinds[scene_seed % len(self.scene_kinds)]

    def make_test(self, scene: Scene, rng: np.random.Generator, robot: RobotSpec, style: str = "neutral",
                  task_id: str = "") -> TaskTest:
        return assemble(self, self.build(scene, rng, robot), robot, style, task_id)


# ---- helpers ----

def _P(name: str, *args: str, **params) -> dict:
    return {"pred": name, "args": list(args), **params}


def _G(term: str, *args: str, **params) -> dict:
    return {"term": term, "args": list(args), **params}


def _pos(scene: Scene, id: str) -> np.ndarray:
    return scene.get(id).pose[:3].copy()


def _label(scene: Scene, id: str) -> str:
    return scene.get(id).label


def _ids(scene: Scene, prefix: str) -> list[str]:
    return [e.id for e in scene.entities if e.id.startswith(prefix)]


def _pick(rng: np.random.Generator, xs: list[str]) -> str:
    return xs[int(rng.integers(len(xs)))]


def _objects(scene: Scene, types: tuple[str, ...]) -> list[str]:
    return [e.id for e in scene.entities if e.kind == "object" and e.id.rsplit("_", 1)[0] in types]


def _xy_dist(scene: Scene, a: str | np.ndarray, b: str | np.ndarray | None = None) -> float:
    pa = _pos(scene, a) if isinstance(a, str) else np.asarray(a)
    pb = np.zeros(3) if b is None else (_pos(scene, b) if isinstance(b, str) else np.asarray(b))
    return float(np.linalg.norm(pa[:2] - pb[:2]))


def _manip_dist(scene: Scene, obj: str, target: str | np.ndarray) -> float:
    # ee from near the base to the object, then object to target
    return _xy_dist(scene, obj) + _xy_dist(scene, obj, target) + 0.2


def _others(scene: Scene, keep: set[str]) -> list[str]:
    return [e.id for e in scene.entities if e.kind == "object" and e.id not in keep]


def _yaw_pose(scene: Scene, id: str, z: float) -> list[float]:
    # [7] target pose at the region centre, its yaw, height z
    e = scene.get(id)
    return [float(e.pose[0]), float(e.pose[1]), float(z), *map(float, e.pose[3:])]


GRASPABLE = ("block", "mug", "can", "box")


# ---- manipulation builders ----

def _reach(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    obj = _pick(r, _objects(s, GRASPABLE))
    tgt = _pos(s, obj) + [0, 0, float(s.get(obj).extent[2]) + 0.1]
    return Built([_P("at_pose", "ee", pose=[float(v) for v in tgt], tol_m=0.03)], "ee",
                 _xy_dist(s, obj) + 0.2, {"obj": _label(s, obj)}, _others(s, set()))


def _pick_up(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    obj = _pick(r, _objects(s, GRASPABLE))
    tgt = _pos(s, obj) + [0, 0, 0.15]
    sup = "counter_0" if any(e.id == "counter_0" for e in s.entities) else "table_0"
    return Built([_P("held", obj), _P("at_pose", obj, pose=[float(v) for v in tgt], tol_m=0.05)], obj,
                 _manip_dist(s, obj, tgt), {"obj": _label(s, obj), "support": _label(s, sup)},
                 _others(s, {obj}), obj)


def _place(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    obj = _pick(r, _objects(s, GRASPABLE))
    preds = [_P("on", obj, "plate_0", tol_m=0.02)] + ([_P("upright", obj, tol_rad=0.3)] if "mug" in obj else [])
    return Built(preds, obj, _manip_dist(s, obj, "plate_0"), {"obj": _label(s, obj), "target": "plate"},
                 _others(s, {obj}), obj)


def _stack(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    a, b = (["block_0", "block_1"] if r.random() < 0.5 else ["block_1", "block_0"])
    return Built([_P("on", a, b, tol_m=0.01)], a, _manip_dist(s, a, b),
                 {"obj": _label(s, a), "target": _label(s, b)}, _others(s, {a, b}), a)


def _push(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    z = _pick(r, ["zone_0", "zone_1"])
    return Built([_P("in_region", "box_0", z), _P("on", "box_0", "table_0", tol_m=0.01)], "box_0",
                 _manip_dist(s, "box_0", z), {"obj": _label(s, "box_0"), "target": _label(s, z)},
                 _others(s, {"box_0"}), "box_0")


def _slide(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    obj = _pick(r, _objects(s, ("block", "mug", "can")))
    z = _pick(r, ["zone_0", "zone_1"])
    return Built([_P("in_region", obj, z), _P("on", obj, "table_0", tol_m=0.01), _P("upright", obj, tol_rad=0.2)],
                 obj, _manip_dist(s, obj, z), {"obj": _label(s, obj), "target": _label(s, z)},
                 _others(s, {obj}), obj)


def _insert(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    return Built([_P("inside", "peg_0", "hole_0"), _P("upright", "peg_0", tol_rad=0.15)], "peg_0",
                 _manip_dist(s, "peg_0", "hole_0"), {"obj": _label(s, "peg_0"), "target": "hole"},
                 _others(s, {"peg_0"}), "peg_0")


def _level_carry(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    return Built([_P("on", "mug_0", "plate_0", tol_m=0.02), _P("upright", "mug_0", tol_rad=0.1)], "mug_0",
                 _manip_dist(s, "mug_0", "plate_0"), {"obj": _label(s, "mug_0"), "target": "plate"},
                 _others(s, {"mug_0"}), "mug_0", [_G("tilt", "mug_0", good=0.05, bad=0.4)])


def _precise_place(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    obj = _pick(r, _objects(s, ("block", "box")))
    plate = s.get("plate_0")
    z = float(plate.pose[2] + plate.extent[2] + s.get(obj).extent[2])
    tgt = [float(plate.pose[0]), float(plate.pose[1]), z]
    return Built([_P("on", obj, "plate_0", tol_m=0.005), _P("at_pose", obj, pose=tgt, tol_m=0.005)], obj,
                 _manip_dist(s, obj, "plate_0"), {"obj": _label(s, obj), "target": "plate"},
                 _others(s, {obj}), obj)


def _pour(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    tgt = _pos(s, "bowl_0") + [0, 0, 0.15]
    return Built([_P("held", "mug_0"), _P("at_pose", "mug_0", pose=[float(v) for v in tgt], tol_m=0.05)], "mug_0",
                 _manip_dist(s, "mug_0", "bowl_0"), {"obj": _label(s, "mug_0"), "target": _label(s, "bowl_0")},
                 _others(s, {"mug_0"}), "mug_0", [_G("tilt", "mug_0", good=1.6, bad=0.4)])


def _wipe(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    return Built([_P("in_region", "sponge_0", "spill_0"), _P("on", "sponge_0", "counter_0", tol_m=0.01)],
                 "sponge_0", _manip_dist(s, "sponge_0", "spill_0"), {"obj": "sponge", "target": "spill"},
                 _others(s, {"sponge_0"}), None, [_G("force", "sponge_0", "counter_0", good=10.0, bad=0.0)])


def _open_door(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    return Built([_P("open_frac", "door_0", min=0.7)], "door_0_handle", _xy_dist(s, "door_0_handle") + 0.3,
                 {"target": _label(s, "door_0")}, _others(s, set()), "door_0_handle")


def _open_drawer(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    return Built([_P("open_frac", "drawer_0", min=0.8)], "drawer_0_handle",
                 _xy_dist(s, "drawer_0_handle") + DRAWER_TRAVEL_M, {"target": _label(s, "drawer_0")},
                 _others(s, set()), "drawer_0_handle")


def _close_drawer(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    tgt = _pos(s, "drawer_1_handle") + [DRAWER_TRAVEL_M, 0, 0]
    return Built([_P("at_pose", "drawer_1_handle", pose=[float(v) for v in tgt], tol_m=0.01)], "drawer_1_handle",
                 _xy_dist(s, "drawer_1_handle") + DRAWER_TRAVEL_M, {"target": _label(s, "drawer_1")},
                 _others(s, set()), "drawer_1_handle")


def _stow(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    return Built([_P("inside", "block_0", "drawer_1")], "block_0", _manip_dist(s, "block_0", "drawer_1"),
                 {"obj": _label(s, "block_0"), "target": _label(s, "drawer_1")}, _others(s, {"block_0"}), "block_0")


def _shelf_place(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    obj = _pick(r, _objects(s, GRASPABLE))
    lv = _pick(r, ["shelf_0_l1", "shelf_0_l2"])
    return Built([_P("on", obj, lv, tol_m=0.02)], obj, _manip_dist(s, obj, lv) + 0.3,
                 {"obj": _label(s, obj), "target": _label(s, lv)}, _others(s, {obj}) + ["shelf_0"], obj)


def _rearrange(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    zs = ["zone_0", "zone_1"] if r.random() < 0.5 else ["zone_1", "zone_0"]
    objs = ["block_0", "mug_0"]
    d = sum(_manip_dist(s, o, z) for o, z in zip(objs, zs))
    return Built([_P("in_region", o, z) for o, z in zip(objs, zs)], "block_0", d,
                 {"obj": _label(s, objs[0]), "target": _label(s, zs[0]), "obj2": _label(s, objs[1]),
                  "target2": _label(s, zs[1])}, _others(s, set(objs)), "block_0")


def _handover(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    obj = _pick(r, _objects(s, GRASPABLE))
    return Built([_P("held", obj), _P("in_region", obj, "handoff_0")], obj, _manip_dist(s, obj, "handoff_0"),
                 {"obj": _label(s, obj)}, _others(s, {obj}) + ["person_0"], obj)


# ---- ground builders ----

def _obstacles(s: Scene) -> list[str]:
    return [e.id for e in s.entities if e.id.split("_")[0] in ("obstacle", "rock", "car", "cone")
            or e.id.endswith("_wall") or e.id in ("road_0_curb", "corridor_0", "tower_0", "person_0")
            or e.id.startswith("gate_")]


def _walk_to(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    return Built([_P("in_region", "base", "goal_0")], "base", _xy_dist(s, "goal_0"),
                 {"target": _label(s, "goal_0")}, _obstacles(s))


def _climb(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    return Built([_P("in_region", "base", "top_0")], "base", _xy_dist(s, "top_0"),
                 {"target": _label(s, "platform_0")}, _obstacles(s))


def _carry(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    return Built([_P("in_region", "base", "goal_0"), _P("in_region", "payload_0", "goal_0"),
                  _P("no_contact", "payload_0", "floor")], "payload_0", _xy_dist(s, "goal_0"),
                 {"obj": _label(s, "payload_0"), "target": _label(s, "goal_0")}, _obstacles(s))


def _go_to(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    return Built([_P("in_region", "base", "goal_0")], "base", _xy_dist(s, "goal_0"),
                 {"target": _label(s, "goal_0")}, _obstacles(s))


def _dock(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    z = layout(robot, "base_height_m")
    return Built([_P("in_region", "base", "dock_0_slot"),
                  _P("at_pose", "base", pose=_yaw_pose(s, "dock_0_slot", z), tol_m=0.5, tol_rad=0.15)], "base",
                 _xy_dist(s, "dock_0_slot") + 1.0, {"target": _label(s, "dock_0")}, _obstacles(s))


def _path_len(s: Scene, id: str, closed: bool) -> float:
    p = s.get(id).points[:, :2]
    p = np.r_[p, p[:1]] if closed else p
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _follow_course(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    return Built([_P("in_region", "base", "finish_0"), _P("no_contact", "robot", "course_0_wall")], "base",
                 _path_len(s, "course_0", False), {"target": _label(s, "finish_0")}, _obstacles(s))


def _drive_lap(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    # finish box sits behind a barrier at the start, so only a full lap reaches it
    return Built([_P("in_region", "base", "finish_0"), _P("no_contact", "robot", "barrier_0"),
                  _P("no_contact", "robot", "track_0_wall")], "base", _path_len(s, "track_0", True),
                 {"target": _label(s, "finish_0")}, _obstacles(s) + ["barrier_0"])


def _lane_follow(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    cars = _ids(s, "car_")
    return Built([_P("in_region", "base", "finish_0"), _P("no_contact", "robot", "road_0_curb"),
                  _P("no_contact", "robot", "cone_0")] + [_P("no_contact", "robot", c) for c in cars], "base",
                 _path_len(s, "lane_0", False), {"target": _label(s, "finish_0")}, _obstacles(s))


def _park(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    cars = _ids(s, "car_")
    z = layout(robot, "base_height_m")
    return Built([_P("in_region", "base", "slot_0"),
                  _P("at_pose", "base", pose=_yaw_pose(s, "slot_0", z), tol_m=0.5, tol_rad=0.15)]
                 + [_P("no_contact", "robot", c) for c in cars], "base", _xy_dist(s, "slot_0") + 3.0,
                 {"target": _label(s, "slot_0")}, _obstacles(s))


# ---- flight builders ----

def _fly_to(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    return Built([_P("in_region", "base", "goal_0"), _P("no_contact", "robot", "tower_0")], "base",
                 float(np.linalg.norm(_pos(s, "goal_0"))), {"target": _label(s, "goal_0")}, _obstacles(s))


def _hover(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    return Built([_P("in_region", "base", "hover_0"), _P("upright", "base", tol_rad=0.15)], "base",
                 float(np.linalg.norm(_pos(s, "hover_0"))), {"target": _label(s, "hover_0")}, _obstacles(s))


def _gate_course(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    # the corridor is closed at the far end and every gate fills its cross-section, so the finish is reached
    # only through each gate opening in order
    gates = _ids(s, "gate_")
    return Built([_P("in_region", "base", "finish_0"), _P("no_contact", "robot", "corridor_0")]
                 + [_P("no_contact", "robot", g) for g in gates], "base",
                 float(np.linalg.norm(_pos(s, "finish_0"))), {"target": _label(s, "finish_0")}, _obstacles(s))


def _inspect(s: Scene, r: np.random.Generator, robot: RobotSpec) -> Built:
    v = s.get("view_0")
    return Built([_P("in_region", "base", "view_0"),
                  _P("at_pose", "base", pose=[float(x) for x in v.pose], tol_m=0.5, tol_rad=0.35),
                  _P("no_contact", "robot", "tower_0")], "base", float(np.linalg.norm(v.pose[:3])),
                 {"target": _label(s, "tower_0")}, _obstacles(s))


BUILDERS: dict[str, Builder] = {
    "reach": _reach, "pick": _pick_up, "place": _place, "stack": _stack, "push": _push, "slide": _slide,
    "insert": _insert, "level_carry": _level_carry, "precise_place": _precise_place, "pour": _pour, "wipe": _wipe,
    "open_door": _open_door, "open_drawer": _open_drawer, "close_drawer": _close_drawer, "stow": _stow,
    "shelf_place": _shelf_place, "rearrange": _rearrange, "handover": _handover, "walk_to": _walk_to,
    "climb": _climb, "carry": _carry, "go_to": _go_to, "dock": _dock, "follow_course": _follow_course,
    "drive_lap": _drive_lap, "lane_follow": _lane_follow, "park": _park, "fly_to": _fly_to, "hover": _hover,
    "gate_course": _gate_course, "inspect": _inspect,
}


# ---- style terms and assembly ----

def style_terms(style: str, b: Built, robot: RobotSpec, overhead_s: float) -> list[dict]:
    """Graded and penalty entries for one style; scales come from the family's nominal motion."""
    f = CFG["families"][robot.family]
    t_nom = overhead_s + b.dist_m / f["v_nom_mps"]
    if style == "neutral":
        return [_G("time", good=round(1.5 * t_nom, 2), bad=round(5.0 * t_nom, 2))]
    if style == "fast":
        return [_G("time", good=round(0.8 * t_nom, 2), bad=round(2.5 * t_nom, 2))]
    if style == "gentle":
        a, j = f["a_nom_mps2"], f["j_nom_mps3"]
        out = [_G("accel", b.subject, good=round(0.3 * a, 3), bad=round(a, 3)),
               _G("jerk", b.subject, good=round(0.2 * j, 3), bad=round(j, 3))]
        if b.touch is not None:
            out.append(_G("force", b.touch, "robot", good=5.0, bad=40.0))
        return out
    if style == "careful":
        m = float(robot.floors["min_clearance_m"])
        return [_G("clearance", good=round(m + 0.1 * max(f["v_nom_mps"], 1.0), 3), bad=m)] + \
            [_G("contact_events", "robot", a, weight=0.1) for a in b.avoid]
    if style == "energy_saving":
        e = f["energy_J_per_m"] * max(b.dist_m, 0.1)
        return [_G("energy", good=round(e, 2), bad=round(4 * e, 2))]
    raise ValueError(f"unknown style {style!r}")


def assemble(goal: GoalSpec, b: Built, robot: RobotSpec, style: str, task_id: str) -> TaskTest:
    preds = list(b.predicates)
    if robot.family in LEGGED:
        tilt = robot.floors.get("max_tilt_rad")
        preds.append(_P("not_fallen", "robot", **({"max_tilt_rad": float(tilt)} if tilt is not None else {})))
    graded = list(b.graded) + style_terms(style, b, robot, goal.overhead_s)
    t_bad = max([float(g["bad"]) for g in graded if g["term"] == "time"] + [0.0])
    f = CFG["families"][robot.family]
    horizon = max(goal.horizon_s, 1.2 * (goal.overhead_s + 5.0 * b.dist_m / f["v_nom_mps"]), 1.2 * t_bad)
    for d in preds:
        assert d["pred"] in PREDICATES, d
    for d in graded:
        assert d["term"] in GRADED or d["term"] in PENALTIES, d
    return TaskTest(task_id, preds, graded, round(float(horizon), 1))


def _load() -> dict[str, GoalSpec]:
    out = {}
    for name, g in CFG["goals"].items():
        out[name] = GoalSpec(name, tuple(g["families"]), tuple(g["scene_kinds"]), BUILDERS[name],
                             tuple(g.get("styles", STYLES)), tuple(g["templates"]), float(g["overhead_s"]),
                             float(g["horizon_s"]))
    assert set(out) == set(BUILDERS), set(out) ^ set(BUILDERS)
    return out


GOALS: dict[str, GoalSpec] = _load()


def goals_for(family: str) -> list[GoalSpec]:
    return [g for g in GOALS.values() if family in g.families]
