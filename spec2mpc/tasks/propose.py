"""LLM-proposed tasks (GenSim2 / RoboGen pattern), kept only when the oracle solves them and the zero program fails.

For each proposal a goal type and a scene are drawn; gpt-6.1-sol sees the robot and the scene entities and writes a
new task in that scene from the fixed vocabularies: a style, a command, and a test (predicates plus graded terms).
The proposal is validated (names, pointers, params, no threshold in the command), then `zero(card)` must fail and
`solve(card)` (Task 8 oracle) must succeed. Both are injected: (TaskCard) -> {"success": bool, "score": float, ...}.
"""
from __future__ import annotations

import json
import zlib
from typing import Callable

import numpy as np

from spec2mpc.core import RobotSpec, Scene, TaskCard, TaskPublic, TaskTest
from spec2mpc.rubric.v2 import GRADED, PENALTIES, PREDICATES
from spec2mpc.tasks.commands import LLM, OpenAIJSON, has_threshold
from spec2mpc.tasks.goals import GOALS, LEGGED, STYLES, goals_for
from spec2mpc.tasks.scenes import make_scene

Solve = Callable[[TaskCard], dict]
RobotFn = Callable[[str, np.random.Generator], RobotSpec]

# required params per predicate (others optional); "count" is left out because it names trace counters
REQUIRED = {"on": (), "inside": (), "open_frac": ("min",), "upright": ("tol_rad",), "at_pose": ("tol_m",),
            "in_region": (), "not_fallen": (), "time_below": ("t_s",), "no_contact": (), "held": ()}
N_ARGS = {"on": 2, "inside": 2, "open_frac": 1, "upright": 1, "at_pose": 2, "in_region": 2, "not_fallen": 1,
          "time_below": 0, "no_contact": 2, "held": 1}
GRADED_ARGS = {"time": 0, "speed": 1, "accel": 1, "jerk": 1, "tilt": 1, "force": 2, "clearance": 0, "energy": 0,
               "contact_events": 2, "fell": 1}

INSTRUCTIONS = """You design training tasks for a robot skill executor. You get a robot and the entities of a \
simulated scene (id, label, kind, position). Propose ONE new task in this scene that this robot can do: a short \
command as a person would say it, a manner (style), and an automatic test built only from the vocabulary below. \
Every test argument is an entity id from the scene, a robot frame of this robot, "robot" (any robot link), or \
"floor". The command must not state any number, tolerance, time, or limit; the test holds those.

Predicates (all must hold at the end of the episode; not_fallen and no_contact must hold throughout):
  on(a, b) tol_m: a rests on top of b | inside(a, b) tol_m: a's centre inside box b | open_frac(a) min: drawer or door
  a opened at least min (0..1) | upright(a) tol_rad: a's z axis within tol of vertical | at_pose(a, b) tol_m [tol_rad]:
  a within tol of entity b's pose | in_region(a, region) | not_fallen(robot) | time_below() t_s: done before t_s |
  no_contact(a, b) | held(a): a touched only by the robot
Graded terms (each scored 0..1 by a ramp from good to bad): time() | speed(a) | accel(a) | jerk(a) | tilt(a) |
  force(a, b) | clearance() | energy()   Penalties (subtracted, scaled by weight): contact_events(a, b) | fell(robot)
The task must be different from simply repeating the base goal type given, physically possible, and not already \
true at the start."""


def _schema(goal_names: list[str]) -> dict:
    param = {"type": "object", "additionalProperties": False, "required": ["name", "value"],
             "properties": {"name": {"type": "string"}, "value": {"type": "number"}}}
    item = {"type": "object", "additionalProperties": False, "required": ["name", "args", "params"],
            "properties": {"name": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"}},
                           "params": {"type": "array", "items": param}}}
    return {"type": "object", "additionalProperties": False,
            "required": ["goal", "style", "command", "predicates", "graded"],
            "properties": {"goal": {"type": "string", "enum": goal_names},
                           "style": {"type": "string", "enum": list(STYLES)},
                           "command": {"type": "string"},
                           "predicates": {"type": "array", "items": item},
                           "graded": {"type": "array", "items": item}}}


def _user_text(robot: RobotSpec, scene: Scene, base_goal: str, goal_names: list[str]) -> str:
    lines = [f"robot: {robot.name} (family {robot.family}), frames: {', '.join(robot.frames)}",
             f"base goal type: {base_goal}; goal types you may label the task with: {', '.join(goal_names)}",
             "scene entities:"]
    for e in scene.entities:
        if e.kind in ("object", "fixture", "region"):
            lines.append(f"  {e.id}: {e.label} ({e.kind}) at {np.round(e.pose[:3], 2).tolist()}")
    return "\n".join(lines)


def _flat(items: list[dict], key: str) -> list[dict]:
    return [{key: it["name"], "args": list(it["args"]), **{p["name"]: float(p["value"]) for p in it["params"]}}
            for it in items]


def validate_proposal(out: dict, robot: RobotSpec, scene: Scene) -> list[str]:
    """Error strings; empty when the proposal only uses the vocabularies and existing pointers."""
    ok_ids = {e.id for e in scene.entities} | set(robot.frames) | {"robot", "floor"}
    errs = []
    if not out["predicates"]:
        errs.append("no predicates")
    for d in out["predicates"]:
        n = d["name"]
        if n not in PREDICATES:
            errs.append(f"unknown predicate {n}")
            continue
        if len(d["args"]) != N_ARGS[n]:
            errs.append(f"{n}: wrong arg count")
        if missing := [k for k in REQUIRED[n] if k not in {p["name"] for p in d["params"]}]:
            errs.append(f"{n}: missing {missing}")
        errs += [f"{n}: unknown pointer {a}" for a in d["args"] if a not in ok_ids]
    for d in out["graded"]:
        n = d["name"]
        if n not in GRADED_ARGS or (n not in GRADED and n not in PENALTIES):
            errs.append(f"unknown graded term {n}")
            continue
        if len(d["args"]) != GRADED_ARGS[n]:
            errs.append(f"{n}: wrong arg count")
        names = {p["name"] for p in d["params"]}
        if n in GRADED and not {"good", "bad"} <= names:
            errs.append(f"{n}: needs good and bad")
        errs += [f"{n}: unknown pointer {a}" for a in d["args"] if a not in ok_ids]
    return errs


def _default_robots() -> RobotFn:
    from spec2mpc.robots.v2 import list_robots, load_robot

    def f(family: str, rng: np.random.Generator) -> RobotSpec:
        names = list_robots(family)
        return load_robot(family, names[int(rng.integers(len(names)))], int(rng.integers(2**31)))

    return f


def propose_tasks(n: int, families: list[str], *, solve: Solve, zero: Solve, llm: LLM | None = None,
                  robots: RobotFn | None = None, seed: int = 0, max_tries: int | None = None,
                  log: list[dict] | None = None) -> list[TaskCard]:
    """Up to n kept proposals over the given families; `log` (if given) receives one record per attempt."""
    llm = llm or OpenAIJSON()
    robots = robots or _default_robots()
    rng = np.random.default_rng([seed, zlib.crc32(b"propose")])
    kept: list[TaskCard] = []
    for k in range(max_tries or 5 * n):
        if len(kept) >= n:
            break
        family = families[int(rng.integers(len(families)))]
        robot = robots(family, rng)
        goals = goals_for(family)
        base = goals[int(rng.integers(len(goals)))]
        scene_seed = int(rng.integers(2**31))
        kind = base.scene_kind(scene_seed)
        _, scene = make_scene(kind, robot, scene_seed)
        # the scene is rebuilt from (goal, seed), so the label goal must map this seed to the same scene kind
        names = [g.name for g in goals if g.scene_kind(scene_seed) == kind]
        rec: dict = {"try": k, "family": family, "robot": robot.name, "base_goal": base.name, "scene_seed": scene_seed}
        try:
            out = json.loads(llm(INSTRUCTIONS, _user_text(robot, scene, base.name, names), _schema(names)))
        except (json.JSONDecodeError, RuntimeError) as e:
            rec["reject"] = f"llm: {e}"
            _log(log, rec)
            continue
        errs = validate_proposal(out, robot, scene)
        if out.get("goal") not in names:
            errs.append(f"goal {out.get('goal')} not valid for this scene")
            out["goal"] = base.name
        g = GOALS[out["goal"]]
        tid = f"prop-{out.get('goal')}-{robot.name}-s{scene_seed}-{k}"
        preds, graded = _flat(out["predicates"], "pred"), _flat(out["graded"], "term")
        if family in LEGGED and not any(d["pred"] == "not_fallen" for d in preds):
            preds.append({"pred": "not_fallen", "args": ["robot"]})
        test = TaskTest(tid, preds, graded, 1.5 * g.horizon_s)
        if not errs and has_threshold(out["command"], test):
            errs.append("command states a test threshold")
        if errs:
            rec["reject"] = "; ".join(errs)
            _log(log, rec)
            continue
        card = TaskCard(TaskPublic(tid, family, robot, scene_seed, " ".join(out["command"].split()), g.name),
                        test, "train", "proposed")
        z = zero(card)
        rec["zero"] = z.get("success")
        if z.get("success"):
            rec["reject"] = "zero program passes"
            _log(log, rec)
            continue
        s = solve(card)
        rec["oracle"] = s.get("success")
        if not s.get("success"):
            rec["reject"] = "oracle fails"
            _log(log, rec)
            continue
        rec["kept"] = tid
        _log(log, rec)
        kept.append(card)
    return kept


def _log(log: list[dict] | None, rec: dict) -> None:
    if log is not None:
        log.append(rec)
