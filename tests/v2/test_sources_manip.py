"""Manipulation training sources (RoboCasa365, Meta-World, robosuite) -> TaskCards.

Core env: the table-backed sources run anywhere; RoboCasa needs its source tree on disk. The live-env tests run
in the envs/robocasa venv:
  PYTHONPATH=. /home/shay/languagespec-assets/venvs/robocasa/bin/python -m pytest -q --noconftest \
    tests/v2/test_sources_manip.py   (--noconftest: tests/conftest.py imports torch, absent there)
"""
import ast
import importlib.util
import json
import os

import numpy as np
import pytest

from spec2mpc.core import TaskCard, TaskTest
from spec2mpc.rubric.v2 import PREDICATES, score
from spec2mpc.tasks.sources import (COVERAGE_DIR, card_errors, command_leaks, metaworld, region_box, robocasa,
                                    robosuite_tasks)

requires_robocasa = pytest.mark.skipif(robocasa.source_dir() is None, reason="RoboCasa source not found")
requires_metaworld = pytest.mark.skipif(not metaworld.installed(), reason="metaworld not installed (envs/robocasa)")
requires_robosuite = pytest.mark.skipif(not robosuite_tasks.installed(), reason="robosuite not installed")
_RC_ASSETS = (robocasa.source_dir() or robocasa.Path("/nonexistent")) / "models" / "assets" / "objects" / "objaverse"
requires_robocasa_assets = pytest.mark.skipif(
    importlib.util.find_spec("robocasa") is None or not _RC_ASSETS.exists(), reason="RoboCasa env or assets absent")

SOURCES = {"robocasa": robocasa, "metaworld": metaworld, "robosuite": robosuite_tasks}


def _check_cards(mod, cards: list[TaskCard], source: str) -> None:
    ents = {t.task: t.entities for t in mod.translations()[0]}
    assert len(cards) == 10
    assert len({c.public.task_id for c in cards}) == 10
    for c in cards:
        task = c.public.task_id.split("/")[1]
        assert c.source == source and c.split == "train"
        assert card_errors(c, ents[task]) == [], (task, card_errors(c, ents[task]))
        assert all(d["pred"] in PREDICATES for d in c.test.predicates)
        assert "predicates" not in json.dumps(c.public_view().__dict__, default=str)
        assert c.test.horizon_s > 0
        assert c.public.robot.floors and set(c.public.robot.floors) >= {"min_clearance_m", "max_speed_frac"}


@requires_robocasa
def test_robocasa_cards():
    _check_cards(robocasa, robocasa.cards(10, seed=0), "robocasa")


def test_metaworld_cards():
    _check_cards(metaworld, metaworld.cards(10, seed=0), "metaworld")


def test_robosuite_cards():
    _check_cards(robosuite_tasks, robosuite_tasks.cards(10, seed=0), "robosuite")


@pytest.mark.parametrize("name", ["metaworld", "robosuite", pytest.param("robocasa", marks=requires_robocasa)])
def test_translation_coverage_logged(name):
    mod = SOURCES[name]
    mod.cards(3, seed=1)
    cov = json.loads((COVERAGE_DIR / f"{name}_coverage.json").read_text())
    assert cov["source"] == name
    assert cov["translated"] > 0 and cov["skipped"] > 0
    assert cov["translated"] + cov["skipped"] == cov["total"]
    assert len(cov["translated_tasks"]) == cov["translated"] and len(cov["skipped_tasks"]) == cov["skipped"]


@pytest.mark.parametrize("name", ["metaworld", "robosuite", pytest.param("robocasa", marks=requires_robocasa)])
def test_cards_seeded(name):
    mod = SOURCES[name]
    a, b, c = mod.cards(5, seed=3), mod.cards(5, seed=3), mod.cards(5, seed=4)
    assert [x.public.task_id for x in a] == [x.public.task_id for x in b]
    assert [x.public.task_id for x in a] != [x.public.task_id for x in c]


@pytest.mark.parametrize("name", ["metaworld", "robosuite", pytest.param("robocasa", marks=requires_robocasa)])
def test_commands_have_no_thresholds(name):
    for t in SOURCES[name].translations()[0]:
        assert t.command and not command_leaks(t.command, t.predicates), t.task


def test_command_leak_check_catches_threshold():
    assert command_leaks("get within 0.05 m", [{"pred": "at_pose", "args": ["obj", "goal"], "tol_m": 0.05}])
    assert not command_leaks("push the puck", [{"pred": "at_pose", "args": ["obj", "goal"], "tol_m": 0.05}])


# ---- RoboCasa static translator on hand-written checks ----

def _cls(src: str) -> dict[str, ast.ClassDef]:
    return {n.name: n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ClassDef)}


def test_robocasa_translates_conjunction_and_drops_gripper_far():
    cls = _cls('''
class T(Kitchen):
    def _check_success(self):
        a = OU.obj_inside_of(self, "obj", self.sink)
        b = OU.gripper_obj_far(self)
        return a and b and OU.check_obj_upright(self, "obj", th=10)
''')
    ps, dropped = robocasa.translate_check("T", cls)
    assert [d["pred"] for d in ps] == ["inside", "upright"]
    assert ps[0]["args"] == ["obj", "sink"] and dropped == ["gripper_obj_far"]
    assert abs(ps[1]["tol_rad"] - np.radians(10)) < 1e-4


@pytest.mark.parametrize("body", [
    "return self.cab.is_closed(self)",  # closed needs an upper bound on open_frac
    "return OU.obj_inside_of(self, 'a', self.sink) or OU.obj_inside_of(self, 'b', self.sink)",
    "return not OU.check_obj_grasped(self, 'obj')",
    "return OU.gripper_obj_far(self)",  # nothing left to test
    "for o in self.objs:\n            ok = OU.check_obj_grasped(self, o)\n        return ok",
])
def test_robocasa_skips_untranslatable(body):
    cls = _cls(f"class T(Kitchen):\n    def _check_success(self):\n        {body}\n")
    with pytest.raises(robocasa.Skip):
        robocasa.translate_check("T", cls)


def test_robocasa_behavior_branch_follows_subclass():
    cls = _cls('''
class Door(Kitchen):
    def _check_success(self):
        if self.behavior == "open":
            return self.fxtr.is_open(env=self)
        elif self.behavior == "close":
            return self.fxtr.is_closed(env=self)
class OpenIt(Door):
    def __init__(self, *a, **k):
        super().__init__(behavior="open", *a, **k)
class CloseIt(Door):
    def __init__(self, *a, **k):
        super().__init__(behavior="close", *a, **k)
''')
    assert robocasa.translate_check("OpenIt", cls)[0] == [{"pred": "open_frac", "args": ["fxtr"], "min": 0.9}]
    with pytest.raises(robocasa.Skip):
        robocasa.translate_check("CloseIt", cls)
    with pytest.raises(robocasa.Skip):  # behavior unknown: branch unreadable
        robocasa.translate_check("Door", cls)


def test_robocasa_parametrized_base_not_a_task():
    # OpenDoor passes *args to a base that needs fixture_id, so robosuite.make cannot build it alone
    cls = _cls('''
class Kitchen:
    def __init__(self, robots, layout_ids=None): pass
class Door(Kitchen):
    def __init__(self, fixture_id, behavior="open", *a, **k): pass
class OpenIt(Door):
    def __init__(self, *a, **k): pass
class OpenFridge(OpenIt):
    def __init__(self, fixture_id="fridge", *a, **k): pass
''')
    assert robocasa._missing_ctor_args("OpenIt", cls) == {"fixture_id"}
    assert robocasa._missing_ctor_args("OpenFridge", cls) == set()
    assert robocasa._missing_ctor_args("Kitchen", cls) == set()


# ---- translated tests score correctly under the rubric ----

def _trace(poses: dict[str, np.ndarray], extents: dict[str, np.ndarray]) -> dict:
    T = len(next(iter(poses.values())))
    tr = {"t_s": np.arange(T) * 0.05, "contacts": [[] for _ in range(T)], "robot_base_height": np.ones(T)}
    tr |= {f"entity_pose/{k}": v for k, v in poses.items()}
    tr |= {f"entity_extent/{k}": v for k, v in extents.items()}
    return tr


def _pose(xyz) -> np.ndarray:
    return np.array([[*xyz, 1.0, 0.0, 0.0, 0.0]])


@pytest.mark.parametrize("end, ok", [((0.0, 0.03, 0.0), True), ((0.0, 0.0, 0.08), False)])
def test_metaworld_at_pose_scores(end, ok):
    t = metaworld.translate("push-v3")  # |obj - goal| <= 0.05
    tr = _trace({"obj": np.vstack([_pose((0.3, 0.0, 0.0)), _pose(end)]), "goal": np.vstack([_pose((0, 0, 0))] * 2)},
                {})
    test = TaskTest("x", t.predicates, [], 5.0)
    assert score(test, tr)["success"] is ok


@pytest.mark.parametrize("y, ok", [(0.019, True), (0.03, False)])
def test_metaworld_slab_region_scores(y, ok):
    t = metaworld.translate("button-press-v3")  # |obj_y - goal_y| <= 0.02, x and z free
    (d,) = t.predicates
    goal = np.array([0.1, 0.8, 0.1])
    rp, re_ = region_box(d, goal)
    tr = _trace({"obj": _pose(goal + [0.5, y, -0.3]), "goal": _pose(goal), "goal_slab": rp[None]},
                {"goal_slab": re_})
    test = TaskTest("x", t.predicates, [], 5.0)
    assert score(test, tr)["success"] is ok


@pytest.mark.parametrize("dz, ok", [(0.05, True), (0.03, False)])
def test_robosuite_lift_region_scores(dz, ok):
    (t,) = [x for x in robosuite_tasks.translations()[0] if x.task == "Lift"]
    (d,) = t.predicates
    top = np.array([0.0, 0.0, 0.8])
    rp, re_ = region_box(d, top)
    tr = _trace({"cube": _pose(top + [0.1, -0.2, dz]), "cube_lifted": rp[None]}, {"cube_lifted": re_})
    test = TaskTest("x", t.predicates, [], 5.0)
    assert score(test, tr)["success"] is ok


def test_robosuite_pickplace_bins_tile_bin2():
    (t,) = [x for x in robosuite_tasks.translations()[0] if x.task == "PickPlace"]
    boxes = [region_box(d, np.zeros(3)) for d in t.predicates]
    centers = {tuple(np.round(p[:2], 4)) for p, _ in boxes}
    assert len(centers) == 4  # four distinct quadrants
    for p, e in boxes:
        assert np.allclose(e[:2], [0.39 / 4, 0.49 / 4]) and np.isclose(e[2], 0.05)


# ---- live source envs (envs/robocasa venv only) ----

@requires_metaworld
def test_metaworld_table_matches_registry():
    assert set(metaworld.TASKS) == set(metaworld.env_names())


@requires_metaworld
def test_metaworld_env_has_goal_and_object():
    import metaworld as mw
    env = mw.ALL_V3_ENVIRONMENTS["push-v3"]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._last_rand_vec = None
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] >= 7 and np.asarray(env._target_pos).shape == (3,)
    assert np.isclose(env.frame_skip * env.model.opt.timestep, metaworld.STEP_S)
    assert env.max_path_length == metaworld.MAX_PATH_LENGTH


@requires_robosuite
def test_robosuite_table_matches_registry():
    assert set(robosuite_tasks.TASKS) == set(robosuite_tasks.env_names())


@requires_robosuite
@pytest.mark.parametrize("task, robots, ids", [
    ("Lift", "Panda", ["cube"]),
    ("Stack", "Panda", ["cubeA", "cubeB"]),
    ("NutAssembly", "Panda", ["SquareNut", "RoundNut"]),
    ("PickPlace", "Panda", ["Milk", "Bread", "Cereal", "Can"]),
])
def test_robosuite_entity_names_exist(task, robots, ids):
    import robosuite
    env = robosuite.make(task, robots=robots, has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, ignore_done=True)
    env.reset()
    names = {o.name for o in getattr(env, "objects", [])} | {getattr(env, a).name for a in ("cube", "cubeA", "cubeB")
                                                              if hasattr(env, a)}
    names |= {n.name for n in getattr(env, "nuts", [])}
    assert set(ids) <= names, names
    env.close()


def _robocasa_ids_resolve(task: str) -> list[str]:
    # every entity of the translated task is an env object name or a fixture attribute after reset
    importlib.import_module("robocasa")  # registers the kitchen envs (the name robocasa here is the adapter)
    import robosuite
    (t,) = [x for x in robocasa.translations()[0] if x.task == task]
    env = robosuite.make(task, robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, ignore_done=True, seed=0)
    env.reset()
    missing = [e for e in t.entities if e not in env.objects and not hasattr(env, e)]
    env.close()
    return missing


@requires_robocasa_assets
@pytest.mark.parametrize("task", ["PickPlaceCounterToSink", "OpenDrawer", "OpenMicrowave"])
def test_robocasa_entity_ids_resolve(task):
    os.environ.setdefault("MUJOCO_GL", "egl")
    assert _robocasa_ids_resolve(task) == []


@pytest.mark.slow
@requires_robocasa_assets
def test_robocasa_entity_ids_resolve_all():
    os.environ.setdefault("MUJOCO_GL", "egl")
    bad = {}
    for t in robocasa.translations()[0]:
        try:
            m = _robocasa_ids_resolve(t.task)
        except Exception as e:  # env construction failure is reported, not hidden
            m = [f"error:{type(e).__name__}:{e}"]
        if m:
            bad[t.task] = m
    assert bad == {}
