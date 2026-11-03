import json
import os
import re
from pathlib import Path

import mujoco
import numpy as np
import pytest

from spec2mpc.core import load_json, to_jsonable
from spec2mpc.eval.v2.suite import load_split_cards
from spec2mpc.robots.v2 import FAMILIES, build_model, list_robots, load_robot, robot_frames_as_entities
from spec2mpc.rubric.v2 import GRADED, INVARIANTS, PENALTIES, PREDICATES, evaluate, normalize
from spec2mpc.tasks import (
    GOALS,
    SCENE_KINDS,
    SPLITS,
    STYLES,
    assign_splits,
    build_scene,
    draft_commands,
    goals_for,
    has_threshold,
    make_card,
    make_scene,
    make_splits,
    propose_tasks,
    save_cards,
    thresholds,
    validate_proposal,
    with_command,
)

ROOT = Path(__file__).parent.parent.parent
BENCH = ("robolab", "libero", "humanoidbench")
_ROBOTS: dict = {}


def robot(family: str, i: int = 0, seed: int | None = 0):
    k = (family, i, seed)
    if k not in _ROBOTS:
        _ROBOTS[k] = load_robot(family, list_robots(family)[i], seed)
    return _ROBOTS[k]


def all_cards(seeds=(0, 1)):
    return [make_card(g, robot(f, i), s, st, "train") for g in GOALS.values() for f in g.families
            for i in range(min(2, len(list_robots(f)))) for s in seeds for st in g.styles]


@pytest.fixture(scope="module")
def cards():
    return all_cards()


def _pointer_ok(card, scene) -> list[str]:
    ok = {e.id for e in scene.entities} | set(card.public.robot.frames) | {"robot", "floor"}
    bad = []
    for d in card.test.predicates + card.test.graded:
        name, args, params = normalize(d)
        if name == "count":
            continue
        bad += [a for a in args if a not in ok]
        if name == "at_pose" and len(args) == 1:
            assert len(params["pose"]) in (3, 7)
    return bad


def test_every_goal_makes_valid_card():
    assert 28 <= len(GOALS) <= 34
    for g in GOALS.values():
        for f in g.families:
            c = make_card(g, robot(f), 0, "neutral", "train")
            xml, scene = build_scene(c.public)
            mujoco.MjModel.from_xml_string(xml)
            assert c.test.task_id == c.public.task_id and c.public.goal == g.name
            assert c.test.predicates and c.test.horizon_s > 0
            assert not _pointer_ok(c, scene), (g.name, f, _pointer_ok(c, scene))
            for d in c.test.predicates:
                assert normalize(d)[0] in PREDICATES
            for d in c.test.graded:
                assert normalize(d)[0] in GRADED or normalize(d)[0] in PENALTIES
            pub = json.dumps(to_jsonable(c.public_view()))
            assert "predicates" not in pub and "graded" not in pub


def test_goal_coverage_per_family():
    for f in FAMILIES:
        assert len(goals_for(f)) >= 2, f
    need = {"insert": "arm", "level_carry": "arm", "precise_place": "arm", "drive_lap": "kart",
            "follow_course": "kart", "go_to": "rover2d", "gate_course": "quadrotor", "inspect": "quadrotor",
            "walk_to": "humanoid", "carry": "quadruped"}
    for g, f in need.items():
        assert f in GOALS[g].families
    for g in GOALS.values():
        assert set(g.styles) == set(STYLES), g.name


def test_scene_kinds_compile_and_are_seeded():
    fam = {"tabletop": "arm", "shelf": "mobile_manipulator", "kitchen": "arm", "drawer_cabinet": "arm",
           "floor_obstacles": "quadruped", "terrain": "humanoid", "course": "rover2d", "track": "kart",
           "road": "mobile_base", "airspace": "quadrotor"}
    assert set(fam) == set(SCENE_KINDS)
    for k, f in fam.items():
        x0, s0 = make_scene(k, robot(f), 3)
        x1, s1 = make_scene(k, robot(f), 3)
        x2, _ = make_scene(k, robot(f), 4)
        assert x0 == x1 and s0 == s1 and x0 != x2
        m = mujoco.MjModel.from_xml_string(x0)
        bodies = {m.body(i).name for i in range(m.nbody)}
        ids = [e.id for e in s0.entities]
        assert len(ids) == len(set(ids))
        assert {e.id for e in s0.entities if e.kind != "path"} <= bodies


def _static_trace(card) -> dict:
    # every entity and robot frame at its initial pose for two steps, nothing touching
    _, scene = build_scene(card.public)
    r = card.public.robot
    ents = list(scene.entities) + robot_frames_as_entities(r, build_model(r).qpos0)
    tr: dict = {"t_s": np.array([0.0, 1.0]), "contacts": [[], []], "robot_base_height": np.ones(2),
                "robot_base_pose": np.tile([0, 0, 1, 1, 0, 0, 0.0], (2, 1))}
    for e in ents:
        tr[f"entity_pose/{e.id}"] = np.tile(e.pose, (2, 1))
        if e.extent is not None:
            tr[f"entity_extent/{e.id}"] = e.extent
        if e.kind == "region" and e.points is not None:
            tr[f"entity_points/{e.id}"] = e.points
        if e.id.startswith(("drawer_", "door_")) and not e.id.endswith("handle"):
            tr[f"open_frac/{e.id}"] = np.full(2, 1.0 if e.id == "drawer_1" else 0.0)
    return tr


def test_cards_not_solved_at_start(cards):
    # the state predicates of a card must not already hold when nothing moves
    for c in {(c.public.goal, c.public.family): c for c in cards}.values():
        state = [d for d in c.test.predicates if normalize(d)[0] not in INVARIANTS | {"time_below"}]
        assert state, c.public.task_id
        assert not all(evaluate(state, _static_trace(c))), c.public.task_id


def test_commands_have_no_thresholds(cards):
    for c in cards:
        assert not has_threshold(c.public.command, c.test), c.public.command
        assert not re.search(r"\d", c.public.command), c.public.command
    pp = make_card("precise_place", robot("arm"), 0, "neutral", "train")
    assert 0.005 in thresholds(pp.test)
    assert has_threshold("put it within 5 mm of the middle", pp.test)
    assert has_threshold("within half a centimetre... 0.5 cm", pp.test)
    assert not has_threshold("place the block exactly in the middle of the plate", pp.test)
    od = make_card("open_door", robot("arm"), 0, "neutral", "train")
    assert has_threshold("open the door at least 70%", od.test)


def test_splits_disjoint(cards):
    s = make_splits(cards)
    assert set(s) == set(SPLITS)
    ids = [t for v in s.values() for t in v]
    assert len(ids) == len(set(ids)) == len({c.public.task_id for c in cards})
    assert all(s[k] for k in SPLITS), {k: len(v) for k, v in s.items()}
    # paraphrase variants follow their base card
    c = cards[0]
    s2 = make_splits([c, with_command(c, "grab it", 1)])
    assert sum(len(v) for v in s2.values()) == 2 and any(len(v) == 2 for v in s2.values())


def test_heldout_type_absent_from_train(cards):
    by = {c.public.task_id: c for c in cards}
    s = make_splits(cards)
    assert {by[t].public.family for t in s["heldout_type"]} == {"quadrotor"}
    for k in ("train", "val", "heldout_goal", "heldout_robot"):
        assert "quadrotor" not in {by[t].public.family for t in s[k]}
    no_quad = [c for c in cards if c.public.family != "quadrotor"]
    s = make_splits(no_quad)
    assert {by[t].public.family for t in s["heldout_type"]} == {"rover2d"}
    assert "rover2d" not in {by[t].public.family for t in s["train"]}
    assert "humanoid" in {by[t].public.family for t in make_splits(cards)["train"]}


def test_save_cards_layout(tmp_path, cards):
    sub = assign_splits(cards[::40])
    paths = save_cards(sub, tmp_path)
    for p, c in zip(paths, sub):
        assert p == tmp_path / c.split / f"{c.public.task_id}.json"
        assert load_json(p) == c
    loaded = {c.public.task_id for k in SPLITS for c in load_split_cards(k, tmp_path)}
    assert loaded == {c.public.task_id for c in sub}


def test_no_benchmark_assets():
    for k in SCENE_KINDS:
        for f in FAMILIES:
            try:
                xml, scene = make_scene(k, robot(f), 0)
            except Exception:
                continue  # a kind may not fit every family; compile coverage is in the tests above
            low = xml.lower()
            assert "file=" not in low and not any(w in low for w in BENCH)
            assert not any(w in e.label.lower() or w in e.id.lower() for e in scene.entities for w in BENCH)
    for p in list((ROOT / "spec2mpc" / "tasks").glob("*.py")) + [ROOT / "configs" / "v2" / "goals.yaml"]:
        low = p.read_text().lower()
        assert not any(re.search(rf"[\w/]*{w}[\w/]*\.(xml|usd|obj|stl|bddl)", low) for w in BENCH), p


def test_source_cards_in_suite(tmp_path, cards):
    import importlib

    from spec2mpc.tasks import SOURCE_MODULES, source_available, source_cards
    from spec2mpc.tasks.cards import CFG
    from spec2mpc.tasks.sources import card_errors, command_leaks

    names = [s for s in SOURCE_MODULES if source_available(s)]
    if not names:
        pytest.skip("no training source available")
    assert not any(w in s for s in SOURCE_MODULES for w in BENCH)
    src = source_cards(n=6)
    assert {c.source for c in src} == set(names)
    for c in src:
        mod = importlib.import_module(f"spec2mpc.tasks.sources.{SOURCE_MODULES[c.source]}")
        if hasattr(mod, "scene"):  # nav sources build their own scene
            ents = [e.id for e in mod.scene(c)[1].entities]
        else:  # manip sources run in the source simulator; entities come from the translation
            ents = {t.task: t.entities for t in mod.translations()[0]}[c.public.task_id.split("/")[1]]
        assert card_errors(c, ents) == [], (c.public.task_id, card_errors(c, ents))
        assert not command_leaks(c.public.command, c.test.predicates), c.public.command
        assert not any(w in c.public.task_id.lower() or w in c.public.robot.mjcf_path.lower() for w in BENCH)
    # source cards obey the generator's held-out rules
    suite = assign_splits(list(cards[::40]) + src, "quadrotor")
    hrob = {n for ns in CFG["splits"]["heldout_robots"].values() for n in ns}
    for c in suite:
        p = c.public
        if p.family == "quadrotor":
            assert c.split == "heldout_type", p.task_id
        elif p.robot.name in hrob:
            assert c.split == "heldout_robot", p.task_id
        elif p.goal in CFG["splits"]["heldout_goals"]:
            assert c.split == "heldout_goal", p.task_id
        else:
            assert c.split in ("train", "val"), p.task_id
    save_cards(suite, tmp_path)
    loaded = {c.public.task_id: c for k in SPLITS for c in load_split_cards(k, tmp_path)}
    assert loaded == {c.public.task_id: c for c in suite}


# ---- LLM-backed parts, stubbed ----

class StubLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, instructions, user, schema):
        self.calls.append((instructions, user, schema))
        r = self.replies.pop(0)
        return json.dumps(r(user) if callable(r) else r)


def test_draft_commands_filters_thresholds():
    c = make_card("precise_place", robot("arm"), 0, "neutral", "train")
    llm = StubLLM([{"commands": ["center the block on the plate", "put it within 5 mm of the middle",
                                 c.public.command, "center the block on the plate"]},
                   {"commands": ["set the block right in the middle of the plate"]}])
    out = draft_commands(c, 2, llm=llm)
    assert out == ["center the block on the plate", "set the block right in the middle of the plate"]
    for ins, user, _ in llm.calls:
        text = ins + user
        assert not any(re.search(rf"(?<![\d.]){re.escape(repr(t))}(?!\d)", text) for t in thresholds(c.test))
        assert "predicates" not in text and "tol_" not in text


def _proposal(user, **over):
    goal = re.search(r"label the task with: ([a-z_]+)", user).group(1)
    p = {"goal": goal, "style": "gentle", "command": "lift the mug and hold it up",
         "predicates": [{"name": "held", "args": ["mug_0"], "params": []},
                        {"name": "upright", "args": ["mug_0"], "params": [{"name": "tol_rad", "value": 0.2}]}],
         "graded": [{"name": "accel", "args": ["mug_0"],
                     "params": [{"name": "good", "value": 0.5}, {"name": "bad", "value": 2.0}]}]}
    p.update(over)
    return p


def test_propose_keeps_only_oracle_solved():
    bad_ptr = lambda u: _proposal(u, predicates=[{"name": "held", "args": ["vase_9"], "params": []}])
    leak = lambda u: _proposal(u, command="hold the mug within 0.2 rad of upright")
    llm = StubLLM([bad_ptr, leak, _proposal, _proposal, _proposal])
    zero_calls, solve_calls = [], []

    def zero(card):
        zero_calls.append(card.public.task_id)
        return {"success": len(zero_calls) == 1, "score": 0.0}  # first valid one: default behaviour passes

    def solve(card):
        solve_calls.append(card.public.task_id)
        return {"success": len(solve_calls) == 2, "score": 1.0}  # oracle fails once, then solves

    log: list = []
    kept = propose_tasks(1, ["arm"], solve=solve, zero=zero, llm=llm, robots=lambda f, rng: robot(f), seed=0,
                         max_tries=5, log=log)
    assert len(kept) == 1 and len(llm.calls) == 5
    assert "unknown pointer" in log[0]["reject"] and "threshold" in log[1]["reject"]
    assert log[2]["reject"] == "zero program passes" and log[3]["reject"] == "oracle fails"
    c = kept[0]
    assert c.source == "proposed" and c.public.task_id == log[4]["kept"] == solve_calls[-1]
    assert c.public.goal in GOALS and c.test.predicates[0] == {"pred": "held", "args": ["mug_0"]}
    # the scene is rebuilt from the public part alone, and the test's pointers exist in it
    _, scene = build_scene(c.public)
    assert not _pointer_ok(c, scene)
    assert "mug_0" in llm.calls[-1][1] and "predicates" in json.dumps(llm.calls[-1][2])


def test_validate_proposal_errors():
    r = robot("quadruped")
    _, scene = make_scene("floor_obstacles", r, 0)
    out = {"predicates": [{"name": "teleport", "args": [], "params": []},
                          {"name": "open_frac", "args": ["goal_0"], "params": []},
                          {"name": "in_region", "args": ["ee", "goal_0"], "params": []}],
           "graded": [{"name": "speed", "args": ["base"], "params": [{"name": "good", "value": 1.0}]}]}
    errs = validate_proposal(out, r, scene)
    assert any("unknown predicate" in e for e in errs)
    assert any("open_frac: missing" in e for e in errs)
    assert any("unknown pointer ee" in e for e in errs)  # quadrupeds have no ee frame
    assert any("needs good and bad" in e for e in errs)


@pytest.mark.slow
def test_draft_commands_live():
    # one live gpt-6.1-sol call: set -a; . ~/.config/languagespec/oai.env; set +a
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    c = make_card("level_carry", robot("arm"), 0, "gentle", "train")
    out = draft_commands(c, 3, rounds=1)
    assert 1 <= len(out) <= 3 and all(not has_threshold(o, c.test) for o in out)
