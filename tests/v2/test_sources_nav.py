import json

import mujoco
import numpy as np
import pytest

from spec2mpc.core import Program, Stage, Term, to_jsonable
from spec2mpc.tasks.sources import COVERAGE_DIR, card_errors, command_leaks
from spec2mpc.tasks.sources import _nav as N
from spec2mpc.tasks.sources import barn, commonroad, playground, tracks

MODS = {"playground": playground, "barn": barn, "commonroad": commonroad, "tracks": tracks}
requires_playground = pytest.mark.skipif(not playground.available(), reason="MuJoCo Playground assets absent")
requires_barn = pytest.mark.skipif(not barn.available(), reason="BARN assets absent")
requires_commonroad = pytest.mark.skipif(not commonroad.available(), reason="CommonRoad scenarios absent")
requires_tracks = pytest.mark.skipif(not tracks.available(), reason="track assets absent")
REQUIRES = {"playground": requires_playground, "barn": requires_barn, "commonroad": requires_commonroad,
            "tracks": requires_tracks}
SOURCES = [pytest.param(s, marks=REQUIRES[s]) for s in MODS]


def _goto(card, scene) -> Program:
    # a reference go-to program; validation shows every card is expressible on its scene and robot
    goal = "finish" if any(e.id == "finish" for e in scene.entities) else "goal"
    return Program([Stage([Term("follow_path", ["base", "route"], {"v_ref_frac": 0.5}),
                           Term("clearance", ["robot", "scene"], {"margin_m": 0.1})],
                          Term("dist_below", ["base", goal], {"d": 1.0})),
                    Stage([Term("pose_to", ["base", goal], {})], Term("timeout", [], {"t_s": card.test.horizon_s}))])


def _check(source: str, n: int = 10, seed: int = 0) -> list:
    mod = MODS[source]
    cards = mod.cards(n, seed)
    assert len(cards) == n
    try:
        from spec2mpc.program import validate
    except ImportError:  # Task 1 not landed
        validate = None
    for c in cards:
        assert c.source == source and c.split == "train" and c.public.task_id.startswith(source + "/")
        xml, sc = mod.scene(c)
        mujoco.MjModel.from_xml_string(xml)  # compiles
        ids = [e.id for e in sc.entities]
        assert {"start", "route"} <= set(ids)
        assert card_errors(c, ids) == [], (c.public.task_id, card_errors(c, ids))
        assert not command_leaks(c.public.command, c.test.predicates)
        assert not any(ch.isdigit() for ch in c.public.command)
        assert 0 < c.test.horizon_s <= 600
        pub = json.dumps(to_jsonable(c.public_view()))
        assert "predicates" not in pub and "graded" not in pub
        low = (xml + c.public.robot.mjcf_path).lower()
        assert not any(w in low for w in N.BENCHMARK_WORDS)
        if validate is not None:
            assert validate(_goto(c, sc), c.public.robot, sc) == [], c.public.task_id
    return cards


@requires_playground
def test_playground_cards():
    cards = _check("playground")
    assert {c.public.family for c in playground.cards(16, 1)} == {"quadruped", "humanoid"}
    assert all(any(d["pred"] == "not_fallen" for d in c.test.predicates) for c in cards)


@requires_barn
def test_barn_cards():
    cards = _check("barn")
    assert all(c.public.family == "rover2d" for c in cards)


@requires_commonroad
def test_commonroad_cards():
    cards = _check("commonroad")
    assert all(c.public.family == "kart" for c in cards)


@requires_tracks
def test_tracks_cards():
    _check("tracks")
    fams = {c.public.family for c in tracks.cards(30, 0)}
    assert fams == {"kart", "quadrotor"}


@pytest.mark.parametrize("source", SOURCES)
def test_translation_coverage_logged(source):
    cov = MODS[source].coverage()
    path = COVERAGE_DIR / f"{source}_coverage.json"
    assert path.exists()
    disk = json.loads(path.read_text())
    assert disk["translated"] == cov["translated"] > 0
    assert disk["translated"] + disk["skipped"] == disk["total"]


@pytest.mark.parametrize("source", SOURCES)
def test_cards_seeded_and_scene_rebuilds(source):
    mod = MODS[source]
    a, b, c = mod.cards(5, 3), mod.cards(5, 3), mod.cards(5, 4)
    assert [x.public.task_id for x in a] == [x.public.task_id for x in b]
    assert [x.public.task_id for x in a] != [x.public.task_id for x in c]
    xml1, s1 = mod.scene(a[0])
    xml2, s2 = mod.scene(a[0].public_view())  # rebuilt from public info alone
    assert xml1 == xml2 and s1 == s2


@requires_barn
def test_barn_route_clear_of_obstacles():
    for i in range(0, barn.N_WORLDS, 30):
        rt, cyl = barn.route(i)[1:-1, :2], barn.cylinders(i)
        assert np.min(np.linalg.norm(rt[:, None] - cyl[None], axis=2)) > 2 * barn.CYL_R_M


@requires_playground
def test_playground_skips_untranslatable():
    cov = playground.coverage()
    assert cov["skipped"] == len(playground.SKIPPED) and "Go1Handstand" in cov["skipped_tasks"]


@requires_commonroad
def test_commonroad_obstacles_and_trajectories():
    for c in commonroad.cards(15, 0):
        _, sc = commonroad.scene(c)
        trajs = commonroad.trajectories(c)
        for eid, tr in trajs.items():
            e = sc.get(eid)
            assert tr.shape[1] == 4 and np.allclose(tr[0, 1:3], e.pose[:2])
            assert np.all(np.diff(tr[:, 0]) > 0)
        no_contact = {d["args"][1] for d in c.test.predicates if d["pred"] == "no_contact"}
        assert no_contact == {e.id for e in sc.entities if e.kind == "object"}


@requires_tracks
def test_tracks_segment_inside_walls():
    for c in tracks.cards(20, 2):
        if c.public.family != "kart":
            continue
        _, sc = tracks.scene(c)
        rt = sc.get("route").points[:, :2]
        walls = np.vstack([sc.get("wall_left").points[:, :2], sc.get("wall_right").points[:, :2]])
        assert np.min(np.linalg.norm(rt[:, None] - walls[None], axis=2)) > 0.5
        fin = sc.get("finish").points[:, :2]
        assert np.linalg.norm(fin.mean(0) - rt[-1]) < 2 * float(np.max(np.linalg.norm(fin - fin.mean(0), axis=1)))
