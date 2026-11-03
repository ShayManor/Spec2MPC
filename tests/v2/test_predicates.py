import numpy as np
import pytest

from spec2mpc.core import TaskTest
from spec2mpc.rubric.v2 import GRADED, PENALTIES, PREDICATES, evaluate, score

T = 11
Q_UP = [1.0, 0.0, 0.0, 0.0]
Q_TIP = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]  # 90 deg about x


def _pose(xyz, q=Q_UP) -> np.ndarray:
    return np.tile(np.array([*xyz, *q], np.float64), (T, 1))  # [T, 7]


def _move(a, b, q=Q_UP) -> np.ndarray:
    # linear motion from a to b over T steps, [T, 7]
    p = np.linspace(np.array(a, float), np.array(b, float), T)
    return np.concatenate([p, np.tile(np.array(q, float), (T, 1))], axis=1)


def _trace(**kw) -> dict:
    tr = {"t_s": np.linspace(0.0, 5.0, T), "contacts": [[] for _ in range(T)]}
    tr.update(kw)
    return tr


def _end(pred: dict, tr: dict) -> bool:
    return evaluate([pred], tr)[0]


def P(name, *args, **params) -> dict:
    return {"pred": name, "args": list(args), **params}


# a 0.1 m cube mug and a 0.6 x 0.4 x 0.02 m plate top at z=0.02
TABLE = {"entity_pose/plate": _pose([0.0, 0.0, 0.0]), "entity_extent/plate": np.array([0.3, 0.2, 0.02]),
         "entity_extent/mug": np.array([0.05, 0.05, 0.05])}


def test_vocab_exact():
    assert set(PREDICATES) == {"on", "inside", "open_frac", "upright", "at_pose", "in_region", "not_fallen",
                               "time_below", "no_contact", "held"}


def test_on():
    tr = _trace(**TABLE, **{"entity_pose/mug": _move([0.5, 0, 0.3], [0.1, 0.05, 0.07])})
    assert _end(P("on", "mug", "plate"), tr)
    tr["entity_pose/mug"] = _move([0.1, 0.05, 0.07], [0.5, 0, 0.07])  # slid off the footprint
    assert not _end(P("on", "mug", "plate"), tr)
    tr["entity_pose/mug"] = _pose([0.1, 0.05, 0.2])  # floating above
    assert not _end(P("on", "mug", "plate"), tr)


def test_inside():
    tr = _trace(**{"entity_pose/drawer": _pose([1, 0, 0.5]), "entity_extent/drawer": np.array([0.2, 0.2, 0.1]),
                   "entity_pose/mug": _pose([1.1, -0.1, 0.45])})
    assert _end(P("inside", "mug", "drawer"), tr)
    tr["entity_pose/mug"] = _pose([1.1, -0.1, 0.7])
    assert not _end(P("inside", "mug", "drawer"), tr)


def test_inside_rotated_container():
    # container yawed 90 deg: its long x half-extent now lies along world y
    qz = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    tr = _trace(**{"entity_pose/bin": _pose([0, 0, 0], qz), "entity_extent/bin": np.array([0.5, 0.1, 0.1]),
                   "entity_pose/mug": _pose([0.0, 0.4, 0.0])})
    assert _end(P("inside", "mug", "bin"), tr)
    tr["entity_pose/mug"] = _pose([0.4, 0.0, 0.0])
    assert not _end(P("inside", "mug", "bin"), tr)


def test_open_frac():
    tr = _trace(**{"open_frac/drawer": np.linspace(0, 0.8, T)})
    assert _end(P("open_frac", "drawer", min=0.7), tr)
    assert not _end(P("open_frac", "drawer", min=0.9), tr)


def test_upright():
    tr = _trace(**{"entity_pose/mug": _pose([0, 0, 0])})
    assert _end(P("upright", "mug", tol_rad=0.1), tr)
    tr["entity_pose/mug"] = _pose([0, 0, 0], Q_TIP)
    assert not _end(P("upright", "mug", tol_rad=0.1), tr)


def test_at_pose_param_and_entity():
    tr = _trace(**{"entity_pose/ee": _move([0, 0, 0], [0.5, 0.0, 0.3]), "entity_pose/goal": _pose([0.51, 0, 0.3])})
    assert _end(P("at_pose", "ee", pose=[0.5, 0, 0.3, 1, 0, 0, 0], tol_m=0.02, tol_rad=0.1), tr)
    assert _end(P("at_pose", "ee", "goal", tol_m=0.02), tr)
    assert not _end(P("at_pose", "ee", "goal", tol_m=0.005), tr)
    assert not _end(P("at_pose", "ee", pose=[0.5, 0, 0.3, *Q_TIP], tol_m=0.02, tol_rad=0.1), tr)
    assert _end(P("at_pose", "ee", pose=[0.5, 0, 0.3], tol_m=0.02), tr)  # position only


def test_in_region_box_and_polygon():
    tr = _trace(**{"entity_pose/base": _move([0, 0, 0.3], [2.0, 1.0, 0.3]),
                   "entity_pose/dock": _pose([2.0, 1.0, 0.0]), "entity_extent/dock": np.array([0.3, 0.3, 1.0]),
                   "entity_points/zone": np.array([[1.5, 0.5, 0], [2.5, 0.5, 0], [2.5, 1.5, 0], [1.5, 1.5, 0]])})
    assert _end(P("in_region", "base", "dock"), tr)
    assert _end(P("in_region", "base", "zone"), tr)
    tr["entity_pose/base"] = _pose([0, 0, 0.3])
    assert not _end(P("in_region", "base", "dock"), tr)
    assert not _end(P("in_region", "base", "zone"), tr)


def test_not_fallen():
    h = np.full(T, 0.3)
    tr = _trace(robot_base_height=h)
    assert _end(P("not_fallen", "robot", min_height_m=0.2), tr)
    assert _end(P("not_fallen", "robot"), tr)  # default: half the initial height
    h2 = h.copy()
    h2[4] = 0.05  # fell then got up: still a failure
    tr["robot_base_height"] = h2
    assert not _end(P("not_fallen", "robot", min_height_m=0.2), tr)
    tr = _trace(robot_base_height=h, robot_base_pose=_pose([0, 0, 0.3], Q_TIP))
    assert not _end(P("not_fallen", "robot", min_height_m=0.2, max_tilt_rad=0.8), tr)


def test_time_below_uses_completion_time():
    # mug reaches the plate at step 5 (t=2.5 s) and stays; the episode runs to 5 s
    p = np.concatenate([_move([0.5, 0, 0.3], [0.1, 0, 0.07])[::2], _pose([0.1, 0, 0.07])[:5]])
    tr = _trace(**TABLE, **{"entity_pose/mug": p})
    preds = [P("on", "mug", "plate"), P("time_below", t_s=3.5)]
    assert evaluate(preds, tr) == [True, True]
    preds[1] = P("time_below", t_s=2.5)
    assert evaluate(preds, tr) == [True, False]
    # never completes: time_below fails
    tr["entity_pose/mug"] = _pose([0.5, 0, 0.3])
    assert evaluate([P("on", "mug", "plate"), P("time_below", t_s=100.0)], tr) == [False, False]
    # alone: true iff the episode ended before t
    assert _end(P("time_below", t_s=6.0), tr) and not _end(P("time_below", t_s=4.0), tr)


def test_no_contact():
    c = [[] for _ in range(T)]
    tr = _trace(contacts=c)
    assert _end(P("no_contact", "robot", "vase"), tr)
    c[3] = [("robot:link7", "vase", 4.0)]
    assert not _end(P("no_contact", "robot", "vase"), tr)  # prefix match on robot parts
    assert _end(P("no_contact", "robot", "mug"), tr)
    tr["contacts"] = np.array([[("vase", "robot:hand")] if i == 2 else [] for i in range(T)], dtype=object)
    assert not _end(P("no_contact", "vase", "robot"), tr)  # either order, object array


def test_held():
    c = [[("mug", "plate")] for _ in range(T)]
    c[-1] = [("robot:left_finger", "mug", 10.0), ("mug", "robot:right_finger", 10.0)]
    tr = _trace(contacts=c)
    assert _end(P("held", "mug"), tr)
    c[-1] = c[-1] + [("mug", "plate")]  # still resting on the plate
    assert not _end(P("held", "mug"), tr)
    c[-1] = [("mug", "plate")]
    assert not _end(P("held", "mug"), tr)


def test_nested_dict_form_accepted():
    tr = _trace(**{"open_frac/drawer": np.full(T, 0.8)})
    assert _end({"type": "open_frac", "args": ["drawer"], "params": {"min": 0.5}}, tr)


def test_unknown_predicate_raises():
    with pytest.raises(ValueError):
        evaluate([P("levitates", "mug")], _trace())


def test_missing_trace_key_raises():
    with pytest.raises(KeyError):
        evaluate([P("upright", "ghost", tol_rad=0.1)], _trace())


def _task(preds, graded=()) -> TaskTest:
    return TaskTest("t0", list(preds), list(graded), 10.0)


def test_success_requires_all():
    tr = _trace(**TABLE, **{"entity_pose/mug": _pose([0.1, 0.05, 0.07])})
    ok = score(_task([P("on", "mug", "plate"), P("upright", "mug", tol_rad=0.1)]), tr)
    assert ok["success"] and ok["score"] == pytest.approx(1.0)
    bad = score(_task([P("on", "mug", "plate"), P("upright", "mug", tol_rad=0.1),
                       P("inside", "mug", "plate")]), tr)
    assert not bad["success"]
    assert bad["score"] == pytest.approx(2 / 3)
    assert list(bad["per_predicate"].values()) == [True, True, False]


def test_graded_terms_and_ramp():
    assert set(GRADED) >= {"time", "speed", "accel", "jerk", "tilt", "force", "clearance", "energy"}
    tr = _trace(**TABLE, **{"entity_pose/mug": _move([0, 0, 0.07], [0.5, 0, 0.07])},  # 0.1 m/s
                min_clearance_m=np.linspace(0.3, 0.1, T), energy_J=np.linspace(0, 20, T))
    g = [{"term": "speed", "args": ["mug"], "good": 0.05, "bad": 0.15},  # 0.1 -> 0.5
         {"term": "clearance", "good": 0.2, "bad": 0.0},  # min 0.1 -> 0.5
         {"term": "energy", "good": 0.0, "bad": 40.0},  # final 20 -> 0.5
         {"term": "tilt", "args": ["mug"], "good": 0.0, "bad": 0.5}]  # 0 -> 1
    out = score(_task([P("upright", "mug", tol_rad=0.1)], g), tr)
    assert [round(v, 6) for v in out["graded"].values()] == [0.5, 0.5, 0.5, 1.0]
    assert out["score"] == pytest.approx((1.0 + 0.5 + 0.5 + 0.5 + 1.0) / 5)


def test_graded_time():
    tr = _trace(**TABLE, **{"entity_pose/mug": _move([0.1, 0.05, 0.3], [0.1, 0.05, 0.07])})  # done at 5 s
    g = [{"term": "time", "good": 0.0, "bad": 10.0}]
    out = score(_task([P("on", "mug", "plate")], g), tr)
    assert out["graded"]["0:time"] == pytest.approx(0.5)
    tr["entity_pose/mug"] = _pose([0.1, 0.05, 0.3])  # never completes
    assert score(_task([P("on", "mug", "plate")], g), tr)["graded"]["0:time"] == 0.0


def test_graded_force():
    c = [[("robot:hand", "mug", 5.0 * i)] for i in range(T)]  # peak 50 N
    out = score(_task([], [{"term": "force", "args": ["robot", "mug"], "good": 0.0, "bad": 100.0}]), _trace(contacts=c))
    assert out["graded"]["0:force"] == pytest.approx(0.5)


def test_score_range():
    rng = np.random.default_rng(0)
    for _ in range(50):
        tr = _trace(**TABLE, **{"entity_pose/mug": np.concatenate(
            [rng.uniform(-1, 1, (T, 3)), np.tile(Q_UP, (T, 1))], axis=1)},
            min_clearance_m=rng.uniform(-1, 1, T), energy_J=np.cumsum(rng.uniform(0, 10, T)))
        g = [{"term": "speed", "args": ["mug"], "good": 0.1, "bad": 1.0},
             {"term": "clearance", "good": 0.2, "bad": 0.0}, {"term": "energy", "good": 0, "bad": 50},
             {"term": "time", "good": 1, "bad": 4}, {"term": "jerk", "args": ["mug"], "good": 0, "bad": 50},
             {"term": "accel", "args": ["mug"], "good": 0, "bad": 5}]
        out = score(_task([P("on", "mug", "plate"), P("upright", "mug", tol_rad=0.2)], g), tr)
        assert 0.0 <= out["score"] <= 1.0


def test_penalties_subtract_after():
    assert set(PENALTIES) >= {"contact_events", "fell", "count"}
    c = [[] for _ in range(T)]
    for i in (2, 3, 7):
        c[i] = [("robot:hand", "vase")]  # two onsets: steps 2 and 7
    tr = _trace(**TABLE, **{"entity_pose/mug": _pose([0.1, 0.05, 0.07])}, contacts=c,
                solver_failures=np.array([0] * 5 + [1] * 6), robot_base_height=np.full(T, 0.3))
    g = [{"term": "contact_events", "args": ["robot", "vase"], "weight": 0.1},
         {"term": "count", "args": ["solver_failures"], "weight": 0.2},
         {"term": "fell", "args": ["robot"], "weight": 1.0}]
    out = score(_task([P("on", "mug", "plate")], g), tr)
    assert out["penalties"] == {"0:contact_events": pytest.approx(0.2), "1:count": pytest.approx(0.2),
                                "2:fell": 0.0}
    assert out["score"] == pytest.approx(1.0 - 0.4)
    assert out["success"]
