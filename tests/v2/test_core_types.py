import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from spec2mpc.core import (
    Entity,
    Episode,
    Program,
    RobotSpec,
    Scene,
    Stage,
    TaskCard,
    TaskPublic,
    TaskTest,
    Term,
    from_jsonable,
    to_jsonable,
)

ROOT = Path(__file__).parent.parent.parent
FAMILIES = ("arm", "mobile_base", "mobile_manipulator", "rover2d", "kart", "quadruped", "humanoid", "quadrotor")


def _robot() -> RobotSpec:
    return RobotSpec(
        family="arm", name="panda", mjcf_path="assets/panda.xml", params={"mass_kg": 18.0},
        frames=["ee", "base"], actuators=["j1", "j2"], limits={"j1_vel_radps": 2.175},
        floors={"min_clearance_m": 0.02, "max_speed_frac": 0.5, "max_force_frac": 0.5},
        runtime={"control_rate_hz": 50.0}, perception={"loc_noise_std_m": 0.005},
    )


def _scene() -> Scene:
    mug = Entity("mug_0", "mug", "object", np.array([0.5, 0.0, 0.1, 1, 0, 0, 0.0]),
                 np.array([0.08, 0.08, 0.1]), None, None)
    return Scene([mug], None, 0.0)


def _program() -> Program:
    return Program([
        Stage([Term("pose_to", ["ee", "mug_0"], {"z_offset_m": 0.1, "w": 1.0}), Term("gripper", ["open"], {})],
              Term("dist_below", ["ee", "mug_0"], {"d": 0.02})),
        Stage([Term("gripper", ["close"], {"force_N": 20.0})], Term("gripper_closed", [], {})),
        Stage([Term("relative_pose", ["mug_0", "ee"], {}), Term("smooth", [], {"w": 0.1})],
              Term("timeout", [], {"t_s": 5.0})),
    ])


def test_program_text_roundtrip():
    p = _program()
    assert Program.from_text(p.to_text()) == p
    assert Program.from_json(p.to_json()) == p
    assert Program.from_json(json.dumps(p.to_json())) == p
    assert "UNTIL dist_below(ee,mug_0; d=0.02)" in p.to_text()


def test_public_view_has_no_test_fields():
    test = TaskTest("t0", [{"pred": "on", "args": ["mug_0", "shelf_0"], "tol_m": 0.0137}],
                    [{"term": "time_below", "t_s": 41.5}], 60.0)
    card = TaskCard(TaskPublic("t0", "arm", _robot(), 7, "put the mug on the shelf", None), test, "train", "gen")
    s = json.dumps(to_jsonable(card.public_view()))
    for bad in ("predicates", "graded", "tol_m", "0.0137", "41.5", "time_below"):
        assert bad not in s
    assert from_jsonable(json.loads(s)) == card.public


def test_scene_get_unknown_raises():
    sc = _scene()
    assert sc.get("mug_0").label == "mug"
    with pytest.raises(KeyError):
        sc.get("nope")


def test_numpy_equality_and_json_roundtrip():
    sc = _scene()
    assert from_jsonable(json.loads(json.dumps(to_jsonable(sc)))) == sc
    ep = Episode("t0", "stub", [_program()], {"qpos": np.zeros((3, 2), np.float32)}, "done", 1.5)
    back = from_jsonable(json.loads(json.dumps(to_jsonable(ep))))
    assert back == ep and back.trace["qpos"].dtype == np.float32


def test_robot_floors_loaded():
    paths = sorted((ROOT / "configs" / "v2" / "robots").glob("*.yaml"))
    assert sorted(p.stem for p in paths) == sorted(FAMILIES)
    for p in paths:
        cfg = yaml.safe_load(p.read_text())
        need = {"min_clearance_m", "max_speed_frac", "max_force_frac"}
        if p.stem in ("quadrotor", "humanoid"):
            need.add("max_tilt_rad")
        assert set(cfg["floors"]) == need, p
        assert cfg["solver"] == ("kart_mpcc" if p.stem == "kart" else "sampling")


def test_describe_mentions_limits_and_floors():
    d = _robot().describe()
    for s in ("panda", "ee", "j1_vel_radps", "max_speed_frac"):
        assert s in d
