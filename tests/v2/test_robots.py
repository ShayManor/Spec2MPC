import subprocess
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from spec2mpc.core import load_family_config
from spec2mpc.robots.v2 import (
    FAMILIES,
    build_model,
    describe_robot,
    frame_ref,
    list_robots,
    load_robot,
    menagerie_path,
    robot_frames_as_entities,
)

ROOT = Path(__file__).parent.parent.parent
ALL = [(f, n) for f in FAMILIES for n in list_robots(f)]


def test_families_match_configs():
    assert sorted(FAMILIES) == sorted(p.stem for p in (ROOT / "configs/v2/robots").glob("*.yaml"))
    for f in FAMILIES:
        assert list_robots(f), f


def test_menagerie_pinned():
    lock = yaml.safe_load((ROOT / "configs/v2/menagerie.lock").read_text())
    head = subprocess.run(["git", "-C", str(menagerie_path()), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == lock["commit"]


@pytest.mark.parametrize("family,name", ALL)
def test_every_family_loads(family, name):
    r = load_robot(family, name, None)
    m = build_model(r)
    assert isinstance(m, mujoco.MjModel)
    assert r.frames and r.actuators and len(r.actuators) == m.nu
    assert r.floors == load_family_config(family)["floors"]
    for fr in r.frames:
        kind, ref = frame_ref(r, fr)
        obj = {"body": mujoco.mjtObj.mjOBJ_BODY, "site": mujoco.mjtObj.mjOBJ_SITE, "geom": mujoco.mjtObj.mjOBJ_GEOM}[kind]
        assert mujoco.mj_name2id(m, obj, ref) >= 0, (name, fr, ref)
    ents = robot_frames_as_entities(r, m.qpos0)
    assert [e.id for e in ents] == r.frames
    assert all(e.kind == "robot_frame" and e.pose.shape == (7,) and np.isfinite(e.pose).all() for e in ents)


@pytest.mark.parametrize("family", FAMILIES)
def test_randomization_seeded(family):
    name = list_robots(family)[0]
    a, b, c = load_robot(family, name, 1), load_robot(family, name, 1), load_robot(family, name, 2)
    assert a == b
    assert a.params["mass_kg"] != c.params["mass_kg"]
    ma, mc = build_model(a), build_model(c)
    assert ma.body_mass.sum() == pytest.approx(a.params["mass_kg"], rel=1e-6)
    assert mc.body_mass.sum() == pytest.approx(c.params["mass_kg"], rel=1e-6)


@pytest.mark.parametrize("family", FAMILIES)
def test_randomization_in_range(family):
    ranges = load_family_config(family)["ranges"]
    name = list_robots(family)[0]
    nom = load_robot(family, name, None)
    for seed in range(20):
        r = load_robot(family, name, seed)
        for k, (lo, hi) in ranges["params"].items():
            assert lo <= r.params[k] <= hi, (k, r.params[k])
        for k, (lo, hi) in ranges["perception"].items():
            assert lo <= r.perception[k] <= hi, (k, r.perception[k])
        for k, choices in ranges["runtime"].items():
            assert r.runtime[k] in choices, (k, r.runtime[k])
        m = build_model(r)
        assert m.body_mass.sum() == pytest.approx(nom.params["mass_kg"] * r.params["mass_scale"], rel=1e-6)


def test_randomized_model_applies_scales():
    nom, r = load_robot("arm", "panda", None), load_robot("arm", "panda", 3)
    m0, m1 = build_model(nom), build_model(r)
    np.testing.assert_allclose(m1.body_mass, m0.body_mass * r.params["mass_scale"])
    np.testing.assert_allclose(m1.geom_friction[:, 0], m0.geom_friction[:, 0] * r.params["friction_scale"])
    np.testing.assert_allclose(m1.dof_damping, m0.dof_damping * r.params["damping_scale"])


@pytest.mark.parametrize("family", FAMILIES)
def test_describe_mentions_limits(family):
    r = load_robot(family, list_robots(family)[0], 0)
    m = build_model(r)
    text = r.describe()
    for a in r.actuators:
        assert a in text
    for k, v in r.limits.items():
        assert f"{k}={v:g}" in text
    # every limited actuator contributes its control range
    for i in range(m.nu):
        if m.actuator_ctrllimited[i]:
            assert f"{r.actuators[i]}_ctrl_max" in r.limits
    full = describe_robot(r)
    assert text in full and "joints:" in full


def test_frames_by_family():
    assert "ee" in load_robot("arm", "panda", None).frames
    assert "ee" not in load_robot("quadruped", "go2", None).frames
    assert {"foot_fl", "foot_fr", "foot_rl", "foot_rr"} <= set(load_robot("quadruped", "go2", None).frames)
    assert {"left_hand", "right_hand"} <= set(load_robot("humanoid", "h1", None).frames)


def test_unknown_member_raises():
    with pytest.raises(KeyError):
        load_robot("arm", "nope", None)


def test_kart_nominal_dims():
    m = build_model(load_robot("kart", "kart", None))
    fl = m.body("steer_fl").pos
    rl = m.body("wheel_rl").pos
    assert fl[0] - rl[0] == pytest.approx(1.2)  # wheelbase, v1 EVC nominal
    assert m.body_mass.sum() == pytest.approx(210.0, abs=0.5)
