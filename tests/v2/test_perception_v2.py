from __future__ import annotations

import mujoco
import numpy as np
import pytest

from spec2mpc.core.types import Entity, RobotSpec, Scene
from spec2mpc.perception.v2 import add_grasps, perceive_gt

# tiny tabletop: one actuated link (robot), a static table (fixture), free objects
XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="1 -1 2" dir="-0.5 0.5 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.6 0.6 0.6 1"/>
    <body name="base" pos="-0.5 0 0.4">
      <site name="base" size="0.01"/>
      <geom type="cylinder" size="0.05 0.05" rgba="0.2 0.2 0.2 1"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" rgba="0.3 0.3 0.3 1"/>
        <site name="gripper" pos="0.3 0 0" size="0.01"/>
      </body>
    </body>
    <body name="table" pos="0 0 0.2">
      <geom type="box" size="0.4 0.4 0.2" rgba="0.75 0.6 0.4 1"/>
    </body>
    <body name="red_mug_0" pos="0.05 0.1 0.45">
      <freejoint/>
      <geom type="cylinder" size="0.035 0.05" rgba="0.9 0.05 0.05 1"/>
      <geom type="box" pos="0.05 0 0" size="0.015 0.008 0.03" rgba="0.9 0.05 0.05 1"/>
    </body>
    <body name="blue_box_0" pos="-0.15 -0.15 0.43">
      <freejoint/>
      <geom type="box" size="0.03 0.03 0.03" rgba="0.1 0.2 0.9 1"/>
    </body>
    <body name="big_crate_0" pos="0.2 -0.2 0.55">
      <freejoint/>
      <geom type="box" size="0.15 0.15 0.15" rgba="0.3 0.6 0.3 1"/>
    </body>
  </worldbody>
  <actuator><motor joint="j1"/></actuator>
</mujoco>
"""

PROFILE = {"range_m": 5.0, "fov_rad": 6.2832, "loc_noise_std_m": 0.01, "loc_bias_m": 0.005}


def _robot(perception: dict[str, float]) -> RobotSpec:
    return RobotSpec("arm", "stub", "", {}, ["base", "gripper"], ["j1"], {}, {}, {}, perception)


def _sim() -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _gt(profile: dict[str, float], seed: int) -> Scene:
    m, d = _sim()
    return perceive_gt(m, d, _robot(profile), profile, np.random.default_rng(seed))


def test_gt_noise_seeded() -> None:
    a, b, c = _gt(PROFILE, 0), _gt(PROFILE, 0), _gt(PROFILE, 1)
    assert a == b
    assert not np.allclose(a.get("red_mug_0").pose[:3], c.get("red_mug_0").pose[:3])
    # zero noise reproduces the simulator exactly
    exact = _gt({**PROFILE, "loc_noise_std_m": 0.0, "loc_bias_m": 0.0}, 0)
    assert np.allclose(exact.get("red_mug_0").pose[:3], [0.05, 0.1, 0.45])
    # noise is bounded by a few std plus the bias
    err = np.linalg.norm(a.get("red_mug_0").pose[:3] - exact.get("red_mug_0").pose[:3])
    assert 0 < err < 5 * 0.01 * np.sqrt(3) + 0.005


def test_gt_entities_cover_objects() -> None:
    s = _gt(PROFILE, 0)
    kinds = {e.id: e.kind for e in s.entities}
    for obj in ("red_mug_0", "blue_box_0", "big_crate_0"):
        assert kinds[obj] == "object"
    assert kinds["table"] == "fixture"
    assert kinds["base"] == kinds["gripper"] == "robot_frame"
    assert "link1" not in kinds  # robot links are not scene entities
    mug = s.get("red_mug_0")
    assert mug.label == "red mug"
    assert mug.points is not None and mug.points.shape[1] == 3 and len(mug.points) >= 64
    assert np.allclose(mug.extent, [0.065, 0.035, 0.05], atol=0.005)
    # robot frames are proprioceptive: exact
    m, d = _sim()
    assert np.allclose(s.get("gripper").pose[:3], d.site("gripper").xpos)


def test_gt_range_drops_far() -> None:
    # sensing from the robot base at x=-0.5: crate at ~0.75 m drops at range 0.7, mug at ~0.57 stays
    s = _gt({**PROFILE, "range_m": 0.7}, 0)
    ids = {e.id for e in s.entities}
    assert "red_mug_0" in ids and "big_crate_0" not in ids
    assert "gripper" in ids  # robot frames never drop


def test_grasps_for_graspables_only() -> None:
    s = add_grasps(_gt({**PROFILE, "loc_noise_std_m": 0.0, "loc_bias_m": 0.0}, 0), n=8)
    grasps = [e for e in s.entities if e.kind == "grasp"]
    parents = {g.parent for g in grasps}
    assert parents == {"red_mug_0", "blue_box_0"}  # crate too wide, table and robot not graspable
    for pid in parents:
        gs = [g for g in grasps if g.parent == pid]
        assert 1 <= len(gs) <= 8
        obj = s.get(pid)
        for g in gs:
            assert g.points.shape == (2, 3)  # the two finger contacts
            width = np.linalg.norm(g.points[0] - g.points[1])
            assert width <= 0.08 + 1e-6
            # grasp center lies inside the object's box, closing axis is the grasp y axis
            assert np.all(np.abs(g.pose[:3] - obj.pose[:3]) <= obj.extent + 0.02)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, g.pose[3:])
            y = R.reshape(3, 3)[:, 1]
            axis = (g.points[1] - g.points[0]) / width
            assert abs(abs(y @ axis) - 1) < 1e-3
    # deterministic
    assert add_grasps(s, n=8) == add_grasps(s, n=8)


@pytest.mark.slow
def test_image_stack_finds_red_mug() -> None:
    from spec2mpc.perception.v2 import perceive_images

    m, d = _sim()
    H, W, fovy = 480, 640, 60.0
    # camera 1 m in front of and above the table, looking at the table center
    eye, target = np.array([0.9, 0.0, 1.1]), np.array([0.0, 0.0, 0.4])
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = target
    cam.distance = float(np.linalg.norm(target - eye))
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cam.elevation = float(np.degrees(np.arcsin(fwd[2])))
    m.vis.global_.fovy = fovy
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, cam)
    rgb = r.render()
    r.enable_depth_rendering()
    r.update_scene(d, cam)
    depth = r.render()
    r.close()
    f = 0.5 * H / np.tan(np.radians(fovy) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    cam_pose = np.eye(4)  # camera to world, OpenCV axes (x right, y down, z forward)
    cam_pose[:3, :3] = np.stack([right, down, fwd], axis=1)
    cam_pose[:3, 3] = eye

    s = perceive_images(rgb, depth, K, cam_pose, ["red mug", "blue box"])
    mugs = [e for e in s.entities if e.kind == "object" and e.label == "red mug"]
    assert mugs, [e.label for e in s.entities]
    err = min(np.linalg.norm(e.pose[:3] - [0.05, 0.1, 0.45]) for e in mugs)
    assert err < 0.03, err
    assert any(e.kind == "grasp" and e.parent == mugs[0].id for e in s.entities)
    assert s.image is not None and s.image.shape == (H, W, 3)
