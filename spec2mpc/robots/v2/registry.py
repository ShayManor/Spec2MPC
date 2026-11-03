from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import yaml

from spec2mpc.core import Entity, RobotSpec, load_family_config
from spec2mpc.core.io import ROBOT_CFG_DIR
from spec2mpc.robots.v2.describe import actuator_limits, actuator_names
from spec2mpc.robots.v2.randomize import apply_params, sample_ranges

FAMILIES = ("arm", "mobile_manipulator", "mobile_base", "rover2d", "kart", "quadruped", "humanoid", "quadrotor")
LOCK_PATH = ROBOT_CFG_DIR.parent / "menagerie.lock"
MODELS_DIR = Path(__file__).parent / "models"
_OBJ = {"body": mujoco.mjtObj.mjOBJ_BODY, "site": mujoco.mjtObj.mjOBJ_SITE, "geom": mujoco.mjtObj.mjOBJ_GEOM}


def menagerie_path() -> Path:
    # env MENAGERIE_PATH, else the path recorded in configs/v2/menagerie.lock
    env = os.environ.get("MENAGERIE_PATH")
    return Path(env) if env else Path(yaml.safe_load(LOCK_PATH.read_text())["path"])


def _member(family: str, name: str) -> dict:
    for m in load_family_config(family)["members"]:
        if m["name"] == name:
            return m
    raise KeyError(f"{family}/{name}")


def _resolve_mjcf(ref: str) -> str:
    src, _, rel = ref.partition(":")
    if src == "menagerie":
        return str(menagerie_path() / rel)
    if src == "generated":
        return str(MODELS_DIR / rel)
    raise ValueError(f"unknown mjcf source {ref!r}")


def list_robots(family: str) -> list[str]:
    return [m["name"] for m in load_family_config(family)["members"]]


@lru_cache(maxsize=None)
def _nominal_model(mjcf_path: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(mjcf_path)


def build_model(robot: RobotSpec) -> mujoco.MjModel:
    # fresh MjModel with robot.params (mass, friction, damping scales) applied
    return apply_params(copy.copy(_nominal_model(robot.mjcf_path)), robot.params)


def load_robot(family: str, name: str, seed: int | None) -> RobotSpec:
    cfg = load_family_config(family)
    member = _member(family, name)
    mjcf_path = _resolve_mjcf(member["mjcf"])
    nominal = _nominal_model(mjcf_path)
    params, perception, runtime = sample_ranges(cfg["ranges"], seed, f"{family}/{name}")
    params["mass_kg"] = float(nominal.body_mass.sum()) * params["mass_scale"]
    return RobotSpec(family=family, name=name, mjcf_path=mjcf_path, params=params,
                     frames=list(member["frames"]), actuators=actuator_names(nominal),
                     limits=actuator_limits(nominal), floors=dict(cfg["floors"]), runtime=runtime,
                     perception=perception)


def frame_ref(robot: RobotSpec, frame: str) -> tuple[str, str]:
    # canonical frame -> (kind, MJCF name), kind in body, site, geom
    kind, _, ref = _member(robot.family, robot.name)["frames"][frame].partition(":")
    return kind, ref


def frame_pose(model: mujoco.MjModel, data: mujoco.MjData, kind: str, ref: str) -> np.ndarray:
    # [7] xyz + quaternion wxyz, world frame; data must have kinematics computed
    i = mujoco.mj_name2id(model, _OBJ[kind], ref)
    if i < 0:
        raise KeyError(f"{kind}:{ref}")
    if kind == "body":
        return np.concatenate([data.xpos[i], data.xquat[i]])
    xmat = data.site_xmat[i] if kind == "site" else data.geom_xmat[i]
    xpos = data.site_xpos[i] if kind == "site" else data.geom_xpos[i]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, xmat)
    return np.concatenate([xpos, q])


def robot_frames_as_entities(robot: RobotSpec, qpos: np.ndarray) -> list[Entity]:
    # qpos: [nq] -> one robot_frame entity per robot.frames entry, in order
    model = _nominal_model(robot.mjcf_path)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    return [Entity(f, f, "robot_frame", frame_pose(model, data, *frame_ref(robot, f)), None, None, None)
            for f in robot.frames]
