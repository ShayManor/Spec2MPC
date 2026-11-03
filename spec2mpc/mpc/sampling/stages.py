"""Stage bookkeeping: UNTIL conditions on the true MuJoCo state, and the robot facts the costs need."""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import mujoco
import numpy as np

from spec2mpc.core import RobotSpec, Scene, Term
from spec2mpc.mpc.sampling.costs import box_dist, quat_to_mat

GROUND_FAMILIES = frozenset({"mobile_base", "mobile_manipulator", "rover2d", "kart", "quadruped", "humanoid"})
ARM_LIKE = frozenset({"arm", "mobile_manipulator"})
_GRIPPER_WORDS = ("gripper", "finger", "hand", "actuator8")
_FOOT_WORDS = ("foot", "fl", "fr", "rl", "rr", "toe", "ankle")


@dataclass
class Frames:
    # robot frames resolved to MuJoCo bodies, sites, or geoms (Task 3 frame_ref)
    names: list[str]
    kinds: list[str]  # body, site, geom
    ids: list[int]

    def pos(self, d) -> np.ndarray:  # works for mujoco.MjData and mjx.Data; -> [F, 3]
        xp = np if isinstance(d, mujoco.MjData) else jnp
        src = {"body": d.xpos, "site": d.site_xpos, "geom": d.geom_xpos}
        return xp.stack([src[k][i] for k, i in zip(self.kinds, self.ids)])

    def mat(self, d) -> np.ndarray:  # -> [F, 3, 3]
        xp = np if isinstance(d, mujoco.MjData) else jnp
        src = {"body": d.xmat, "site": d.site_xmat, "geom": d.geom_xmat}
        return xp.stack([src[k][i].reshape(3, 3) for k, i in zip(self.kinds, self.ids)])


_OBJ = {"body": mujoco.mjtObj.mjOBJ_BODY, "site": mujoco.mjtObj.mjOBJ_SITE, "geom": mujoco.mjtObj.mjOBJ_GEOM}


def root_body(m: mujoco.MjModel) -> int:
    # the body of the first joint (the floating or mobile base), else the first body
    return int(m.jnt_bodyid[0]) if m.njnt else min(1, m.nbody - 1)


def resolve_frames(m: mujoco.MjModel, robot: RobotSpec) -> Frames:
    # registry frame map first (Task 3); a frame missing there may name a site or body directly
    kinds, ids = [], []
    for n in robot.frames:
        try:
            from spec2mpc.robots.v2 import frame_ref
            kind, ref = frame_ref(robot, n)
        except (ImportError, KeyError):
            kind, ref = next(((k, n) for k in ("site", "body") if mujoco.mj_name2id(m, _OBJ[k], n) >= 0),
                             ("body", n))
        i = mujoco.mj_name2id(m, _OBJ[kind], ref)
        if i < 0:
            raise KeyError(f"frame {n!r} ({kind}:{ref}) is not in the MuJoCo model")
        kinds.append(kind), ids.append(i)
    return Frames(list(robot.frames), kinds, ids)


def gripper_actuators(m: mujoco.MjModel, names: list[str]) -> list[int]:
    out = []
    for i in range(m.nu):
        n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").lower()
        if any(w in n for w in _GRIPPER_WORDS):
            out.append(i)
    return out


def foot_frames(names: list[str]) -> list[int]:
    return [i for i, n in enumerate(names) if any(w in n.lower() for w in _FOOT_WORDS)]


def hold_ctrl(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    # the stop command: position actuators hold their joint, velocity actuators command 0, motors cancel gravity
    # (least squares over the motor columns of the actuation map); d must be forwarded at the current state
    u = np.zeros(m.nu)
    motors = []
    for i in range(m.nu):
        bias = m.actuator_biasprm[i]
        is_pos = m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE and bias[1] < 0
        is_vel = m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE and bias[1] == 0 and bias[2] < 0
        if is_pos and m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            u[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
        elif not is_vel and not is_pos:
            motors.append(i)
    if motors:
        cols = np.zeros((m.nv, len(motors)))
        dd = mujoco.MjData(m)
        dd.qpos[:], dd.qvel[:] = d.qpos, 0.0
        for k, i in enumerate(motors):
            dd.ctrl[:] = 0.0
            dd.ctrl[i] = 1.0
            mujoco.mj_forward(m, dd)
            cols[:, k] = dd.qfrc_actuator
        u[motors] = np.linalg.lstsq(cols, _bias_at_rest(m, d), rcond=None)[0]
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    lim = m.actuator_ctrllimited.astype(bool)
    return np.where(lim, np.clip(u, lo, hi), u)


def _bias_at_rest(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    dd = mujoco.MjData(m)
    dd.qpos[:] = d.qpos
    mujoco.mj_forward(m, dd)
    return dd.qfrc_bias.copy()


class Until:
    # an UNTIL condition evaluated on the true state after each control step
    def __init__(self, term: Term, frames: Frames, gripper: list[int]):
        self.term, self.frames, self.gripper = term, frames, gripper

    def entity_ids(self, scene: Scene) -> list[str]:
        return [a for a in self.term.args if a not in self.frames.names]

    def _pose(self, name: str, d: mujoco.MjData, scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # -> (pos [3], mat [3, 3], extent [3])
        if name in self.frames.names:
            j = self.frames.names.index(name)
            return self.frames.pos(d)[j], self.frames.mat(d)[j], np.zeros(3)
        e = scene.get(name)
        ext = np.zeros(3) if e.extent is None else np.asarray(e.extent, np.float64)
        return np.asarray(e.pose[:3], np.float64), quat_to_mat(e.pose[3:]), ext

    def fired(self, m: mujoco.MjModel, d: mujoco.MjData, scene: Scene, t_stage_s: float) -> bool:
        t, p = self.term.type, self.term.params
        if t == "timeout":
            return t_stage_s >= p.get("t_s", 0.0) - 1e-9
        if t == "gripper_closed":
            dofs = [m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in self.gripper
                    if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT]
            return t_stage_s >= 0.3 and bool(np.all(np.abs(d.qvel[dofs]) < 0.02)) if dofs else t_stage_s >= 0.3
        pa, ra, _ = self._pose(self.term.args[0], d, scene)
        pb, rb, eb = self._pose(self.term.args[1], d, scene)
        if t == "dist_below":
            return float(np.linalg.norm(pa - pb)) < p.get("d", 0.0)
        if t == "contact":
            return float(box_dist(pa, pb, rb, eb)) < 0.01
        if t == "at_pose":
            ok = float(np.linalg.norm(pa - pb)) < p.get("tol_m", 0.02)
            if "tol_rad" in p:
                c = np.clip(0.5 * (np.trace(ra.T @ rb) - 1.0), -1.0, 1.0)
                ok = ok and float(np.arccos(c)) < p["tol_rad"]
            return ok
        raise ValueError(f"unknown UNTIL condition {t!r}")
