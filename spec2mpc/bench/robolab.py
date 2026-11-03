"""RoboLab-120 adapter (Isaac Sim 5.0 + Isaac Lab 2.2.0, DROID Franka with joint-position actions).

Runs only inside envs/robolab (the Isaac Lab container); every Isaac import is deferred until the app is launched.
RoboLab tasks, layouts, instructions, and success checks are used for evaluation only. The MPC models the robot as
the Menagerie Panda replica (8 controls: 7 joint-position targets, gripper 0..255 with 255 open); `to_droid_action`
maps that to the DROID action (7 joint positions, gripper 0 open / 1 closed, binarized by RoboLab at 0.5).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from spec2mpc.bench.client import BenchObs, run_episode
from spec2mpc.core import RobotSpec

BENCH = "robolab"
CAMERA = "over_shoulder_left_camera"
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
GRIPPER_OPEN_CTRL = 255.0  # Menagerie panda actuator8
FINGER_OPEN_M = 0.04  # Menagerie panda finger joint at open


def to_droid_action(ctrl: np.ndarray) -> np.ndarray:
    # [8] replica controls -> [8] DROID joint-position action
    ctrl = np.asarray(ctrl, dtype=np.float64)
    close = 1.0 - np.clip(ctrl[7] / GRIPPER_OPEN_CTRL, 0.0, 1.0)
    return np.concatenate([ctrl[:7], [close]]).astype(np.float32)


def replica_qpos(arm_joint_pos: np.ndarray, gripper_pos: np.ndarray) -> np.ndarray:
    # DROID proprio ([7] joints, [1] gripper 0 open .. 1 closed) -> [9] Menagerie panda qpos
    finger = FINGER_OPEN_M * (1.0 - float(np.clip(np.asarray(gripper_pos).reshape(-1)[0], 0.0, 1.0)))
    return np.concatenate([np.asarray(arm_joint_pos, dtype=np.float64).reshape(7), [finger, finger]])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # wxyz Hamilton product
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_mul(quat_mul(q, np.r_[0.0, v]), q * np.array([1, -1, -1, -1]))[1:]


def cam_pose_in_base(cam_pos: np.ndarray, cam_quat_gl: np.ndarray, base_pos: np.ndarray,
                     base_quat: np.ndarray) -> np.ndarray:
    # camera pose (env-local position, world wxyz OpenGL axes) -> [7] robot-base frame, OpenCV axes
    q_cv = quat_mul(np.asarray(cam_quat_gl, np.float64), np.array([0.0, 1.0, 0.0, 0.0]))  # flip y and z
    inv = np.asarray(base_quat, np.float64) * np.array([1, -1, -1, -1])
    p = quat_rotate(inv, np.asarray(cam_pos, np.float64) - np.asarray(base_pos, np.float64))
    q = quat_mul(inv, q_cv)
    return np.concatenate([p, q / np.linalg.norm(q)])


def register(task: list[str] | None = None) -> None:
    """Register RoboLab-120 DROID joint-position envs with depth on the over-shoulder camera.
    Call after AppLauncher has launched."""
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.robots.droid import WristCameraCfg
    from robolab.variations.camera import OverShoulderLeftCameraCfg, with_depth

    auto_register_droid_envs(task=task, cameras=[with_depth(OverShoulderLeftCameraCfg), WristCameraCfg],
                             lazy_sensor_update=False)


def _np(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class RoboLabEnv:
    """BenchEnv over one registered RoboLab env (num_envs=1)."""

    def __init__(self, task_env: str, robot: RobotSpec, *, device: str = "cuda:0", seed: int = 0,
                 camera: str = CAMERA):
        from robolab.core.environments.runtime import create_env
        from robolab.core.task.instruction import primary_instruction

        self.env, self.env_cfg = create_env(task_env, device=device, seed=seed, num_envs=1)
        self.name, self.robot, self.camera = task_env, robot, camera
        self.control_dt_s = float(self.env_cfg.sim.dt * self.env_cfg.decimation)
        self.max_s = float(self.env.max_episode_length * self.control_dt_s)
        self.instruction = primary_instruction(self.env_cfg.instruction)[1]  # evaluation only
        self._run, self._steps, self._last_info = -1, 0, None

    def _obs(self, obs: dict) -> BenchObs:
        from robolab.core.utils.isaaclab_compat import quat_isaaclab_to_wxyz

        img = obs["image_obs"]
        robot = self.env.scene["robot"]
        base_pos = _np(robot.data.root_pos_w[0] - self.env.scene.env_origins[0])
        base_quat = _np(quat_isaaclab_to_wxyz(robot.data.root_quat_w)[0])
        idx = [robot.data.joint_names.index(j) for j in ARM_JOINTS]
        qvel = np.concatenate([_np(robot.data.joint_vel[0])[idx], [0.0, 0.0]])  # [9]
        depth = img.get(f"{self.camera}_depth")
        return BenchObs(
            rgb=_np(img[self.camera][0]).astype(np.uint8),
            depth=None if depth is None else _np(depth[0])[..., 0].astype(np.float32),
            K=None if depth is None else _np(img[f"{self.camera}_K"][0]),
            cam_pose=None if depth is None else cam_pose_in_base(
                _np(img[f"{self.camera}_pos"][0]), _np(img[f"{self.camera}_quat"][0]), base_pos, base_quat),
            qpos=replica_qpos(_np(obs["proprio_obs"]["arm_joint_pos"][0]),
                              _np(obs["proprio_obs"]["gripper_pos"][0])),
            qvel=qvel,
            t_s=self._steps * self.control_dt_s,
        )

    def reset(self, seed: int) -> BenchObs:
        from robolab.core.environments.runtime import end_episode

        if self._run >= 0:
            end_episode(self.env)
            self.env.reset_eval_state()
        self._run += 1
        rm = self.env.recorder_manager
        if rm is not None and hasattr(rm, "set_hdf5_file"):
            rm.set_hdf5_file(f"run_{self._run}.hdf5")
            rm.set_episode_index(0, env_ids=[0])
        self.env.reset(seed=seed)
        obs, _ = self.env.reset()  # RoboLab resets twice so cameras render the settled scene
        self._steps, self._last_info = 0, None
        return self._obs(obs)

    def step(self, ctrl: np.ndarray) -> tuple[BenchObs, bool]:
        import torch
        from robolab.core.logging.results import get_all_env_subtask_infos

        a = torch.tensor(to_droid_action(ctrl), device=self.env.device)[None]  # [1, 8]
        obs, *_ = self.env.step(a)
        self._steps += 1
        infos = get_all_env_subtask_infos(self.env)
        if infos:
            self._last_info = infos[0]
        done = bool(self.env.all_terminated) or self._steps >= self.env.max_episode_length
        return self._obs(obs), done

    def result(self) -> dict:
        # RoboLab's verdict: success from its termination predicates, score from its subtask tracker
        r = self.env.get_env_results()[0]
        success = bool(r["success"])
        score = 1.0 if success else float((self._last_info or {}).get("score") or 0.0)
        return {"success": success, "score": score, "info": {"bench_term_step": r["step"]}}

    def close(self) -> None:
        self.env.close()


class RoboLabAdapter:
    """BenchAdapter for RoboLab-120; one env per task, kept open across seeds."""

    name = BENCH

    def __init__(self, executor_name: str, robot: RobotSpec, *, device: str = "cuda:0", **episode_kw: Any):
        self.executor_name, self.robot, self.device, self.episode_kw = executor_name, robot, device, episode_kw
        self._envs: dict[str, RoboLabEnv] = {}

    def tasks(self) -> list[str]:
        from robolab.core.environments.factory import get_envs

        return list(get_envs())

    def env(self, task_name: str) -> RoboLabEnv:
        if task_name not in self._envs:
            for e in self._envs.values():  # one Isaac env at a time
                e.close()
            self._envs = {task_name: RoboLabEnv(task_name, self.robot, device=self.device)}
        return self._envs[task_name]

    def run(self, task_name: str, executor_url: str, seed: int) -> dict:
        env = self.env(task_name)
        return run_episode(env, executor_url, self.executor_name, env.instruction, bench=BENCH, seed=seed,
                           **self.episode_kw)

