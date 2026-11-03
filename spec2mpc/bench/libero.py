"""LIBERO adapter (held-out evaluation only; never used for training data).

LIBERO pins robosuite 1.4.0, so the benchmark runs in its own env (envs/libero) as a JSON-lines worker
subprocess. `LiberoBenchEnv` is the client's BenchEnv: the MPC models the Menagerie Panda replica
(7 arm position actuators + tendon gripper 0..255) and this module maps replica controls to LIBERO's
Panda under robosuite's JOINT_POSITION controller (delta joint targets + gripper -1 open, +1 close).
Only the benchmark's success flag (score = success) leaves the worker.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from spec2mpc.bench.client import BenchObs, run_episode
from spec2mpc.core.types import RobotSpec

REPO = Path(__file__).resolve().parent.parent.parent
ENV_DIR = REPO / "envs" / "libero"


def installed(env_dir: Path = ENV_DIR) -> bool:
    # envs/libero/setup.sh finished: venv has robosuite and LIBERO has its config
    return (any((env_dir / ".venv" / "lib").glob("python3.*/site-packages/robosuite"))
            and (env_dir / "config" / "config.yaml").exists())


# standard LIBERO eval suites: tasks per suite, episode horizon in control steps at 20 Hz (openpi LIBERO eval)
SUITES: dict[str, tuple[int, int]] = {
    "libero_spatial": (10, 220),
    "libero_object": (10, 280),
    "libero_goal": (10, 300),
    "libero_10": (10, 520),
}
N_ARM = 7
GRIP_CTRL_MAX = 255.0  # Menagerie panda actuator8: 255 open, 0 closed


def task_names() -> list[str]:
    return [f"{s}/{i}" for s, (n, _) in SUITES.items() for i in range(n)]


def ctrl_to_action(ctrl: np.ndarray, joint_pos: np.ndarray, dq_max: np.ndarray) -> np.ndarray:
    # replica ctrl [8] (joint targets rad, gripper 0..255) -> robosuite action [8] in [-1, 1]
    dq = np.clip(ctrl[:N_ARM] - joint_pos, -dq_max, dq_max)  # [7] rad this step
    grip = 1.0 - 2.0 * np.clip(ctrl[N_ARM] / GRIP_CTRL_MAX, 0.0, 1.0)  # 255 -> -1 open, 0 -> +1 close
    return np.concatenate([dq / dq_max, [grip]])


def replica_qpos(joint_pos: np.ndarray, gripper_qpos: np.ndarray) -> np.ndarray:
    # robosuite finger joints are (+w, -w); Menagerie finger_joint1/2 are both +w
    return np.concatenate([joint_pos, [gripper_qpos[0], -gripper_qpos[1]]])  # [9]


def _mat_to_pose(T: np.ndarray) -> np.ndarray:
    # [4, 4] -> [7] xyz + wxyz
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(T[:3, :3]).ravel())
    return np.concatenate([T[:3, 3], q])


def _arr(d: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=d["dtype"]).reshape(d["shape"])


def panda_replica() -> RobotSpec:
    from spec2mpc.robots.v2 import load_robot

    return load_robot("arm", "panda", None)


class LiberoBenchEnv:
    """One LIBERO task behind a worker process; implements spec2mpc.bench.client.BenchEnv."""

    def __init__(self, task: str, robot: RobotSpec | None = None, img: int = 256, env_dir: Path = ENV_DIR):
        suite, idx = task.split("/")
        self.suite, self.index = suite, int(idx)
        self.robot = robot or panda_replica()
        env = {**os.environ, "PYTHONPATH": str(REPO), "LIBERO_CONFIG_PATH": str(env_dir / "config"),
               "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
        env["PYOPENGL_PLATFORM"] = env["MUJOCO_GL"]
        self.proc = subprocess.Popen(
            [str(env_dir / ".venv" / "bin" / "python"), str(env_dir / "libero_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True)
        meta = self._call("open", suite=suite, index=self.index, img=img)
        self.name = f"{task}:{meta['name']}"
        self.goal_text: str = meta["language"]  # benchmark instruction, goal for the orchestrator
        self.dq_max = np.asarray(meta["dq_max"])  # [7] rad per control step
        self.control_dt_s = 1.0 / meta["control_freq"]
        self.max_s = SUITES[suite][1] * self.control_dt_s
        self.obs: BenchObs | None = None
        self._joint_pos = np.zeros(N_ARM)

    def _call(self, cmd: str, **kw) -> dict:
        self.proc.stdin.write(json.dumps({"cmd": cmd, **kw}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"libero worker exited (code {self.proc.poll()})")
        rep = json.loads(line)
        if "error" in rep:
            raise RuntimeError(f"libero worker {cmd}: {rep['error']}")
        return rep

    def _to_obs(self, r: dict) -> BenchObs:
        self._joint_pos = np.asarray(r["joint_pos"])
        qvel = np.concatenate([r["joint_vel"], [r["gripper_qvel"][0], -r["gripper_qvel"][1]]])  # [9]
        self.obs = BenchObs(_arr(r["rgb"]).copy(), _arr(r["depth"]).copy(), np.asarray(r["K"]),
                            _mat_to_pose(np.asarray(r["T_bc"])),
                            replica_qpos(self._joint_pos, np.asarray(r["gripper_qpos"])), qvel,
                            self._t_s)
        return self.obs

    def reset(self, seed: int) -> BenchObs:
        self._t_s = 0.0
        return self._to_obs(self._call("reset", seed=seed))

    def step(self, ctrl: np.ndarray) -> tuple[BenchObs, bool]:
        a = ctrl_to_action(np.asarray(ctrl, dtype=float), self._joint_pos, self.dq_max)  # [8]
        r = self._call("step", action=a.tolist())
        self._t_s += self.control_dt_s
        return self._to_obs(r), bool(r["done"])

    def result(self) -> dict:
        r = self._call("result")
        return {"success": bool(r["success"]), "score": float(r["success"]), "info": {"steps": r["steps"]}}

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._call("close")
            except RuntimeError:
                pass
            self.proc.wait(timeout=30)


EpisodeFn = Callable[..., dict]


class LiberoAdapter:
    """BenchAdapter for LIBERO; `episode_fn` defaults to client.run_episode, extra kwargs go to it."""

    name = "libero"

    def __init__(self, executor_name: str = "spec2mpc", episode_fn: EpisodeFn | None = None,
                 results_dir: Path | None = None, img: int = 256, **episode_kw):
        self.executor_name, self.episode_fn = executor_name, episode_fn or run_episode
        self.results_dir, self.img, self.episode_kw = results_dir, img, episode_kw

    def tasks(self) -> list[str]:
        return task_names()

    def run(self, task_name: str, executor_url: str, seed: int) -> dict:
        env = LiberoBenchEnv(task_name, img=self.img)
        try:
            out = self.episode_fn(env, executor_url, self.executor_name, env.goal_text, bench=self.name,
                                  seed=seed, results_dir=self.results_dir, **self.episode_kw)
        finally:
            env.close()
        return {"success": bool(out["success"]), "score": float(out["score"]), "info": dict(out.get("info", {}))}


__all__ = ["LiberoAdapter", "LiberoBenchEnv", "SUITES", "installed", "ctrl_to_action", "replica_qpos",
           "task_names"]
