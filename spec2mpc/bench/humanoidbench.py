"""HumanoidBench adapter (held-out evaluation only; never used for training data).

HumanoidBench pins mujoco 3.1.6, so it runs in its own env (envs/humanoidbench) as a
JSON-lines worker subprocess. This module holds only task names, robot ids, and the
goal text given to the orchestrator; success thresholds stay inside the worker.
The MPC models the Menagerie H1 replica (19 torque motors); `HumanoidBenchEnv` maps each
replica torque to the H1 position target whose PD law produces it, and holds the hands still.
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from spec2mpc.bench.client import BenchObs, run_episode
from spec2mpc.core.types import RobotSpec

REPO = Path(__file__).resolve().parent.parent.parent
ENV_DIR = REPO / "envs" / "humanoidbench"
UV = Path.home() / ".local" / "bin" / "uv"

# main benchmark tasks (HumanoidBench README) -> benchmark robot id
TASKS: dict[str, str] = {t: "h1hand" for t in [
    "walk", "reach", "hurdle", "crawl", "maze", "push", "cabinet", "door", "truck", "cube",
    "bookshelf_simple", "bookshelf_hard", "basketball", "window", "spoon", "kitchen", "package",
    "powerlift", "room", "stand", "run", "sit_simple", "sit_hard", "balance_simple",
    "balance_hard", "stair", "slide", "pole", "insert_normal", "insert_small"]}
TASKS["highbar_hard"] = "h1strong"

# goal text for the orchestrator, written by us (HumanoidBench has no instructions)
GOALS: dict[str, str] = {
    "walk": "Walk forward without falling.",
    "run": "Run forward as fast as you can without falling.",
    "stand": "Stand still and upright.",
    "crawl": "Crawl forward through the low tunnel.",
    "hurdle": "Move forward and get over the hurdles without falling.",
    "stair": "Walk up and down the stairs without falling.",
    "slide": "Walk up and down the slopes without falling.",
    "maze": "Walk through the maze to its end.",
    "pole": "Walk forward through the forest of poles without hitting them.",
    "sit_simple": "Sit down on the chair behind you.",
    "sit_hard": "Sit down on the movable chair.",
    "balance_simple": "Keep your balance standing on the board on top of the ball.",
    "balance_hard": "Keep your balance standing on the moving board on top of the ball.",
    "reach": "Touch the target point with your left hand.",
    "push": "Push the box on the table to the target spot.",
    "cabinet": "Open the cabinet doors and drawer and move the objects inside.",
    "highbar_hard": "Grab the high bar and swing up over it.",
    "door": "Open the door and walk through it.",
    "truck": "Unload the packages from the truck onto the table.",
    "cube": "Rotate the cubes in your hands to match the target orientation.",
    "bookshelf_simple": "Move the items to their places on the bookshelf.",
    "bookshelf_hard": "Move the items to their places on the bookshelf.",
    "basketball": "Catch the ball and throw it into the hoop.",
    "window": "Wipe the window with the tool.",
    "spoon": "Stir the pot with the spoon.",
    "kitchen": "Do the kitchen chores: open the microwave, move the kettle, turn the burner and light switches.",
    "package": "Carry the package to the target spot.",
    "powerlift": "Lift the barbell up.",
    "room": "Tidy the room by moving the objects to one side.",
    "insert_normal": "Insert the block ends into the pegs.",
    "insert_small": "Insert the small block ends into the pegs.",
}


def installed(env_dir: Path = ENV_DIR) -> bool:
    # envs/humanoidbench/setup.sh finished: venv has the editable humanoid_bench clone
    return (env_dir / ".venv").exists() and (env_dir / "humanoid-bench" / "humanoid_bench").exists()


def h1_replica() -> RobotSpec:
    from spec2mpc.robots.v2 import load_robot

    return load_robot("humanoid", "h1", None)


def torque_to_target(tau: np.ndarray, q: np.ndarray, qd: np.ndarray, kp: np.ndarray, kv: np.ndarray) -> np.ndarray:
    # [n] torques -> [n] position targets: HumanoidBench PD kp (target - q) - kv qd equals tau
    return q + (tau + kv * qd) / kp


def _arr(r: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(r["b64"]), np.dtype(r.get("dtype", "uint8"))).reshape(r["shape"])


class HumanoidBenchEnv:
    """One HumanoidBench episode behind a worker process; implements spec2mpc.bench.client.BenchEnv.

    step(ctrl): ctrl [19] replica torques (N m) in robot.actuators order. HumanoidBench scores the
    full horizon, so result() first holds the last position targets until the episode ends.
    """

    def __init__(self, task: str, robot: RobotSpec | None = None, render: bool = False,
                 width: int = 256, height: int = 256, env_dir: Path = ENV_DIR) -> None:
        if task not in TASKS:
            raise KeyError(f"unknown HumanoidBench task {task!r}")
        self.name, self.hb_robot = task, TASKS[task]
        self.goal_text = GOALS[task]
        self.robot = robot or h1_replica()
        self.render, self.width, self.height = render, width, height
        self.proc = subprocess.Popen(
            [str(UV), "run", "--project", str(env_dir), "python", str(env_dir / "hb_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            cwd=str(env_dir))
        info = self._call("make", task=task, robot=self.hb_robot, render=render, width=width, height=height)
        self.ctrl_low = np.asarray(info["ctrl_low"])  # [nu_hb] joint position targets, rad
        self.ctrl_high = np.asarray(info["ctrl_high"])  # [nu_hb]
        self.kp, self.kv = np.asarray(info["kp"]), np.asarray(info["kv"])  # [nu_hb]
        self.control_dt_s: float = info["control_dt"]
        self.max_steps: int = info["max_steps"]
        self.max_s = self.max_steps * self.control_dt_s
        # replica actuator k drives HumanoidBench actuator idx[k]; joints share names
        hb_joint = {j: a for a, j in enumerate(info["joints"])}
        self.idx = np.array([hb_joint[a] for a in self.robot.actuators])  # [19]
        self.qadr = np.asarray(info["qposadr"])[self.idx]  # [19]
        self.vadr = np.asarray(info["dofadr"])[self.idx]  # [19]
        self.target: np.ndarray | None = None  # [nu_hb] last position targets
        self.done = False
        self.obs: BenchObs | None = None

    def _call(self, cmd: str, **kw) -> dict:
        self.proc.stdin.write(json.dumps({"cmd": cmd, **kw}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"humanoidbench worker exited during {cmd}")
        out = json.loads(line)
        if not out.pop("ok"):
            raise RuntimeError(f"humanoidbench worker {cmd}: {out['error']}")
        return out

    def _to_obs(self, o: dict) -> BenchObs:
        self._q, self._qd = np.asarray(o["qpos"]), np.asarray(o["qvel"])  # [nq_hb], [nv_hb]
        qpos = np.concatenate([self._q[:7], self._q[self.qadr]])  # [26] free base + replica joints
        qvel = np.concatenate([self._qd[:6], self._qd[self.vadr]])  # [25]
        if "rgb" in o:
            rgb, depth = _arr(o["rgb"]).copy(), _arr(o["depth"]).copy()
            K, cam = np.asarray(o["K"]), np.asarray(o["cam_pose"])
        else:
            rgb, depth, K, cam = np.zeros((self.height, self.width, 3), np.uint8), None, None, None
        self.done = bool(o["done"])
        self.obs = BenchObs(rgb, depth, K, cam, qpos, qvel, float(o["t_s"]))
        return self.obs

    def reset(self, seed: int) -> BenchObs:
        o = self._call("reset", seed=int(seed))
        self.target = np.asarray(o["hold"])
        return self._to_obs(o)

    def step(self, ctrl: np.ndarray) -> tuple[BenchObs, bool]:
        tau = np.asarray(ctrl, dtype=np.float64)  # [19]
        i = self.idx
        self.target = self.target.copy()
        self.target[i] = torque_to_target(tau, self._q[self.qadr], self._qd[self.vadr], self.kp[i], self.kv[i])
        self.target = np.clip(self.target, self.ctrl_low, self.ctrl_high)
        obs = self._to_obs(self._call("step", ctrl=self.target.tolist()))
        return obs, self.done

    def finish(self) -> None:
        while not self.done:
            self._to_obs(self._call("step", ctrl=self.target.tolist()))

    def result(self) -> dict:
        self.finish()
        r = self._call("result")
        return {"success": bool(r["success"]), "score": float(r["score"]), "info": r["info"]}

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._call("close")
            except Exception:
                pass
            self.proc.wait(timeout=30)


EpisodeFn = Callable[..., dict]


class HumanoidBenchAdapter:
    """BenchAdapter for HumanoidBench; `episode_fn` defaults to client.run_episode, extra kwargs go to it."""

    name = "humanoidbench"

    def __init__(self, executor_name: str = "spec2mpc", episode_fn: EpisodeFn | None = None,
                 results_dir: Path | None = None, render: bool = True, **episode_kw) -> None:
        self.executor_name, self.episode_fn = executor_name, episode_fn or run_episode
        self.results_dir, self.render, self.episode_kw = results_dir, render, episode_kw

    def tasks(self) -> list[str]:
        return list(TASKS)

    def run(self, task_name: str, executor_url: str, seed: int) -> dict:
        env = HumanoidBenchEnv(task_name, render=self.render)
        try:
            out = self.episode_fn(env, executor_url, self.executor_name, env.goal_text, bench=self.name,
                                  seed=seed, results_dir=self.results_dir, **self.episode_kw)
        finally:
            env.close()
        return {"success": bool(out["success"]), "score": float(out["score"]), "info": dict(out.get("info", {}))}


__all__ = ["GOALS", "TASKS", "HumanoidBenchAdapter", "HumanoidBenchEnv", "h1_replica", "installed",
           "torque_to_target"]
