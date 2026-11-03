"""Benchmark client: drives a held-out benchmark's own loop through the orchestrator, perception, executor, and MPC.

A benchmark env (RoboLab, LIBERO, HumanoidBench) implements `BenchEnv` in the benchmark's own Python env. The
client wraps it as a SimEnv-like object (`BenchSimEnv`: reset, run_program, run_actions, observe, qpos) and hands it
to the orchestrator's `run_goal`, which calls the executor server over HTTP. Programs run on the MPC of the
benchmark robot's MuJoCo Menagerie replica; the env maps replica controls to the benchmark action interface.
Only the benchmark's own success and score leave this module.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from spec2mpc.core import Episode, Program, RobotSpec, Scene

RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "v2" / "bench"


@dataclass(frozen=True, eq=False)
class BenchObs:
    rgb: np.ndarray  # [H, W, 3] uint8, main benchmark camera
    depth: np.ndarray | None  # [H, W] float32, m along the optical axis
    K: np.ndarray | None  # [3, 3] pixel intrinsics
    cam_pose: np.ndarray | None  # [7] xyz + wxyz, robot-base frame, OpenCV axes (x right, y down, z forward)
    qpos: np.ndarray  # [nq] replica joint order
    qvel: np.ndarray | None  # [nv]
    t_s: float


@runtime_checkable
class BenchEnv(Protocol):
    name: str  # benchmark task name
    robot: RobotSpec  # Menagerie replica the MPC models
    control_dt_s: float
    max_s: float  # benchmark episode horizon

    def reset(self, seed: int) -> BenchObs: ...

    def step(self, ctrl: np.ndarray) -> tuple[BenchObs, bool]: ...  # ctrl [nu] replica controls; True when done

    def result(self) -> dict: ...  # {"success": bool, "score": float, "info": dict}, the benchmark's own verdict


class Controller(Protocol):
    status: str  # "running", "done", "timeout", "invalid_reference", "floor_clip"

    def set_program(self, program: Program, scene: Scene) -> None: ...

    def step(self, qpos: np.ndarray, qvel: np.ndarray, scene: Scene) -> np.ndarray: ...  # [nu]


Perceive = Callable[[BenchObs, RobotSpec], Scene]
MakeController = Callable[[RobotSpec], Controller]


def default_perceive(obs: BenchObs, robot: RobotSpec) -> Scene:
    # image perception (Task 7) plus robot frames from replica kinematics (Task 3)
    from spec2mpc.perception.v2 import perceive_images
    from spec2mpc.robots.v2 import robot_frames_as_entities

    s = perceive_images(obs.rgb, obs.depth, obs.K, obs.cam_pose, None)
    return Scene(s.entities + robot_frames_as_entities(robot, obs.qpos), obs.rgb, obs.t_s)


def default_controller(robot: RobotSpec) -> Controller:
    from spec2mpc.mpc.sampling import make_solver

    return make_solver(robot)


class HoldController:
    """Stub MPC for smoke tests: holds the current joint positions until the stage timeout, then reports done."""

    def __init__(self, robot: RobotSpec, dt_s: float, gripper_ctrl: float | None = None):
        self.robot, self.dt_s, self.gripper_ctrl = robot, dt_s, gripper_ctrl
        self.status, self.t_s, self.until_s, self.hold = "running", 0.0, 0.0, None

    def set_program(self, program: Program, scene: Scene) -> None:
        self.status, self.t_s, self.hold = "running", 0.0, None
        self.until_s = sum(st.until.params.get("t_s", 0.0) for st in program.stages)

    def step(self, qpos: np.ndarray, qvel: np.ndarray, scene: Scene) -> np.ndarray:
        if self.hold is None:
            n = len(self.robot.actuators) or len(qpos)
            self.hold = np.array(qpos[:n], dtype=np.float64)  # [nu]
            if self.gripper_ctrl is not None:
                self.hold[-1] = self.gripper_ctrl
        self.t_s += self.dt_s
        if self.t_s >= self.until_s:
            self.status = "done"
        return self.hold


class BenchSimEnv:
    """SimEnv-like wrapper the orchestrator drives; one per benchmark episode."""

    def __init__(self, bench: BenchEnv, perceive: Perceive, make_controller: MakeController,
                 perceive_every: int = 15):
        self.bench, self.perceive, self.make_controller = bench, perceive, make_controller
        self.perceive_every = perceive_every
        self.obs: BenchObs | None = None
        self.scene: Scene | None = None
        self.done = False
        self.n_steps = 0

    def reset(self, seed: int) -> Scene:
        self.obs, self.done, self.n_steps = self.bench.reset(seed), False, 0
        return self.observe()

    def observe(self) -> Scene:
        self.scene = self.perceive(self.obs, self.bench.robot)
        return self.scene

    def qpos(self) -> np.ndarray:
        return self.obs.qpos

    def _remaining_s(self, max_s: float) -> float:
        return min(max_s, self.bench.max_s - self.n_steps * self.bench.control_dt_s)

    def _step(self, ctrl: np.ndarray, k: int) -> None:
        self.obs, self.done = self.bench.step(ctrl)
        self.n_steps += 1
        if (k + 1) % self.perceive_every == 0 and not self.done:
            self.observe()

    def _episode(self, programs: list[Program], qs: list, us: list, status: str, t0: float) -> Episode:
        trace = {"qpos": np.asarray(qs), "ctrl": np.asarray(us)}  # [T, nq], [T, nu]
        return Episode(self.bench.name, "", programs, trace, status, time.time() - t0)

    def run_program(self, program: Program, max_s: float) -> Episode:
        t0, qs, us = time.time(), [], []
        if self.done:
            return self._episode([program], qs, us, "bench_done", t0)
        mpc = self.make_controller(self.bench.robot)
        mpc.set_program(program, self.scene)
        n = int(round(self._remaining_s(max_s) / self.bench.control_dt_s))
        status = "timeout"
        for k in range(n):
            qvel = self.obs.qvel if self.obs.qvel is not None else np.zeros_like(self.obs.qpos)
            u = mpc.step(self.obs.qpos, qvel, self.scene)  # [nu]
            if mpc.status != "running" and mpc.status != "done":
                status = mpc.status  # stage stopped safely (invalid_reference, timeout, floor_clip)
                break
            qs.append(self.obs.qpos)
            us.append(u)
            self._step(u, k)
            if mpc.status == "done" or self.done:
                status = "done" if mpc.status == "done" else "bench_done"
                break
        return self._episode([program], qs, us, status, t0)

    def run_actions(self, policy: Callable[[Scene], np.ndarray], max_s: float) -> Episode:
        t0, qs, us = time.time(), [], []
        if self.done:
            return self._episode([], qs, us, "bench_done", t0)
        n = int(round(self._remaining_s(max_s) / self.bench.control_dt_s))
        for k in range(n):
            u = np.asarray(policy(self.scene))  # [nu]
            qs.append(self.obs.qpos)
            us.append(u)
            self._step(u, k)
            if self.done:
                return self._episode([], qs, us, "bench_done", t0)
        return self._episode([], qs, us, "timeout", t0)


def append_result(bench: str, executor_name: str, row: dict, results_dir: Path | None = None) -> Path:
    path = Path(results_dir or RESULTS_DIR) / bench / f"{executor_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return path


def run_episode(bench_env: BenchEnv, executor_url: str, executor_name: str, goal_text: str, *,
                bench: str = "bench", seed: int = 0, perceive: Perceive | None = None,
                make_controller: MakeController | None = None, agent: Any = None, max_commands: int = 12,
                max_command_s: float = 60.0, perceive_every: int = 15, post: Any = None,
                results_dir: Path | None = None) -> dict:
    """One benchmark episode for one goal; returns {"success", "score", "info"} and appends it to
    data/v2/bench/<bench>/<executor>.jsonl. `agent` defaults to the orchestrator's OpenAIAgent."""
    from spec2mpc.orchestrator import run_goal

    t0 = time.time()
    env = BenchSimEnv(bench_env, perceive or default_perceive, make_controller or default_controller,
                      perceive_every)
    scene = env.reset(seed)
    episodes = run_goal(goal_text, env, executor_url, executor_name, max_commands, robot=bench_env.robot,
                        scene=scene, agent=agent, max_s=max_command_s, task_id=bench_env.name, post=post)
    res = bench_env.result()
    info = {**res.get("info", {}), "n_commands": len(episodes), "statuses": [e.status for e in episodes],
            "bench_steps": env.n_steps, "wall_s": time.time() - t0}
    out = {"success": bool(res["success"]), "score": float(res["score"]), "info": info}
    append_result(bench, executor_name, {"task": bench_env.name, "seed": seed, "executor": executor_name, **out},
                  results_dir)
    return out
