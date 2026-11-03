from __future__ import annotations

import json
import threading

import numpy as np
import pytest
from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from spec2mpc.bench.humanoidbench import (
    GOALS,
    TASKS,
    HumanoidBenchAdapter,
    HumanoidBenchEnv,
    installed,
    torque_to_target,
)
from spec2mpc.core.types import Program, Scene, Stage, Term
from spec2mpc.orchestrator.agent import RecordedAgent

requires_humanoidbench = pytest.mark.skipif(
    not installed(), reason="envs/humanoidbench not installed (run envs/humanoidbench/setup.sh)")

_PROG = Program([Stage([Term("velocity_to", ["base"], {"vx": 0.5})], Term("timeout", [], {"t_s": 1.0}))])


@pytest.fixture
def stub_executor():
    # stub executor server: POST /act returns a fixed program and records requests
    app, seen = Flask("stub_executor"), []

    @app.post("/act")
    def act():
        seen.append(request.get_json())
        return jsonify({"program": _PROG.to_json()})

    srv = make_server("127.0.0.1", 0, app)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


def _stub_perceive(obs, robot) -> Scene:
    # stand-in for perceive_images: no entities, the camera image only
    return Scene([], obs.rgb, obs.t_s)


class _ZeroTorque:
    # stub MPC: zero replica torques until the stage timeout
    def __init__(self, robot, dt_s: float = 0.02):
        self.n, self.dt_s, self.status, self.t_s, self.until_s = len(robot.actuators), dt_s, "running", 0.0, 0.0

    def set_program(self, program: Program, scene: Scene) -> None:
        self.status, self.t_s = "running", 0.0
        self.until_s = sum(st.until.params.get("t_s", 0.0) for st in program.stages)

    def step(self, qpos, qvel, scene) -> np.ndarray:
        self.t_s += self.dt_s
        if self.t_s >= self.until_s:
            self.status = "done"
        return np.zeros(self.n)


def _adapter(tmp_path, command: str) -> HumanoidBenchAdapter:
    return HumanoidBenchAdapter("stub", results_dir=tmp_path, render=False, agent=RecordedAgent([command]),
                                perceive=_stub_perceive, make_controller=_ZeroTorque)


def test_task_list_has_goals():
    assert len(TASKS) == 31 and set(GOALS) == set(TASKS)
    assert TASKS["walk"] == "h1hand" and TASKS["highbar_hard"] == "h1strong"
    assert HumanoidBenchAdapter().tasks() == list(TASKS)


def test_torque_to_target():
    q, qd = np.array([0.1, -0.2]), np.array([0.5, 1.0])
    kp, kv = np.array([200.0, 40.0]), np.array([5.0, 2.0])
    tau = np.array([10.0, -3.0])
    tgt = torque_to_target(tau, q, qd, kp, kv)
    assert np.allclose(kp * (tgt - q) - kv * qd, tau)  # the benchmark's PD law reproduces the torque


@requires_humanoidbench
def test_humanoidbench_one_task_runs(stub_executor, tmp_path):
    url, seen = stub_executor
    out = _adapter(tmp_path, "walk forward").run("walk", url, seed=0)
    assert set(out) == {"success", "score", "info"}
    assert isinstance(out["success"], bool) and 0.0 <= out["score"] <= 1.0
    assert out["info"]["n_commands"] == 1 and out["info"]["done"]
    assert 1 <= out["info"]["bench_steps"] <= 50  # 1 s stage at 50 Hz, or earlier if the robot falls
    assert len(seen) == 1 and seen[0]["command"] == "walk forward"
    assert len(seen[0]["robot"]["actuators"]) == 19
    rows = [json.loads(s) for s in (tmp_path / "humanoidbench" / "stub.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["task"] == "walk" and rows[0]["score"] == out["score"]


@requires_humanoidbench
def test_adapter_never_exports_tests(stub_executor, tmp_path):
    url, _ = stub_executor
    out = _adapter(tmp_path, "stand still").run("stand", url, seed=1)
    assert set(out) == {"success", "score", "info"}
    assert not any("bar" in k or "threshold" in k for k in out["info"])


@requires_humanoidbench
def test_env_seeded_and_mapped():
    env = HumanoidBenchEnv("walk")
    try:
        a, b = env.reset(3), env.reset(3)
        assert np.array_equal(a.qpos, b.qpos) and a.qpos.shape == (26,) and a.qvel.shape == (25,)
        assert env.control_dt_s == 0.02 and env.max_s == pytest.approx(20.0)
        obs, done = env.step(np.full(19, 1e4))
        assert np.all(env.target <= env.ctrl_high) and not done
        assert obs.qpos.shape == (26,)
    finally:
        env.close()


@requires_humanoidbench
def test_camera_obs():
    env = HumanoidBenchEnv("walk", render=True, width=128, height=96)
    try:
        o = env.reset(0)
        assert o.rgb.shape == (96, 128, 3) and o.rgb.dtype == np.uint8 and o.depth.shape == (96, 128)
        assert o.K.shape == (3, 3) and o.cam_pose.shape == (7,)
        assert np.dot(o.cam_pose[:3], o.cam_pose[:3]) > 1.0  # task camera sits away from the pelvis
    finally:
        env.close()
