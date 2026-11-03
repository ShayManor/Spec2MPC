from __future__ import annotations

import json
import threading

import numpy as np
import pytest
from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from spec2mpc.bench.client import HoldController
from spec2mpc.bench.libero import (
    SUITES,
    LiberoAdapter,
    LiberoBenchEnv,
    ctrl_to_action,
    installed,
    replica_qpos,
    task_names,
)
from spec2mpc.core.types import Program, Scene, Stage, Term
from spec2mpc.orchestrator.agent import RecordedAgent

requires_libero = pytest.mark.skipif(not installed(), reason="envs/libero not installed (run envs/libero/setup.sh)")

_PROG = Program([Stage([Term("pose_to", ["ee", "ee"], {})], Term("timeout", [], {"t_s": 1.0}))])


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


def test_task_names():
    names = task_names()
    assert len(names) == 40 and len(set(names)) == 40
    assert names[0] == "libero_spatial/0" and names[-1] == "libero_10/9"
    assert LiberoAdapter().tasks() == names and set(SUITES) == {s.split("/")[0] for s in names}


def test_ctrl_mapping():
    q = np.linspace(-0.5, 0.5, 7)
    dq_max = np.full(7, 0.05)
    a = ctrl_to_action(np.concatenate([q + 0.01, [255.0]]), q, dq_max)
    assert a.shape == (8,) and np.allclose(a[:7], 0.2) and a[7] == -1.0  # small delta, gripper open
    a = ctrl_to_action(np.concatenate([q + 1.0, [0.0]]), q, dq_max)
    assert np.allclose(a[:7], 1.0) and a[7] == 1.0  # clipped to the controller limit, gripper closed
    assert np.allclose(replica_qpos(q, np.array([0.03, -0.03])), np.concatenate([q, [0.03, 0.03]]))


@pytest.mark.slow
@requires_libero
def test_libero_one_task_runs(stub_executor, tmp_path):
    url, seen = stub_executor
    ad = LiberoAdapter("stub", results_dir=tmp_path, img=128, agent=RecordedAgent(["pick up the bowl"]),
                       perceive=_stub_perceive,
                       make_controller=lambda r: HoldController(r, 0.05, gripper_ctrl=255.0))
    out = ad.run("libero_spatial/0", url, seed=0)
    assert set(out) == {"success", "score", "info"}  # never exports BDDL goal predicates
    assert isinstance(out["success"], bool) and out["score"] == float(out["success"])
    assert out["info"]["n_commands"] == 1 and out["info"]["bench_steps"] == 20  # 1 s timeout at 20 Hz
    assert len(seen) == 1 and seen[0]["command"] == "pick up the bowl"
    assert len(seen[0]["robot"]["actuators"]) == 8
    rows = [json.loads(s) for s in (tmp_path / "libero" / "stub.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["task"].startswith("libero_spatial/0:") and rows[0]["score"] == out["score"]


@pytest.mark.slow
@requires_libero
def test_env_obs_and_hold():
    env = LiberoBenchEnv("libero_spatial/0", img=128)
    try:
        assert env.goal_text and env.max_s == pytest.approx(11.0) and env.control_dt_s == pytest.approx(0.05)
        a, b = env.reset(3), env.reset(3)
        assert np.allclose(a.qpos, b.qpos) and a.qpos.shape == (9,)
        assert a.rgb.shape == (128, 128, 3) and a.rgb.dtype == np.uint8 and a.rgb.std() > 0
        assert a.depth.shape == (128, 128) and 0.1 < float(np.median(a.depth)) < 5.0
        assert a.K.shape == (3, 3) and a.cam_pose.shape == (7,)
        hold = np.concatenate([a.qpos[:7], [255.0]])
        for _ in range(20):
            obs, done = env.step(hold)
        assert np.abs(obs.qpos[:7] - a.qpos[:7]).max() < 0.05  # holding joint targets keeps the arm still
        assert not done and set(env.result()) == {"success", "score", "info"}
    finally:
        env.close()
