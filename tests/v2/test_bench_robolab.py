import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from spec2mpc.bench import BenchObs, BenchSimEnv, HoldController, run_episode
from spec2mpc.bench.robolab import (cam_pose_in_base, quat_mul, quat_rotate, replica_qpos, to_droid_action)
from spec2mpc.core import Program, RobotSpec, Scene, Stage, Term
from spec2mpc.orchestrator import RecordedAgent

ROOT = Path(__file__).parent.parent.parent
ENV_DIR = ROOT / "envs" / "robolab"
PANDA = RobotSpec("arm", "panda", "panda.xml", {}, ["ee"], [f"actuator{i}" for i in range(1, 9)], {}, {}, {}, {})
SECRET = "hidden_predicate_threshold_0.123"


def _stub_app():
    spec = importlib.util.spec_from_file_location("stub_executor", ENV_DIR / "stub_executor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.create_app()


@pytest.fixture(scope="module")
def server_url():
    from werkzeug.serving import make_server

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = make_server("127.0.0.1", port, _stub_app(), threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


class FakeBench:
    """BenchEnv with a first-order joint plant; succeeds when the gripper control stays open for `need` steps."""

    def __init__(self, need: int = 10, horizon: int = 200):
        self.name, self.robot, self.control_dt_s = "FakeTask", PANDA, 1 / 15
        self.max_s, self.need, self.horizon = horizon / 15, need, horizon
        self.secret = SECRET  # stands in for the benchmark's success test; must never leave the adapter

    def _obs(self) -> BenchObs:
        return BenchObs(np.zeros((4, 6, 3), np.uint8), None, None, None, self.q.copy(), np.zeros(9),
                        self.k * self.control_dt_s)

    def reset(self, seed: int) -> BenchObs:
        self.q, self.k, self.open_steps, self.ctrls = np.full(9, 0.1), 0, 0, []
        return self._obs()

    def step(self, ctrl):
        a = to_droid_action(ctrl)
        self.ctrls.append(a)
        self.q[:7] += 0.5 * (a[:7] - self.q[:7])
        self.open_steps += int(a[7] < 0.5)
        self.k += 1
        return self._obs(), self.open_steps >= self.need or self.k >= self.horizon

    def result(self) -> dict:
        ok = self.open_steps >= self.need
        return {"success": ok, "score": 1.0 if ok else self.open_steps / self.need, "info": {"bench_term_step": self.k}}


def _perceive(obs, robot):
    return Scene([], obs.rgb, obs.t_s)


def _hold(robot):
    return HoldController(robot, 1 / 15, 255.0)


def test_droid_action_mapping():
    q = np.arange(7) * 0.1
    np.testing.assert_allclose(to_droid_action(np.r_[q, 255.0]), np.r_[q, 0.0], atol=1e-6)  # open
    np.testing.assert_allclose(to_droid_action(np.r_[q, 0.0])[7], 1.0)  # closed
    assert to_droid_action(np.r_[q, 255.0]).shape == (8,)


def test_replica_qpos():
    q = replica_qpos(np.arange(7) * 0.1, np.array([0.0]))
    assert q.shape == (9,) and q[7] == q[8] == pytest.approx(0.04)
    assert replica_qpos(np.zeros(7), np.array([1.0]))[7] == pytest.approx(0.0)


def test_cam_pose_opengl_to_opencv_in_base():
    # OpenGL camera with identity world orientation looks along -z; in OpenCV axes its z (forward) is world -z
    base_pos, base_quat = np.array([1.0, 0.0, 0.0]), np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    p = cam_pose_in_base(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0, 0, 0]), base_pos, base_quat)
    np.testing.assert_allclose(p[:3], [2.0, 0.0, 3.0], atol=1e-9)  # base yawed +90 deg
    fwd = quat_rotate(quat_mul(base_quat, p[3:]), np.array([0, 0, 1.0]))  # back to world
    np.testing.assert_allclose(fwd, [0, 0, -1.0], atol=1e-9)
    assert np.linalg.norm(p[3:]) == pytest.approx(1.0)


def test_one_episode_through_server(server_url, tmp_path):
    bench = FakeBench(need=10)
    out = run_episode(bench, server_url, "stub", "put the banana in the bowl", bench="fake", perceive=_perceive,
                      make_controller=_hold, agent=RecordedAgent(["put the banana in the bowl"]),
                      results_dir=tmp_path)
    assert out["success"] is True and out["score"] == 1.0
    assert out["info"]["n_commands"] == 1 and out["info"]["statuses"] == ["bench_done"]
    np.testing.assert_allclose(bench.ctrls[0][:7], 0.1)  # hold program kept the arm at its start pose
    rows = (tmp_path / "fake" / "stub.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["success"] is True


def test_program_ends_then_next_command(server_url, tmp_path):
    # the hold program ends at its 3 s timeout (45 steps); the agent then issues a second command
    bench = FakeBench(need=60)
    out = run_episode(bench, server_url, "stub", "goal", bench="fake", perceive=_perceive, make_controller=_hold,
                      agent=RecordedAgent(["a", "b"]), results_dir=tmp_path)
    assert out["info"]["statuses"] == ["done", "bench_done"] and out["success"]


def test_bench_done_stops_later_commands(server_url, tmp_path):
    bench = FakeBench(need=5)
    out = run_episode(bench, server_url, "stub", "goal", bench="fake", perceive=_perceive, make_controller=_hold,
                      agent=RecordedAgent(["a", "b", "c"]), results_dir=tmp_path)
    assert out["info"]["statuses"] == ["bench_done", "bench_done", "bench_done"]
    assert out["info"]["bench_steps"] == 5


def test_executor_error_reported(tmp_path):
    def post(url, payload):
        return 500, {"error": "invalid_program"}

    bench = FakeBench()
    out = run_episode(bench, "http://unused", "stub", "goal", bench="fake", perceive=_perceive, make_controller=_hold,
                      agent=RecordedAgent(["a"]), post=post, results_dir=tmp_path)
    assert out["info"]["statuses"] == ["executor_error:invalid_program"] and out["success"] is False


def test_invalid_reference_stops_without_stepping():
    class Dropped(HoldController):
        def step(self, qpos, qvel, scene):
            self.status = "invalid_reference"
            return np.zeros(8)

    env = BenchSimEnv(FakeBench(), _perceive, lambda r: Dropped(r, 1 / 15))
    env.reset(0)
    ep = env.run_program(Program([Stage([], Term("timeout", [], {"t_s": 1.0}))]), 5.0)
    assert ep.status == "invalid_reference" and env.n_steps == 0


def test_adapters_never_export_tests(server_url, tmp_path):
    out = run_episode(FakeBench(), server_url, "stub", "goal", bench="fake", perceive=_perceive,
                      make_controller=_hold, agent=RecordedAgent(["a"]), results_dir=tmp_path)
    assert set(out) == {"success", "score", "info"}
    assert SECRET not in json.dumps(out) and SECRET not in (tmp_path / "fake" / "stub.jsonl").read_text()


def test_adapter_module_has_no_isaac_imports_at_top():
    # spec2mpc.bench.robolab must import without Isaac Sim; Isaac imports are deferred
    assert "isaaclab" not in sys.modules and "robolab" not in sys.modules


HAS_ISAAC = importlib.util.find_spec("isaaclab") is not None


@pytest.mark.slow
@pytest.mark.requires_robolab
@pytest.mark.skipif(not HAS_ISAAC, reason="needs the Isaac Lab 2.2.0 container on an RTX 48 GB GPU")
def test_robolab_one_task_runs(server_url, tmp_path):
    r = subprocess.run([sys.executable, str(ENV_DIR / "smoke.py"), "--task", "BananaInBowlTask", "--headless",
                        "--executor-url", server_url, "--results-dir", str(tmp_path)],
                       capture_output=True, text=True, timeout=3600, env=os.environ)
    line = [s for s in r.stdout.splitlines() if s.startswith("RESULT ")]
    assert r.returncode == 0 and line, r.stdout[-3000:] + r.stderr[-3000:]
    out = json.loads(line[0][len("RESULT "):])
    assert isinstance(out["success"], bool) and 0.0 <= out["score"] <= 1.0
