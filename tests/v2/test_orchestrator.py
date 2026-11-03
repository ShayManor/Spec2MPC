import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from flask import Flask, jsonify, request

from spec2mpc.core import Entity, Episode, ExecutorError, Program, RobotSpec, Scene, Stage, Term, from_jsonable
from spec2mpc.orchestrator import (
    HTTPExecutor,
    OpenAIAgent,
    RecordedAgent,
    feedback_text,
    run_command,
    run_goal,
)


def _robot() -> RobotSpec:
    return RobotSpec(
        family="arm", name="panda", mjcf_path="assets/panda.xml", params={"mass_kg": 18.0},
        frames=["ee", "base"], actuators=["j1", "j2"], limits={"j1_vel_radps": 2.175},
        floors={"min_clearance_m": 0.02, "max_speed_frac": 0.5, "max_force_frac": 0.5},
        runtime={"control_rate_hz": 50.0}, perception={"loc_noise_std_m": 0.005},
    )


def _scene() -> Scene:
    mug = Entity("mug_0", "mug", "object", np.array([0.5, 0.0, 0.1, 1, 0, 0, 0.0]),
                 np.array([0.08, 0.08, 0.1]), None, None)
    return Scene([mug], np.zeros((8, 8, 3), np.uint8), 0.0)


def _program() -> Program:
    return Program([Stage([Term("pose_to", ["ee", "mug_0"], {"w": 1.0})], Term("timeout", [], {"t_s": 2.0}))])


class StubEnv:
    # runs nothing; reports "done" for every program or action policy
    def __init__(self):
        self.programs: list[Program] = []
        self.n_actions = 0

    def observe(self) -> Scene:
        return _scene()

    def run_program(self, program: Program, max_s: float) -> Episode:
        self.programs.append(program)
        return Episode("t0", "", [program], {}, "done", 0.0)

    def run_actions(self, policy, max_s: float) -> Episode:
        a = policy(_scene())  # [act_dim]
        assert a.shape == (2,)
        self.n_actions += 1
        return Episode("t0", "", [], {}, "done", 0.0)


class StubProgramExecutor:
    def __init__(self, name: str, fail: int = 0):
        self.name = name
        self.fail = fail
        self.calls: list[tuple[str, str | None]] = []

    def act(self, command, robot, scene, feedback):
        self.calls.append((command, feedback))
        if self.fail > 0:
            self.fail -= 1
            raise ExecutorError("invalid_program", "truncated")
        return _program()


class StubChunks:
    def next_chunk(self, scene, qpos):
        return np.zeros((4, 2))  # [T, act_dim]


class StubVLA(StubProgramExecutor):
    def act(self, command, robot, scene, feedback):
        self.calls.append((command, feedback))
        return StubChunks()


TRANSCRIPT = ["pick up the mug", "put the mug on the shelf", "DONE"]


def test_same_commands_for_all_executors():
    execs = [StubProgramExecutor("spec2mpc"), StubVLA("vla"), StubProgramExecutor("llm_program")]
    for ex in execs:
        eps = run_goal("tidy the table", StubEnv(), "", ex.name, robot=_robot(), scene=_scene(),
                       agent=RecordedAgent(TRANSCRIPT), executor=ex)
        assert len(eps) == 2 and all(e.status == "done" and e.executor == ex.name for e in eps)
    seqs = [json.dumps(ex.calls).encode() for ex in execs]
    assert seqs[0] == seqs[1] == seqs[2]
    assert [c for c, _ in execs[0].calls] == TRANSCRIPT[:2]


def test_retry_once_on_executor_error():
    # Review Focus 3: one retry, then a reported failure that the agent sees
    agent = RecordedAgent(["pick up the mug", "DONE"])
    ok = StubProgramExecutor("s", fail=1)
    env = StubEnv()
    eps = run_goal("g", env, "", "s", robot=_robot(), scene=_scene(), agent=agent, executor=ok)
    assert [e.status for e in eps] == ["done"] and len(ok.calls) == 2
    assert ok.calls[1][1] == feedback_text("pick up the mug", "executor_error:invalid_program")

    agent = RecordedAgent(["pick up the mug", "DONE"])
    bad = StubProgramExecutor("s", fail=5)
    env = StubEnv()
    eps = run_goal("g", env, "", "s", robot=_robot(), scene=_scene(), agent=agent, executor=bad)
    assert len(bad.calls) == 2 and env.programs == []
    assert [e.status for e in eps] == ["executor_error:invalid_program"]
    assert agent.histories[-1] == [("pick up the mug", "executor_error:invalid_program")]


def test_stops_at_max_commands():
    ex = StubProgramExecutor("s")
    agent = RecordedAgent([f"cmd {i}" for i in range(50)])
    eps = run_goal("g", StubEnv(), "", "s", robot=_robot(), scene=_scene(), agent=agent, executor=ex,
                   max_commands=3)
    assert len(eps) == 3 and len(ex.calls) == 3 and len(agent.histories) == 3


def test_run_command_bypasses_agent():
    ex = StubProgramExecutor("s")
    env = StubEnv()
    ep = run_command("go to the dock", env, "", "s", robot=_robot(), scene=_scene(), executor=ex)
    assert ep.status == "done" and ex.calls == [("go to the dock", None)] and env.programs == [_program()]


def _stub_server(fail: int) -> Flask:
    app = Flask("stub")
    state = {"fail": fail, "seen": []}
    app.config["state"] = state

    @app.post("/act")
    def act():
        body = request.get_json()
        robot, scene = from_jsonable(body["robot"]), from_jsonable(body["scene"])
        assert isinstance(robot, RobotSpec) and isinstance(scene, Scene)
        state["seen"].append(body)
        if state["fail"] > 0:
            state["fail"] -= 1
            return jsonify({"error": "invalid_program"}), 422
        return jsonify({"program": _program().to_json()})

    return app


def _post_via(app: Flask):
    client = app.test_client()

    def post(url: str, payload: dict) -> tuple[int, dict]:
        r = client.post("/" + url.rsplit("/", 1)[-1], json=payload)
        return r.status_code, r.get_json()

    return post


def test_http_executor_roundtrip_and_error():
    app = _stub_server(fail=1)
    ex = HTTPExecutor("http://x:8000", "spec2mpc", "ep1", post=_post_via(app))
    with pytest.raises(ExecutorError) as e:
        ex.act("pick", _robot(), _scene(), None)
    assert e.value.code == "invalid_program"
    assert ex.act("pick", _robot(), _scene(), "fb") == _program()
    seen = app.config["state"]["seen"]
    assert [s["executor"] for s in seen] == ["spec2mpc"] * 2 and seen[1]["episode_id"] == "ep1"
    assert seen[1]["feedback"] == "fb"


def test_run_goal_over_http_uses_one_episode_id():
    app = _stub_server(fail=0)
    eps = run_goal("g", StubEnv(), "http://x:8000", "spec2mpc", robot=_robot(), scene=_scene(),
                   agent=RecordedAgent(TRANSCRIPT), post=_post_via(app))
    seen = app.config["state"]["seen"]
    assert len(eps) == 2 and len({s["episode_id"] for s in seen}) == 1


class FakeResponses:
    def __init__(self, outputs: list[dict]):
        self.outputs = outputs
        self.requests: list[dict] = []

    def create(self, **kw):
        self.requests.append(kw)
        return SimpleNamespace(status="completed", output_text=json.dumps(self.outputs.pop(0)))


def test_openai_agent_request_and_parse():
    fake = FakeResponses([{"done": False, "command": "pick up the mug"}, {"done": True, "command": ""}])
    agent = OpenAIAgent(client=SimpleNamespace(responses=fake))
    assert agent.next("tidy", _scene().image, []) == "pick up the mug"
    assert agent.next("tidy", None, [("pick up the mug", "done")]) == "DONE"
    r0, r1 = fake.requests
    assert r0["model"] == "gpt-6.1-sol" and r0["reasoning"] == {"effort": "high"}
    assert r0["text"]["format"]["strict"] is True
    kinds0 = [c["type"] for c in r0["input"][0]["content"]]
    kinds1 = [c["type"] for c in r1["input"][0]["content"]]
    assert kinds0 == ["input_text", "input_image"] and kinds1 == ["input_text"]
    assert r0["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert "pick up the mug" in r1["input"][0]["content"][0]["text"] and "done" in r1["input"][0]["content"][0]["text"]


@pytest.mark.slow
def test_openai_agent_live_once():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    cmd = OpenAIAgent().next("pick up the red mug and put it in the bin", _scene().image, [])
    assert isinstance(cmd, str) and cmd.strip()
