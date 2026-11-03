import json
from types import SimpleNamespace

import numpy as np
import pytest
from flask import Flask

from spec2mpc.core import Entity, ExecutorError, Program, RobotSpec, Scene, Stage, Term, to_jsonable
from spec2mpc.executor import build_prompt, create_app, prompt_text
from spec2mpc.executor import server
from spec2mpc.program import json_schema, validate


def _robot() -> RobotSpec:
    return RobotSpec(family="kart", name="kart_0", mjcf_path="x.xml", params={"mass_kg": 160.0},
                     frames=["base"], actuators=["steer", "throttle"], limits={"speed_mps": 10.0},
                     floors={"min_clearance_m": 0.3, "max_speed_frac": 0.9, "max_force_frac": 0.9},
                     runtime={"control_hz": 20.0}, perception={"pos_noise_m": 0.05})


def _scene(image: np.ndarray | None = None) -> Scene:
    return Scene([Entity("goal_0", "goal", "region", np.array([5.0, 2.0, 0.0, 1, 0, 0, 0.0]),
                         np.array([0.5, 0.5, 0.1]), None, None),
                  Entity("cone_0", "cone", "object", np.array([2.0, 1.0, 0.0, 1, 0, 0, 0.0]),
                         np.array([0.1, 0.1, 0.2]), None, None)], image, 0.0)


def _goto() -> Program:
    return Program([Stage([Term("pose_to", ["base", "goal_0"], {}), Term("limit", ["speed"], {"value": 0.5})],
                          Term("dist_below", ["base", "goal_0"], {"d": 0.3}))])


class FakeResponses:
    def __init__(self, texts: list[str]):
        self.texts, self.calls = texts, []

    def create(self, **kw):
        self.calls.append(kw)
        t = self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]
        return SimpleNamespace(status="completed", output_text=t)


def _client(texts: list[str]) -> SimpleNamespace:
    return SimpleNamespace(responses=FakeResponses(texts))


def _ex(tmp_path, texts: list[str]):
    from spec2mpc.baselines.llm_program import LLMProgramExecutor

    c = _client(texts)
    return LLMProgramExecutor(client=c, cache_path=tmp_path / "cache.json"), c.responses


def _sent_text(kw: dict) -> str:
    # every text part the API call carries: instructions plus input_text parts
    parts = [kw["instructions"]] + [p["text"] for m in kw["input"] for p in m["content"] if p["type"] == "input_text"]
    return "\n\n".join(parts)


def test_same_prompt_content_as_executor(tmp_path):
    ex, api = _ex(tmp_path, [json.dumps(_goto().to_json())])
    img = np.zeros((4, 4, 3), np.uint8)
    ex.act("drive to the goal", _robot(), _scene(img), "previous command ended with status timeout")
    kw = api.calls[0]
    want = prompt_text(build_prompt("drive to the goal", _robot(), _scene(img),
                                    "previous command ended with status timeout", docs=True))
    assert _sent_text(kw) == want
    assert "vocabulary" in kw["instructions"] and "pose_to(" in kw["instructions"]
    assert [p["type"] for p in kw["input"][0]["content"]] == ["input_image", "input_text"]
    fmt = kw["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert kw["model"] == "gpt-6.1-sol"


def test_schema_is_program_schema(tmp_path):
    from spec2mpc.baselines.llm_program import openai_schema

    ex, api = _ex(tmp_path, [json.dumps(_goto().to_json())])
    ex.act("drive to the goal", _robot(), _scene(), None)
    assert api.calls[0]["text"]["format"]["schema"] == openai_schema(json_schema(_robot(), _scene()))


def test_cache_hit_no_call(tmp_path):
    ex, api = _ex(tmp_path, [json.dumps(_goto().to_json())])
    p1 = ex.act("drive to the goal", _robot(), _scene(), None)
    assert len(api.calls) == 1
    p2 = ex.act("drive to the goal", _robot(), _scene(), None)
    assert len(api.calls) == 1 and p1 == p2
    # a fresh executor on the same cache file also hits
    ex2, api2 = _ex(tmp_path, ["unused"])
    assert ex2.act("drive to the goal", _robot(), _scene(), None) == p1 and api2.calls == []
    # a different command, robot name, or scene misses
    ex.act("drive to the cone", _robot(), _scene(), None)
    moved = Scene([_scene().entities[0]], None, 0.0)
    ex.act("drive to the goal", _robot(), moved, None)
    assert len(api.calls) == 3


def test_output_validated(tmp_path):
    bad = Program([Stage([Term("pose_to", ["base", "mug_9"], {})], Term("timeout", [], {"t_s": 3.0}))])
    assert validate(bad, _robot(), _scene())
    good = json.dumps(_goto().to_json())
    # invalid, then valid: one retry
    ex, api = _ex(tmp_path, [json.dumps(bad.to_json()), good])
    assert ex.act("drive to the goal", _robot(), _scene(), None) == _goto() and len(api.calls) == 2
    # invalid twice (truncated JSON, then a dropped entity): typed error, nothing cached
    ex, api = _ex(tmp_path / "b", ['{"stages": [{"terms"', json.dumps(bad.to_json())])
    with pytest.raises(ExecutorError) as e:
        ex.act("drive to the goal", _robot(), _scene(), None)
    assert e.value.code == "invalid_program" and len(api.calls) == 2
    assert not (tmp_path / "b" / "cache.json").exists() or json.loads((tmp_path / "b" / "cache.json").read_text()) == {}


def test_openai_schema_roundtrip():
    from spec2mpc.baselines.llm_program import from_openai, openai_schema

    s = openai_schema(json_schema(_robot(), _scene()))
    # strict structured output: every object lists all its properties as required
    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object":
                assert set(n.get("required", [])) == set(n.get("properties", {})) and n["additionalProperties"] is False
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(s)
    assert from_openai({"type": "limit", "args": ["speed"], "params": {"value": 0.5, "w": None}}) == \
        {"type": "limit", "args": ["speed"], "params": {"value": 0.5}}


def test_served_over_act(tmp_path):
    ex, api = _ex(tmp_path, [json.dumps(_goto().to_json())])
    app = create_app({"llm_program": ex}).test_client()
    r = app.post("/act", json={"executor": "llm_program", "episode_id": "e0", "command": "drive to the goal",
                               "robot": to_jsonable(_robot()), "scene": to_jsonable(_scene()), "feedback": None})
    assert r.status_code == 200 and Program.from_json(r.get_json()["program"]) == _goto()


def test_server_main_registers_llm_program(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(Flask, "run", lambda self, *a, **k: seen.setdefault("app", self))
    import spec2mpc.baselines.llm_program as lp

    monkeypatch.setattr(lp, "LLMProgramExecutor", lambda: SimpleNamespace(name="llm_program", act=lambda *a: _goto()))
    server.main({"server": {"host": "127.0.0.1", "port": 0}}, names=("llm_program",))
    r = seen["app"].test_client().post("/act", json={"executor": "llm_program", "episode_id": "e", "command": "c",
                                                     "robot": to_jsonable(_robot()), "scene": to_jsonable(_scene())})
    assert r.status_code == 200 and Program.from_json(r.get_json()["program"]) == _goto()
