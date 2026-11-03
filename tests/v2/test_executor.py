import json
import threading
import time

import jsonschema
import numpy as np
import pytest

from spec2mpc.core import (Entity, ExecutorError, Program, RobotSpec, Scene, Stage, TaskCard, TaskPublic, TaskTest,
                           Term, to_jsonable)
from spec2mpc.executor import HFGenerator, VLMExecutor, build_prompt, create_app, prompt_text
from spec2mpc.executor.decode import Generation, parse_program, split_output
from spec2mpc.executor.modal_app import (BudgetExceeded, budget_total, estimate_usd, log_cost, measured_usd,
                                         reserve)
from spec2mpc.executor.prompt import EXAMPLES
from spec2mpc.executor.rl import RewardClient, make_reward_fn, to_grpo_row
from spec2mpc.executor.sft import time_checkpoint_callback, to_sft_row, zero_shot_eval
from spec2mpc.orchestrator.loop import HTTPExecutor
from spec2mpc.program import json_schema, validate

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
TINY_REV = "ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda"


def _robot(family: str = "arm", frames: tuple[str, ...] = ("ee", "base")) -> RobotSpec:
    return RobotSpec(family=family, name=f"{family}_0", mjcf_path="x.xml", params={"mass_kg": 18.0},
                     frames=list(frames), actuators=["a0", "a1"], limits={"speed_mps": 1.0},
                     floors={"min_clearance_m": 0.02, "max_speed_frac": 0.5, "max_force_frac": 0.5},
                     runtime={"control_hz": 50.0}, perception={"pos_noise_m": 0.005})


def _scene(ids: tuple[tuple[str, str], ...] = (("mug_0", "object"), ("mug_0_grasp_0", "grasp"),
                                               ("table_0", "fixture"), ("goal_0", "region"))) -> Scene:
    return Scene([Entity(i, i.split("_")[0], k, np.array([0.5, 0.0, 0.1, 1, 0, 0, 0.0]),
                         np.array([0.04, 0.04, 0.05]), None, None) for i, k in ids], None, 0.0)


def _pick() -> Program:
    return Program([Stage([Term("pose_to", ["ee", "mug_0_grasp_0"], {"z_offset_m": 0.1}), Term("gripper", ["open"], {})],
                          Term("dist_below", ["ee", "mug_0_grasp_0"], {"d": 0.02})),
                    Stage([Term("gripper", ["close"], {"force_N": 20.0})], Term("gripper_closed", [], {}))])


class StubGen:
    def __init__(self, texts: list[str]):
        self.texts, self.calls = texts, 0

    def generate(self, messages: list[dict], schema: dict, n: int) -> list[Generation]:
        t = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return [Generation("r", t)] * n


def test_prompt_has_no_test_content():
    test = TaskTest("t0", predicates=[{"pred": "in_region", "args": ["mug_0", "goal_0"], "tol_m": 0.0137},
                                      {"pred": "not_fallen", "args": ["robot"]}],
                    graded=[{"pred": "time_below", "t_s": 4.321, "weight": 0.777}], horizon_s=12.5)
    card = TaskCard(TaskPublic("t0", "arm", _robot(), 3, "put the mug in the goal area", "place"), test,
                    "train", "generator")
    pub = card.public_view()
    msgs = build_prompt(pub.command, pub.robot, _scene(), "previous command ended with status timeout",
                        docs=True, n_examples=3)
    text = json.dumps([[p.get("text", "") for p in m["content"]] for m in msgs]) + prompt_text(msgs)
    assert "put the mug in the goal area" in text and "mug_0_grasp_0" in text
    for bad in ("predicates", "graded", "in_region", "not_fallen", "time_below", "0.0137", "4.321", "0.777",
                "12.5", "horizon"):
        assert bad not in text, bad


def test_image_attached_when_present():
    img = np.zeros((8, 8, 3), np.uint8)
    s = _scene()
    msgs = build_prompt("go", _robot(), Scene(s.entities, img, 0.0), None)
    assert msgs[1]["content"][0]["type"] == "image" and msgs[1]["content"][-1]["type"] == "text"
    assert all(p["type"] == "text" for p in build_prompt("go", _robot(), s, None)[1]["content"])


def test_examples_valid():
    for ex in EXAMPLES:
        fam, rest = ex["robot"].split(" (frames: ")
        robot = _robot(fam, tuple(rest.rstrip(")").split(", ")))
        ids = tuple((e.split(" ")[0], e.split("[")[1].split("]")[0]) for e in ex["entities"])
        assert validate(Program.from_text(ex["program"]), robot, _scene(ids)) == [], ex["command"]


@pytest.fixture(scope="module")
def tiny():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(TINY, revision=TINY_REV)
    model = transformers.AutoModelForCausalLM.from_pretrained(TINY, revision=TINY_REV).eval()
    return model, tok


def test_decode_always_valid(tiny):
    model, tok = tiny
    robot, scene = _robot("arm", ("ee", "base", "link7")), _scene()
    schema = json_schema(robot, scene)
    gen = HFGenerator(model, tok, max_reasoning_tokens=0, temperature=1.0, seed=0, n_numbers=5, max_extra_items=2)
    outs = gen.generate(build_prompt("pick up the mug", robot, scene, None), schema, 50)
    assert len(outs) == 50
    texts = set()
    for g in outs:
        jsonschema.Draft202012Validator(schema).validate(json.loads(g.text))
        p = parse_program(g.text)
        assert p is not None and validate(p, robot, scene) == []
        texts.add(g.text)
    assert len(texts) > 10  # sampling, not one fixed program


def test_hf_scorer_matches_full_forward(tiny):
    import torch

    from spec2mpc.executor.decode import HFScorer

    model, tok = tiny
    sc = HFScorer(model, tok)
    a, b, cands = "<|im_start|>user\npick the mug", '{"stages": [', ["]", '{"terms"', " x y z"]
    sc(a, ["x", "y"])
    got = sc(a + b, cands)
    ids = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    want = []
    for c in cands:
        x = ids(a) + ids(b) + ids(c)
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=torch.tensor([x])).logits[0].float(), -1)
        n = len(ids(c))
        want.append(np.mean([float(lp[len(x) - n - 1 + j, x[len(x) - n + j]]) for j in range(n)]))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    assert sc.cache.get_seq_length() == len(ids(a)) + len(ids(b))


def test_executor_with_tiny_model(tiny):
    model, tok = tiny
    ex = VLMExecutor(HFGenerator(model, tok, max_reasoning_tokens=4, n_numbers=5, max_extra_items=1), n_samples=3)
    p = ex.act("walk to the goal", _robot("quadruped", ("base", "foot_fl")), _scene(), None)
    assert validate(p, _robot("quadruped", ("base", "foot_fl")), _scene()) == []


def test_invalid_then_error():
    # Review Focus 3: truncated JSON at the token limit, twice -> typed error
    trunc = json.dumps(_pick().to_json())[:57]
    gen = StubGen([trunc, trunc, json.dumps(_pick().to_json())])
    with pytest.raises(ExecutorError) as e:
        VLMExecutor(gen).act("pick up the mug", _robot(), _scene(), None)
    assert e.value.code == "invalid_program" and gen.calls == 2
    gen = StubGen([trunc, json.dumps(_pick().to_json())])
    assert VLMExecutor(gen).act("pick up the mug", _robot(), _scene(), None) == _pick() and gen.calls == 2


def test_invalid_reference_is_retried():
    bad = Program([Stage([Term("pose_to", ["ee", "ghost_9"], {})], Term("timeout", [], {"t_s": 1.0}))])
    gen = StubGen([json.dumps(bad.to_json())])
    with pytest.raises(ExecutorError) as e:
        VLMExecutor(gen).act("go", _robot(), _scene(), None)
    assert "invalid_reference" in str(e.value) and gen.calls == 2


def test_split_output():
    g = split_output("<think>\nmug is red\n</think>\n\n{\"stages\": []}")
    assert g.reasoning == "mug is red" and g.text == '{"stages": []}'
    assert split_output('{"a": 1}').reasoning == ""


class EchoExecutor:
    name = "echo"

    def __init__(self, barrier: threading.Barrier | None = None):
        self.seen: list[tuple[str, str | None]] = []
        self.barrier = barrier

    def act(self, command, robot, scene, feedback):
        self.seen.append((command, feedback))
        if self.barrier is not None:
            self.barrier.wait(timeout=10)  # both episodes are inside act at once
        return _pick()


def test_server_isolates_episodes():
    # Review Focus 5: concurrent clients with different episode ids keep separate feedback state
    ex = EchoExecutor(threading.Barrier(2))
    app = create_app({"echo": ex})
    payload = {"executor": "echo", "robot": to_jsonable(_robot()), "scene": to_jsonable(_scene())}
    results: dict[str, list] = {}

    def client(eid: str) -> None:
        c = app.test_client()
        out = []
        for k in range(2):
            r = c.post("/act", json={**payload, "episode_id": eid, "command": f"cmd {eid}",
                                     "feedback": f"fb {eid} {k}"})
            out.append(r.status_code)
        results[eid] = out

    ts = [threading.Thread(target=client, args=(e,)) for e in ("A", "B")]
    [t.start() for t in ts]
    [t.join(20) for t in ts]
    assert results == {"A": [200, 200], "B": [200, 200]}
    eps = app.config["episodes"]
    assert eps["A"].feedback == ["fb A 0", "fb A 1"] and eps["B"].feedback == ["fb B 0", "fb B 1"]
    for cmd, fb in ex.seen:
        eid = cmd.split()[-1]
        other = "B" if eid == "A" else "A"
        assert f"fb {eid} 0" in fb and f"fb {other}" not in fb


def _post_via(app):
    c = app.test_client()

    def post(url: str, payload: dict) -> tuple[int, dict]:
        r = c.post(url.removeprefix("http://exec"), json=payload)
        return r.status_code, r.get_json()

    return post


def test_server_roundtrip_with_http_executor():
    class Failing:
        name = "bad"

        def act(self, *a):
            raise ExecutorError("invalid_program", "truncated")

    app = create_app({"echo": EchoExecutor(), "bad": Failing()})
    ok = HTTPExecutor("http://exec", "echo", "ep1", post=_post_via(app))
    assert ok.act("pick", _robot(), _scene(), None) == _pick()
    with pytest.raises(ExecutorError) as e:
        HTTPExecutor("http://exec", "bad", "ep2", post=_post_via(app)).act("pick", _robot(), _scene(), None)
    assert e.value.code == "invalid_program"
    with pytest.raises(ExecutorError) as e:
        HTTPExecutor("http://exec", "nope", "ep3", post=_post_via(app)).act("pick", _robot(), _scene(), None)
    assert e.value.code == "unknown_executor"


def test_chunk_endpoint():
    class Chunks:
        def __init__(self, k: float):
            self.k = k

        def next_chunk(self, scene, qpos):
            return np.full((4, 3), self.k) + qpos[:3]

    class VLA:
        name = "vla"

        def act(self, command, robot, scene, feedback):
            return Chunks(1.0 if "one" in command else 2.0)

    app = create_app({"vla": VLA()})
    c = app.test_client()
    base = {"executor": "vla", "robot": to_jsonable(_robot()), "scene": to_jsonable(_scene())}
    assert c.post("/chunk", json={"executor": "vla", "episode_id": "x", "scene": to_jsonable(_scene()),
                                  "qpos": [0.0] * 3}).status_code == 409
    for eid, cmd in (("e1", "one"), ("e2", "two")):
        r = c.post("/act", json={**base, "episode_id": eid, "command": cmd, "feedback": None})
        assert r.status_code == 200 and r.get_json() == {"chunks": True}
    r1 = c.post("/chunk", json={"executor": "vla", "episode_id": "e1", "scene": to_jsonable(_scene()),
                                "qpos": [0.5, 0.0, 0.0, 9.0]})
    r2 = c.post("/chunk", json={"executor": "vla", "episode_id": "e2", "scene": to_jsonable(_scene()),
                                "qpos": [0.0] * 4})
    a1, a2 = np.array(r1.get_json()["actions"]), np.array(r2.get_json()["actions"])
    assert a1.shape == (4, 3) and a1[0, 0] == 1.5 and np.all(a2 == 2.0)


def test_budget_guard(tmp_path):
    led = tmp_path / "budget.jsonl"
    log_cost("old-run", 90.0, "final", path=led)
    r = reserve("sft", 8.0, 100.0, led)
    assert budget_total(led) == pytest.approx(98.0)
    with pytest.raises(BudgetExceeded):
        reserve("rl", 2.5, 100.0, led)  # 98 + 2.5 > 100 refuses
    lines_before = led.read_text().count("\n")
    log_cost(r, 3.0, "final", path=led)  # measured below the reservation frees budget
    assert budget_total(led) == pytest.approx(93.0)
    reserve("rl", 2.5, 100.0, led)
    assert led.read_text().count("\n") == lines_before + 2


def test_budget_estimate_from_config():
    import yaml

    from spec2mpc.executor.modal_app import ROOT

    mc = yaml.safe_load((ROOT / "configs/v2/executor.yaml").read_text())["modal"]
    usd, h = estimate_usd(mc, "rl")
    assert h == 6.0 and usd == pytest.approx((2 * 3.95 + 0.2) * 6 * 1.15)
    assert measured_usd(mc, "rl", h * 3600) == pytest.approx(usd)
    assert all(estimate_usd(mc, j)[0] < mc["cap_usd"] for j in mc["jobs"])


def _rec(task_id: str = "t1") -> dict:
    return {"task_id": task_id, "seed": 2, "command": "pick up the mug", "robot": to_jsonable(_robot()),
            "scene": to_jsonable(_scene())}


def test_reward_fn_scores_valid_and_penalizes_invalid():
    calls = []

    def post(url, payload):
        calls.append((url, payload))
        return {"success": True, "score": 0.75, "status": "done", "per_predicate": {"x": 1}}

    row = to_grpo_row(_rec())
    fn = make_reward_fn(RewardClient(["http://w1", "http://w2"], post), invalid_reward=-1.0, workers=2)
    good = "<think>grasp the mug</think>\n" + json.dumps(_pick().to_json())
    out = fn(prompts=[row["prompt"]] * 3, completions=[good, good[:40], [{"role": "assistant", "content": good}]],
             task_id=[row["task_id"]] * 3, seed=[row["seed"]] * 3, robot=[row["robot"]] * 3,
             scene=[row["scene"]] * 3)
    assert out == [0.75, -1.0, 0.75]
    assert {u for u, _ in calls} == {"http://w1/score", "http://w2/score"}
    assert calls[0][1]["task_id"] == "t1" and Program.from_json(calls[0][1]["program"]) == _pick()


def test_sft_row_target_parses():
    row = to_sft_row({**_rec(), "program": _pick().to_json(), "rationale": "the handle faces +y"})
    tgt = row["messages"][-1]
    assert tgt["role"] == "assistant" and row["images"] == []
    g = split_output(tgt["content"])
    assert g.reasoning == "the handle faces +y" and parse_program(g.text) == _pick()


def test_time_checkpoint_callback(monkeypatch):
    saved = []
    cb = time_checkpoint_callback(30, lambda: saved.append(1))

    class Ctl:
        should_save = False

    t = [1000.0]
    monkeypatch.setattr(time, "time", lambda: t[0])
    cb.last = 1000.0
    assert not cb.on_step_end(None, None, Ctl()).should_save
    t[0] += 30 * 60
    ctl = cb.on_step_end(None, None, Ctl())
    assert ctl.should_save
    cb.on_save(None, None, ctl)
    assert saved == [1] and cb.last == t[0]


@pytest.mark.parametrize("n_ok,decision", [(1, "skip_sft"), (0, "run_sft")])
def test_zero_shot_gate(tmp_path, n_ok, decision):
    class Ex:
        name = "z"

        def __init__(self):
            self.i = 0

        def act(self, command, robot, scene, feedback):
            self.i += 1
            if self.i == 5:
                raise ExecutorError("invalid_program")
            return _pick()

    ok_ids = {f"t{i}" for i in range(n_ok)}
    client = RewardClient(["http://w"], lambda url, p: {"success": p["task_id"] in ok_ids, "score": 0.5})
    cfg = {"zero_shot": {"gate_success": 0.2, "out": str(tmp_path / "zs.json"), "env_urls": ["http://w"],
                         "prompt": {}}}
    s = zero_shot_eval(cfg, Ex(), client, [_rec(f"t{i}") for i in range(5)])
    assert s["n"] == 5 and s["validity_rate"] == pytest.approx(0.8)
    assert s["success_rate"] == pytest.approx(n_ok / 5) and s["decision"] == decision
    assert json.loads((tmp_path / "zs.json").read_text())["decision"] == decision
