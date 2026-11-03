import json
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spec2mpc.core import Episode, ExecutorError, Program, RobotSpec, Scene, TaskCard, TaskPublic, TaskTest
from spec2mpc.eval.v2 import figure_round, load_round, pooled, run_suite

FAMILIES = ("arm", "quadruped", "kart")
GOALS = ("reach", "pick_place", "go_to", "open_drawer")


def _robot(family: str) -> RobotSpec:
    return RobotSpec(family=family, name=f"{family}_0", mjcf_path="x.xml", params={}, frames=["base"], actuators=[],
                     limits={}, floors={}, runtime={}, perception={})


def _cards(n: int, split: str = "val") -> list[TaskCard]:
    out = []
    for i in range(n):
        fam, goal = FAMILIES[i % 3], GOALS[i % 4]
        pub = TaskPublic(f"t{i:03d}", fam, _robot(fam), i, f"command {i}", goal)
        test = TaskTest(f"t{i:03d}", [{"type": "at_pose", "tol_m": 0.0123}], [], 5.0)
        out.append(TaskCard(pub, test, split, "gen"))
    return out


class StubEnv:
    # SimEnv stand-in: records the task it was reset to, never simulates
    def __init__(self, task: TaskPublic):
        self.task, self.reset_to = task, None

    def reset(self, task: TaskPublic, seed: int) -> Scene:
        self.reset_to = (task, seed)
        return Scene([], None, 0.0)

    def run_program(self, program: Program, max_s: float) -> Episode:
        raise AssertionError("run_suite drives the env only through run_command")

    def run_actions(self, policy: Callable, max_s: float) -> Episode:
        raise AssertionError("run_suite drives the env only through run_command")


def _stub_run_command(calls: list) -> Callable:
    # mirrors spec2mpc.orchestrator.run_command: executor failures come back as an episode with an empty trace
    def run_command(command: str, env: StubEnv, executor_url: str, executor_name: str, *, robot: RobotSpec,
                    scene: Scene, max_s: float = 60.0, task_id: str = "") -> Episode:
        calls.append((command, env.reset_to[0], executor_url, executor_name, robot, max_s))
        assert task_id == env.reset_to[0].task_id
        if task_id == "t003":
            return Episode(task_id, executor_name, [], {}, "executor_error:invalid_program", 0.0)
        if task_id == "t005":
            raise ExecutorError("http_500")
        x = np.full((4, 1), float(int(task_id[1:]) % 2))  # odd tasks succeed
        return Episode(task_id, executor_name, [], {"x": x}, "done", 0.1)
    return run_command


def _score(test: TaskTest, trace: dict) -> dict:
    ok = bool(trace["x"][-1, 0] > 0.5)
    return {"success": ok, "score": 0.9 if ok else 0.2, "per_predicate": {}}


def test_run_suite_writes_rows(tmp_path: Path):
    calls: list = []
    cards = _cards(6)
    path = run_suite("val", "stub", "http://x", n_seeds=2, workers=2, cards=cards, make_env=StubEnv,
                     run_command=_stub_run_command(calls), score_fn=_score, round_name="r0", root=tmp_path)
    assert path == tmp_path / "r0" / "stub_val.parquet"
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == 12
    for k in ("task_id", "success", "score", "family", "goal", "split", "wall_s", "n_commands", "status"):
        assert k in rows[0]
    by = {(r["task_id"], r["seed"]): r for r in rows}
    assert by[("t001", 0)]["success"] and by[("t001", 0)]["score"] == pytest.approx(0.9)
    assert not by[("t002", 1)]["success"]
    assert by[("t003", 0)]["status"] == "executor_error:invalid_program" and by[("t003", 0)]["score"] == 0.0
    assert by[("t005", 1)]["status"] == "http_500" and not by[("t005", 1)]["success"]
    # executors receive the public view only, never the test
    assert all(isinstance(c[1], TaskPublic) for c in calls)
    assert "0.0123" not in json.dumps([str(c) for c in calls])


def test_run_suite_resumes(tmp_path: Path):
    calls: list = []
    kw = dict(cards=_cards(4), make_env=StubEnv, run_command=_stub_run_command(calls), score_fn=_score,
              round_name="r0", root=tmp_path)
    run_suite("val", "stub", "http://x", **kw)
    n = len(calls)
    run_suite("val", "stub", "http://x", **kw)
    assert len(calls) == n


def test_pooled_ci_contains_mean():
    rng = np.random.default_rng(0)
    rows = [{"task_id": f"t{i}", "success": bool(rng.random() < 0.6), "score": float(rng.random())} for i in range(40)]
    p = pooled(pa.Table.from_pylist(rows))
    assert p["n_tasks"] == 40
    for m in ("success_rate", "score"):
        assert p[f"{m}_lo"] <= p[m] <= p[f"{m}_hi"]
        assert p[f"{m}_lo"] < p[f"{m}_hi"]
    assert p["score"] == pytest.approx(np.mean([r["score"] for r in rows]))


def _write_round(root: Path, executors: dict, n: int = 24, compute: bool = True) -> Path:
    # synthetic round: each executor has a base success probability; the method also at three compute budgets
    rng = np.random.default_rng(1)
    rd = root / "r0"
    rd.mkdir(parents=True)
    for split in ("val", "heldout_goal"):
        for name, p in executors.items():
            budgets = [(name, None)]
            if compute and name == "spec2mpc":
                budgets = [(f"{name}@{k}", float(k)) for k in (1, 4, 16)]
            for ex, c in budgets:
                bump = 0.0 if c is None else 0.08 * np.log2(c)
                rows = []
                for card in _cards(n, split):
                    s = float(np.clip(rng.normal(p + bump, 0.15), 0, 1))
                    rows.append({"task_id": card.public.task_id, "executor": ex, "seed": 0, "split": split,
                                 "family": card.public.family, "goal": card.public.goal, "success": s > 0.6,
                                 "score": s, "wall_s": 1.0, "n_commands": 1, "status": "done", "compute": c})
                pq.write_table(pa.Table.from_pylist(rows), rd / f"{ex}_{split}.parquet")
    return rd


def test_same_tasks_across_executors(tmp_path: Path):
    rd = _write_round(tmp_path, {"spec2mpc": 0.6, "vla": 0.4}, compute=False)
    load_round(rd)  # same sets: fine
    t = pq.read_table(rd / "vla_val.parquet")
    pq.write_table(t.slice(0, t.num_rows - 1), rd / "vla_val.parquet")
    with pytest.raises(ValueError, match="task"):
        load_round(rd)


def test_figures_render(tmp_path: Path):
    rd = _write_round(tmp_path, {"spec2mpc": 0.55, "vla": 0.35, "llm_program": 0.45})
    paths = figure_round(rd, out_dir=tmp_path / "previews")
    names = sorted(p.name for p in paths)
    assert names == ["v2_r0_compute.png", "v2_r0_family.png", "v2_r0_goal.png", "v2_r0_pooled.png"]
    for p in paths:
        assert p.stat().st_size > 5000


def test_run_suite_with_orchestrator_run_command(tmp_path: Path):
    # the real Task 10 run_command with an in-process executor, so no server is needed
    orch = pytest.importorskip("spec2mpc.orchestrator")
    from functools import partial

    class Exec:
        name = "stub"

        def act(self, command, robot, scene, feedback):
            return Program([])

    class Env(StubEnv):
        def run_program(self, program: Program, max_s: float) -> Episode:
            return Episode(self.task.task_id, "", [program], {"x": np.ones((2, 1))}, "done", 0.0)

    path = run_suite("val", "stub", "http://unused", cards=_cards(3), make_env=Env,
                     run_command=partial(orch.run_command, executor=Exec()), score_fn=_score, root=tmp_path)
    rows = pq.read_table(path).to_pylist()
    assert [r["success"] for r in rows] == [True, True, True] and {r["status"] for r in rows} == {"done"}
