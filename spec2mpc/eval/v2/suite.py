"""Run one executor over one split of the generated suite: one row per (task, seed), scored by the hidden test."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from spec2mpc.core import Episode, ExecutorError, SimEnv, TaskCard, TaskPublic, load_json

CARD_ROOT = Path("data/v2/cards")
EVAL_ROOT = Path("data/v2/eval")
ROW_KEYS = ("task_id", "executor", "seed", "split", "family", "goal", "success", "score", "wall_s", "n_commands",
            "status", "compute")

RunCommand = Callable[..., "Episode | list[Episode]"]  # orchestrator.run_command signature
ScoreFn = Callable[..., dict]


def load_split_cards(split: str, root: Path = CARD_ROOT) -> list[TaskCard]:
    # data/v2/cards/<split>/<task_id>.json written by the task generator (Task 5)
    return [load_json(p) for p in sorted((root / split).glob("*.json"))]


def _default_run_command() -> RunCommand:
    from spec2mpc.orchestrator import run_command  # Task 10
    return run_command


def _default_score() -> ScoreFn:
    from spec2mpc.rubric.v2 import score  # Task 4
    return score


def _one(card: TaskCard, seed: int, executor_name: str, executor_url: str, make_env: Callable[[TaskPublic], SimEnv],
         run_command: RunCommand, score_fn: ScoreFn, compute: float | None) -> dict:
    pub = card.public_view()  # the env and the executor only ever see the public view
    env = make_env(pub)
    t0 = time.perf_counter()
    scene = env.reset(pub, seed)
    try:
        # the horizon only bounds the env episode; the executor never receives it
        res = run_command(pub.command, env, executor_url, executor_name, robot=pub.robot, scene=scene,
                          max_s=card.test.horizon_s, task_id=pub.task_id)
        eps = res if isinstance(res, list) else [res]
        # the final state decides success, so the last episode's trace is scored; executor failures have no trace
        if eps and eps[-1].trace:
            s = score_fn(card.test, eps[-1].trace)
            success, score_v = bool(s["success"]), float(s["score"])
        else:
            success, score_v = False, 0.0
        status, n_cmd = eps[-1].status if eps else "empty", len(eps)
    except ExecutorError as e:
        success, score_v, status, n_cmd = False, 0.0, e.code, 1
    return {"task_id": pub.task_id, "executor": executor_name, "seed": seed, "split": card.split,
            "family": pub.family, "goal": pub.goal or "", "success": success, "score": score_v,
            "wall_s": time.perf_counter() - t0, "n_commands": n_cmd, "status": status, "compute": compute}


def run_suite(split: str, executor_name: str, executor_url: str, n_seeds: int = 1, workers: int = 1, *,
              cards: Sequence[TaskCard] | None = None, make_env: Callable[[TaskPublic], SimEnv] | None = None,
              run_command: RunCommand | None = None, score_fn: ScoreFn | None = None, round_name: str = "r0",
              root: Path = EVAL_ROOT, compute: float | None = None) -> Path:
    # writes <root>/<round>/<executor>_<split>.parquet; finished rows are appended to a .partial.jsonl next to it
    # as they arrive, so a resubmitted run skips them
    if make_env is None:
        raise ValueError("make_env is required: pass a factory TaskPublic -> SimEnv")
    cards = load_split_cards(split) if cards is None else list(cards)
    run_command = run_command or _default_run_command()
    score_fn = score_fn or _default_score()
    out = Path(root) / round_name / f"{executor_name}_{split}.parquet"
    part = out.with_suffix(".partial.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in part.read_text().splitlines() if l.strip()] if part.exists() else []
    done = {(r["task_id"], r["seed"]) for r in rows}
    todo = [(c, s) for c in cards for s in range(n_seeds) if (c.public.task_id, s) not in done]
    lock = threading.Lock()

    def work(item: tuple[TaskCard, int]) -> None:
        r = _one(item[0], item[1], executor_name, executor_url, make_env, run_command, score_fn, compute)
        with lock:
            rows.append(r)
            with part.open("a") as f:
                f.write(json.dumps(r) + "\n")

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(work, todo))
    keep = {(c.public.task_id, s) for c in cards for s in range(n_seeds)}
    rows = sorted((r for r in rows if (r["task_id"], r["seed"]) in keep), key=lambda r: (r["task_id"], r["seed"]))
    schema = pa.schema([("task_id", pa.string()), ("executor", pa.string()), ("seed", pa.int64()),
                        ("split", pa.string()), ("family", pa.string()), ("goal", pa.string()),
                        ("success", pa.bool_()), ("score", pa.float64()), ("wall_s", pa.float64()),
                        ("n_commands", pa.int64()), ("status", pa.string()), ("compute", pa.float64())])
    pq.write_table(pa.Table.from_pylist([{k: r[k] for k in ROW_KEYS} for r in rows], schema=schema), out)
    return out
