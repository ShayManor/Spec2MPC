"""Training-source adapters: map existing benchmark task suites to TaskCards (training data only).

Each source module exposes `cards(n, seed) -> list[TaskCard]` and `coverage() -> dict`. A source task becomes a
card only when its success check translates into the rubric v2 predicate vocabulary; the rest are skipped and
counted in data/v2/sources/<source>_coverage.json. Cards run in the source's own simulator (envs/robocasa), so the
robot is the source's robot and predicate args are the source's object and fixture names.

Region convention: an in_region predicate on a source card names a region that only exists as a box relative to
a source entity, carried in the predicate itself: {"pred": "in_region", "args": [obj, region_id], "at": <entity
id>, "lo": [3], "hi": [3]} means the world-aligned box at(position) + [lo, hi]. The source env turns it into the
region entity's pose and extent with region_box() and logs them as entity_pose/<region_id> and
entity_extent/<region_id>; the rubric ignores the extra fields.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spec2mpc.core import RobotSpec, TaskCard, TaskPublic, TaskTest
from spec2mpc.core.io import load_family_config

try:  # Task 4 vocabulary; the fallback keeps these adapters importable before it lands
    from spec2mpc.rubric.v2 import PREDICATES as _PREDICATES
    PREDICATE_NAMES: frozenset[str] = frozenset(_PREDICATES)
except ImportError:  # pragma: no cover
    PREDICATE_NAMES = frozenset({"on", "inside", "open_frac", "upright", "at_pose", "in_region", "not_fallen",
                                 "time_below", "no_contact", "held"})

REPO = Path(__file__).resolve().parents[3]
COVERAGE_DIR = REPO / "data" / "v2" / "sources"


SLAB_M = 10.0  # half-size standing in for "unbounded" along a free axis of a region box


def pred(name: str, *args: str, **params: float | str | list[float]) -> dict:
    # flat predicate dict, rubric v2 format: {"pred": name, "args": [...], <param>: value}
    assert name in PREDICATE_NAMES, name
    return {"pred": name, "args": list(args), **params}


def region(obj: str, region_id: str, at: str, lo: list[float], hi: list[float]) -> dict:
    # in_region on a box at(position) + [lo, hi], world-aligned (see the region convention above)
    return pred("in_region", obj, region_id, at=at, lo=[float(x) for x in lo], hi=[float(x) for x in hi])


def region_box(d: dict, at_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # region predicate + the at-entity position [3] -> (pose [7] wxyz, extent [3]) of the region entity
    lo, hi = np.asarray(d["lo"], np.float64), np.asarray(d["hi"], np.float64)
    pose = np.concatenate([np.asarray(at_pos, np.float64) + 0.5 * (lo + hi), [1.0, 0.0, 0.0, 0.0]])
    return pose, 0.5 * (hi - lo)


@dataclass
class Translation:
    task: str  # source task name
    predicates: list[dict]  # rubric v2 predicate dicts
    entities: list[str]  # ids the source env must expose as entities (predicate args)
    command: str  # natural-language instruction, no thresholds
    goal: str  # coarse goal type for pooled metrics
    horizon_s: float
    dropped: list[str] = field(default_factory=list)  # source conjuncts with no predicate (approximated away)
    robot: str = "default"  # key into the source's robot table


def source_robot(family: str, name: str, mjcf_path: str, frames: list[str], control_hz: float) -> RobotSpec:
    # the source simulator's own robot; floors always come from configs/v2/robots/<family>.yaml
    floors = {k: float(v) for k, v in load_family_config(family)["floors"].items()}
    return RobotSpec(family=family, name=name, mjcf_path=mjcf_path, params={}, frames=list(frames), actuators=[],
                     limits={}, floors=floors, runtime={"control_hz": float(control_hz)}, perception={})


def _numbers(d: dict) -> list[float]:
    out = []
    for k, v in d.items():
        if k in ("pred", "args"):
            continue
        if not isinstance(v, str):
            out += [float(x) for x in np.ravel(v)]
    return out


def command_leaks(command: str, predicates: list[dict]) -> bool:
    # true if any predicate threshold appears as a number in the command
    nums = {float(m) for m in re.findall(r"\d+(?:\.\d+)?", command)}
    return any(abs(x - y) < 1e-9 for d in predicates for x in _numbers(d) for y in nums)


def card_errors(card: TaskCard, entities: list[str], robot_ids: tuple[str, ...] = ("robot",)) -> list[str]:
    # card rules shared with the generator: known predicates, args resolve, no thresholds in the command
    errs = []
    if not card.source:
        errs.append("no_source")
    if not card.test.predicates:
        errs.append("no_predicates")
    known = set(entities) | set(card.public.robot.frames) | set(robot_ids)
    for d in card.test.predicates:
        if d.get("pred") not in PREDICATE_NAMES:
            errs.append(f"unknown_predicate:{d.get('pred')}")
        errs += [f"unknown_entity:{a}" for a in d.get("args", []) if a not in known]
    if command_leaks(card.public.command, card.test.predicates):
        errs.append("command_threshold")
    return errs


def make_cards(source: str, translations: list[Translation], robots: dict[str, RobotSpec], n: int,
               seed: int) -> list[TaskCard]:
    # n cards cycling through a seeded permutation of the translated tasks (distinct tasks first)
    if not translations:
        raise RuntimeError(f"{source}: no translated tasks (is the source installed?)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(translations))
    out = []
    for i in range(n):
        tr = translations[int(order[i % len(order)])]
        scene_seed = int(rng.integers(0, 2**31 - 1))
        tid = f"{source}/{tr.task}/{scene_seed}"
        robot = robots[tr.robot]
        pub = TaskPublic(tid, robot.family, robot, scene_seed, tr.command, tr.goal)
        out.append(TaskCard(pub, TaskTest(tid, [dict(d) for d in tr.predicates], [], tr.horizon_s), "train",
                            source))
    return out


def write_coverage(source: str, total: list[str], translations: list[Translation], skipped: dict[str, str],
                   extra: dict | None = None) -> dict:
    # data/v2/sources/<source>_coverage.json: translated and skipped counts plus reasons
    reasons: dict[str, int] = {}
    for r in skipped.values():
        key = r.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    cov = {
        "source": source,
        "total": len(total),
        "translated": len(translations),
        "skipped": len(skipped),
        "translated_with_dropped_conjuncts": sum(bool(t.dropped) for t in translations),
        "skip_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "translated_tasks": {t.task: {"predicates": t.predicates, "dropped": t.dropped} for t in translations},
        "skipped_tasks": dict(sorted(skipped.items())),
        **(extra or {}),
    }
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    (COVERAGE_DIR / f"{source}_coverage.json").write_text(json.dumps(cov, indent=1))
    return cov
