from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import numpy as np

EntityKind = Literal["object", "fixture", "robot_frame", "grasp", "path", "region"]


def _eq(a: Any, b: Any) -> bool:
    # structural equality that treats numpy arrays by value
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(
            _eq(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b


class _NumpyEq:
    def __eq__(self, other: object) -> bool:
        return _eq(self, other)

    __hash__ = None  # type: ignore[assignment]


# ---- scene ----

@dataclass(frozen=True, eq=False)
class Entity(_NumpyEq):
    id: str
    label: str
    kind: EntityKind
    pose: np.ndarray  # [7] xyz + quaternion wxyz, world frame
    extent: np.ndarray | None  # [3] box half-sizes, m
    points: np.ndarray | None  # [N, 3] world-frame points (paths, regions, clouds)
    parent: str | None  # id of the entity this one is attached to


@dataclass(frozen=True, eq=False)
class Scene(_NumpyEq):
    entities: list[Entity]
    image: np.ndarray | None  # [H, W, 3] uint8
    t_s: float

    def get(self, id: str) -> Entity:
        for e in self.entities:
            if e.id == id:
                return e
        raise KeyError(id)


# ---- robot ----

@dataclass(frozen=True)
class RobotSpec:
    family: str
    name: str
    mjcf_path: str
    params: dict[str, float]
    frames: list[str]
    actuators: list[str]
    limits: dict[str, float]
    floors: dict[str, float]
    runtime: dict[str, float]
    perception: dict[str, float]

    def describe(self) -> str:
        # the text the executor reads about this robot
        def kv(d: dict[str, float]) -> str:
            return ", ".join(f"{k}={v:g}" for k, v in d.items()) or "none"

        return "\n".join([
            f"robot: {self.name} (family {self.family})",
            f"frames: {', '.join(self.frames) or 'none'}",
            f"actuators: {', '.join(self.actuators) or 'none'}",
            f"limits: {kv(self.limits)}",
            f"safety floors: {kv(self.floors)}",
            f"params: {kv(self.params)}",
            f"runtime: {kv(self.runtime)}",
            f"perception: {kv(self.perception)}",
        ])


# ---- program ----

_TERM_RE = re.compile(r"^(TERM|UNTIL) ([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$")


@dataclass(frozen=True)
class Term:
    type: str
    args: list[str]  # pointers: scene entity ids, robot frames, or symbols (open, speed, ...)
    params: dict[str, float]

    def to_text(self) -> str:
        inner = ",".join(self.args)
        if self.params:
            inner += "; " + ",".join(f"{k}={float(v)!r}" for k, v in self.params.items())
        return f"{self.type}({inner})"

    @staticmethod
    def parse(kw: str, line: str) -> Term:
        m = _TERM_RE.match(line.strip())
        if m is None or m.group(1) != kw:
            raise ValueError(f"expected {kw} line, got {line!r}")
        a, _, p = m.group(3).partition(";")
        args = [s.strip() for s in a.split(",") if s.strip()]
        params = {}
        for kv in (s for s in p.split(",") if s.strip()):
            k, _, v = kv.partition("=")
            params[k.strip()] = float(v)
        return Term(m.group(2), args, params)


@dataclass(frozen=True)
class Stage:
    terms: list[Term]
    until: Term


@dataclass(frozen=True)
class Program:
    stages: list[Stage]

    def to_text(self) -> str:
        out = []
        for st in self.stages:
            out.append("STAGE")
            out += [f"  TERM {t.to_text()}" for t in st.terms]
            out += [f"  UNTIL {st.until.to_text()}", "END"]
        return "\n".join(out)

    @staticmethod
    def from_text(s: str) -> Program:
        stages, terms, until, open_ = [], [], None, False
        for raw in s.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line == "STAGE":
                if open_:
                    raise ValueError("STAGE inside STAGE")
                terms, until, open_ = [], None, True
            elif line == "END":
                if not open_ or until is None:
                    raise ValueError("END without STAGE or UNTIL")
                stages.append(Stage(terms, until))
                open_ = False
            elif not open_ or until is not None:
                raise ValueError(f"unexpected line {raw!r}")
            elif line.startswith("TERM "):
                terms.append(Term.parse("TERM", line))
            else:
                until = Term.parse("UNTIL", line)
        if open_:
            raise ValueError("unterminated STAGE")
        return Program(stages)

    def to_json(self) -> dict:
        return {"stages": [{"terms": [dataclasses.asdict(t) for t in st.terms],
                            "until": dataclasses.asdict(st.until)} for st in self.stages]}

    @staticmethod
    def from_json(d: dict | str) -> Program:
        if isinstance(d, str):
            d = json.loads(d)

        def term(t: dict) -> Term:
            return Term(t["type"], list(t["args"]), {k: float(v) for k, v in t["params"].items()})

        return Program([Stage([term(t) for t in st["terms"]], term(st["until"])) for st in d["stages"]])


# ---- tasks ----

@dataclass(frozen=True)
class TaskPublic:
    task_id: str
    family: str
    robot: RobotSpec
    scene_seed: int
    command: str
    goal: str | None


@dataclass(frozen=True)
class TaskTest:
    # hidden: only rubric/, eval/, oracle/, tasks/ may read it
    task_id: str
    predicates: list[dict]
    graded: list[dict]
    horizon_s: float


@dataclass(frozen=True)
class TaskCard:
    public: TaskPublic
    test: TaskTest
    split: str
    source: str

    def public_view(self) -> TaskPublic:
        # the only object passed to executors
        return self.public


# ---- episodes ----

@dataclass(frozen=True, eq=False)
class StepResult(_NumpyEq):
    obs_scene: Scene
    qpos: np.ndarray  # [nq]
    done: bool
    info: dict


@dataclass(frozen=True, eq=False)
class Episode(_NumpyEq):
    task_id: str
    executor: str
    programs: list[Program]
    trace: dict[str, np.ndarray]  # name -> [T, ...]
    status: str
    wall_s: float


# ---- protocols ----

class ExecutorError(Exception):
    def __init__(self, code: str, msg: str = ""):
        super().__init__(f"{code}: {msg}" if msg else code)
        self.code = code


@runtime_checkable
class ActionChunkSource(Protocol):
    def next_chunk(self, scene: Scene, qpos: np.ndarray) -> np.ndarray: ...  # [T, act_dim]


@runtime_checkable
class Executor(Protocol):
    name: str

    def act(self, command: str, robot: RobotSpec, scene: Scene,
            feedback: str | None) -> Program | ActionChunkSource: ...


@runtime_checkable
class SimEnv(Protocol):
    def reset(self, task: TaskPublic, seed: int) -> Scene: ...

    def run_program(self, program: Program, max_s: float) -> Episode: ...

    def run_actions(self, policy: Callable[[Scene], np.ndarray], max_s: float) -> Episode: ...


@runtime_checkable
class BenchAdapter(Protocol):
    name: str

    def tasks(self) -> list[str]: ...

    def run(self, task_name: str, executor_url: str, seed: int) -> dict: ...  # {"success", "score", "info"}
