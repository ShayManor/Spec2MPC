"""Task cards: one (goal, robot, scene seed, style) -> TaskCard, splits, and the on-disk layout.

Cards live at data/v2/cards/<split>/<task_id>.json (spec2mpc.core.io JSON), the layout the eval harness and the
oracle read. The scene of a card is rebuilt from its public part alone (build_scene), so no scene file is stored.
"""
from __future__ import annotations

import dataclasses
import importlib
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np

from spec2mpc.core import RobotSpec, Scene, TaskCard, TaskPublic, save_json
from spec2mpc.tasks.commands import template_command
from spec2mpc.tasks.goals import CFG, GOALS, STYLES, GoalSpec, assemble
from spec2mpc.tasks.scenes import make_scene

CARD_ROOT = Path("data/v2/cards")
SPLITS = ("train", "val", "heldout_goal", "heldout_robot", "heldout_type")
BENCH_WORDS = ("robolab", "libero", "humanoidbench")


def _crc(s: str) -> int:
    return zlib.crc32(s.encode())


def robot_tag(robot: RobotSpec) -> str:
    # short id of the randomized parameter draw, so two draws of one robot get distinct task ids
    return f"{_crc(repr(sorted(robot.params.items()))):08x}"[:6]


def task_id(goal: str, style: str, robot: RobotSpec, scene_seed: int) -> str:
    return f"{goal}-{style}-{robot.name}-{robot_tag(robot)}-s{scene_seed}"


def build_scene(public: TaskPublic) -> tuple[str, Scene]:
    """(MJCF world, ground-truth Scene) of a card, from its public part (goal name, robot, scene seed)."""
    goal = GOALS[public.goal]
    return make_scene(goal.scene_kind(public.scene_seed), public.robot, public.scene_seed)


def make_card(goal: GoalSpec | str, robot: RobotSpec, scene_seed: int, style: str, split: str,
              source: str = "generated") -> TaskCard:
    goal = GOALS[goal] if isinstance(goal, str) else goal
    if robot.family not in goal.families:
        raise ValueError(f"goal {goal.name} does not apply to family {robot.family}")
    if style not in goal.styles:
        raise ValueError(f"goal {goal.name} has no style {style!r}")
    _, scene = make_scene(goal.scene_kind(scene_seed), robot, scene_seed)
    tid = task_id(goal.name, style, robot, scene_seed)
    # entity choices depend on (scene, goal) only, so every style of one scene and goal tests the same objects
    built = goal.build(scene, np.random.default_rng([scene_seed, _crc(goal.name)]), robot)
    test = assemble(goal, built, robot, style, tid)
    r = np.random.default_rng([scene_seed, _crc(goal.name), _crc(style)])
    phrases = STYLES[style]
    command = template_command(goal.templates, built.slots, phrases[int(r.integers(len(phrases)))], r)
    return TaskCard(TaskPublic(tid, robot.family, robot, scene_seed, command, goal.name), test, split, source)


def with_command(card: TaskCard, command: str, k: int) -> TaskCard:
    # paraphrase variant k of a card: same scene and test, new command, id "<base>~p<k>"
    tid = f"{base_id(card.public.task_id)}~p{k}"
    pub = dataclasses.replace(card.public, task_id=tid, command=command)
    return dataclasses.replace(card, public=pub, test=dataclasses.replace(card.test, task_id=tid))


def base_id(tid: str) -> str:
    return tid.split("~", 1)[0]


def make_splits(cards: Iterable[TaskCard], heldout_type: str | None = None) -> dict[str, list[str]]:
    """Task ids per split. Precedence: held-out type, held-out robot, held-out goal, val (held-out scenes), train.
    Paraphrase variants follow their base card."""
    cards = list(cards)
    cfg = CFG["splits"]
    fams = {c.public.family for c in cards}
    if heldout_type is None:
        heldout_type = cfg["heldout_type"] if cfg["heldout_type"] in fams else cfg["heldout_type_fallback"]
    hrob = {(f, n) for f, ns in cfg["heldout_robots"].items() for n in ns}
    hgoal = set(cfg["heldout_goals"])
    val_cut = int(float(cfg["val_frac"]) * 1000)
    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    for c in cards:
        p = c.public
        if p.family == heldout_type:
            s = "heldout_type"
        elif (p.family, p.robot.name) in hrob:
            s = "heldout_robot"
        elif p.goal in hgoal:
            s = "heldout_goal"
        elif _crc(f"{p.robot.name}/{p.scene_seed}") % 1000 < val_cut:
            s = "val"
        else:
            s = "train"
        out[s].append(p.task_id)
    return out


def assign_splits(cards: Iterable[TaskCard], heldout_type: str | None = None) -> list[TaskCard]:
    cards = list(cards)
    of = {t: s for s, ids in make_splits(cards, heldout_type).items() for t in ids}
    return [dataclasses.replace(c, split=of[c.public.task_id]) for c in cards]


SOURCE_MODULES = {"barn": "barn", "commonroad": "commonroad", "playground": "playground", "tracks": "tracks",
                  "robocasa": "robocasa", "metaworld": "metaworld", "robosuite": "robosuite_tasks"}


def source_available(name: str) -> bool:
    # nav sources check their assets; robocasa needs a source tree (read by static AST, no simulator import);
    # metaworld and robosuite fall back to their own task tables
    mod = importlib.import_module(f"spec2mpc.tasks.sources.{SOURCE_MODULES[name]}")
    if hasattr(mod, "available"):
        return mod.available()
    if hasattr(mod, "source_dir"):
        return mod.source_dir() is not None
    return True


def source_cards(names: Iterable[str] | None = None, n: int | None = None, seed: int = 0) -> list[TaskCard]:
    """Cards from the training-source adapters (default: every available source), split "train" until
    assign_splits. n cards per source; None = one per translated task."""
    names = [s for s in SOURCE_MODULES if source_available(s)] if names is None else list(names)
    out = []
    for s in names:
        mod = importlib.import_module(f"spec2mpc.tasks.sources.{SOURCE_MODULES[s]}")
        out += mod.cards(mod.coverage()["translated"] if n is None else n, seed)
    return out


def save_cards(cards: Iterable[TaskCard], root: Path = CARD_ROOT) -> list[Path]:
    paths = []
    for c in cards:
        if c.split not in SPLITS:
            raise ValueError(f"unknown split {c.split!r}")
        # source task ids are "<source>/<task>/<seed>"; keep one flat file per card
        p = Path(root) / c.split / f"{c.public.task_id.replace('/', '__')}.json"
        save_json(c, p)
        paths.append(p)
    return paths
