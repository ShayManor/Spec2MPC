"""BARN navigation worlds (Perille et al. 2020): 300 cluttered obstacle maps for 2D rovers.

Assets: <ASSET_ROOT>/barn/BARN_dataset (BARN_dataset.zip from cs.utexas.edu/~xiao/BARN). Obstacles are the Gazebo
world cylinders (radius 0.075 m), start and goal follow the BARN challenge (start (-2.25, 3) facing +y, goal 10 m
ahead, success within 1 m of the goal without collision inside 100 s), and the BARN path (grid cells) becomes the
route. The BARN score clip(OT / clip(t, 2 OT, 8 OT)) with OT = path length / 2 m/s becomes the graded time term.
"""
from __future__ import annotations

import functools
import xml.etree.ElementTree as ET

import numpy as np

from spec2mpc.core import Entity, Scene, TaskCard, TaskPublic
from spec2mpc.tasks.sources import Translation, pred, write_coverage
from spec2mpc.tasks.sources import _nav as N

SOURCE = "barn"
ROOT = N.ASSET_ROOT / "barn" / "BARN_dataset"
N_WORLDS = 300
CELL_M = 0.15  # BARN grid cell = cylinder diameter
GRID_ORIGIN = np.array([-4.425, 5.175])  # world xy of grid cell (row 0, col 0), fit to the world files
START_XY, START_YAW = np.array([-2.25, 3.0]), np.pi / 2
GOAL_XY = np.array([-2.25, 13.0])
GOAL_R_M = 1.0
TIME_LIMIT_S = 100.0
OT_SPEED_MPS = 2.0  # BARN optimal-time speed
CYL_R_M, CYL_HALF_H_M = 0.075, 0.5
FAMILY, MEMBER = "rover2d", "rover2d"

STYLES = {
    "default": ["Drive through the cluttered obstacle field to the goal without hitting anything.",
                "Get through the obstacle course to the far side, no collisions.",
                "Navigate the cluttered area and reach the goal past it without touching any obstacle."],
    "fast": ["Get through the obstacle field to the goal as quickly as you can without hitting anything.",
             "Race through the cluttered course to the goal, but do not touch the obstacles."],
    "careful": ["Carefully make your way through the obstacles to the goal, keeping well away from them.",
                "Navigate the cluttered field to the goal and stay as far from the obstacles as you can."],
}


def available() -> bool:
    return (ROOT / "world_files" / "world_0.world").exists()


@functools.lru_cache(maxsize=None)
def cylinders(i: int) -> np.ndarray:
    # [K, 2] cylinder centers of world_i (state block holds the placed poses)
    w = ET.parse(ROOT / "world_files" / f"world_{i}.world").getroot().find("world")
    st = w.find("state")
    models = (st if st is not None else w).findall("model")
    xy = [[float(v) for v in m.find("pose").text.split()[:2]] for m in models
          if m.get("name", "").startswith("unit_cylinder")]
    return np.asarray(xy, np.float64)


def route(i: int) -> np.ndarray:
    # [N, 3] start -> BARN path cells (row, col) -> goal
    rc = np.load(ROOT / "path_files" / f"path_{i}.npy").astype(np.float64)
    xy = GRID_ORIGIN + CELL_M * rc
    return np.c_[np.vstack([START_XY, xy, GOAL_XY]), np.zeros(len(xy) + 2)]


def optimal_time_s(i: int) -> float:
    return N.arclen(route(i)) / OT_SPEED_MPS


def _build(pub: TaskPublic) -> tuple[str, Scene]:
    _, task, _ = N.parse_task_id(pub.task_id)
    i = int(task.split("_")[1])
    cyl = cylinders(i)
    w = N.World(f"barn_{task}", floor_half_m=20.0)
    for k, (x, y) in enumerate(cyl):
        w.geom(f"obstacles:{k}", "cylinder", [CYL_R_M, CYL_HALF_H_M], [x, y, CYL_HALF_H_M])
    obstacles = Entity("obstacles", "obstacle field", "fixture", N.pose7([*cyl.mean(0), CYL_HALF_H_M]),
                       np.array([CYL_R_M, CYL_R_M, CYL_HALF_H_M]), np.c_[cyl, np.full(len(cyl), CYL_HALF_H_M)],
                       None)
    ents = [N.start([*START_XY, 0.0], START_YAW), N.region("goal", N.circle(GOAL_XY, GOAL_R_M)),
            N.path("route", route(i)), obstacles]
    return w.xml(), N.scene_of(ents)


def scene(card: TaskCard | TaskPublic) -> tuple[str, Scene]:
    """MJCF world xml (no robot) and the Scene for a barn card."""
    return _build(card.public if isinstance(card, TaskCard) else card)


def _test(i: int, style: str) -> tuple[list[dict], list[dict]]:
    ot = optimal_time_s(i)
    preds = [pred("in_region", "base", "goal"), pred("no_contact", "robot", "obstacles"),
             pred("time_below", t_s=TIME_LIMIT_S)]
    if style == "careful":
        graded = [{"term": "clearance", "args": [], "good": 0.3, "bad": 0.0}]
    else:
        graded = [{"term": "time", "args": [], "good": 2 * ot, "bad": 8 * ot}]
    if style == "fast":
        preds.append(pred("time_below", t_s=round(4 * ot, 2)))
    return preds, graded


def translations() -> list[Translation]:
    return [Translation(f"world_{i}", _test(i, "default")[0], ["base", "goal", "obstacles"], STYLES["default"][0],
                        "go_to", TIME_LIMIT_S) for i in range(N_WORLDS)]


def coverage() -> dict:
    total = [f"world_{i}" for i in range(N_WORLDS)]
    return write_coverage(SOURCE, total, translations(), {}, {"styles": sorted(STYLES)})


def cards(n: int, seed: int) -> list[TaskCard]:
    """n cards over a seeded permutation of the 300 worlds, each with a style and a command."""
    if not available():
        raise RuntimeError(f"BARN assets missing under {ROOT}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(N_WORLDS)
    coverage()
    out = []
    for j in range(n):
        i = int(order[j % N_WORLDS])
        style = N.pick(rng, sorted(STYLES))
        sseed = int(rng.integers(0, 2**31 - 1))
        rb = N.robot(FAMILY, MEMBER, sseed)
        preds, graded = _test(i, style)
        out.append(N.card(SOURCE, f"world_{i}", sseed, rb, N.pick(rng, STYLES[style]), "go_to", preds, graded,
                          TIME_LIMIT_S))
    return out
