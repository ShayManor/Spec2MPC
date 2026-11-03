"""Shared helpers for the locomotion, navigation, and vehicle sources (playground, barn, commonroad, tracks).

Scene conventions every module here follows (the SimEnv reads them at reset):
  start   region  robot initial pose (xyz + yaw quaternion); the sim places the robot base there at rest
  goal    region  target region; `points` holds an xy polygon (in_region uses it), else pose + extent is a box
  route   path    reference path from start to goal, [N, 3] world points
MJCF geoms of a multi-geom entity are named "<entity id>:<k>", so contact ids match rubric `match()`.
Scenes and tests are rebuilt from TaskPublic alone (task_id, scene_seed, robot), so executors can get the scene.
"""
from __future__ import annotations

import os
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from spec2mpc.core import Entity, RobotSpec, Scene, TaskCard, TaskPublic, TaskTest

ASSET_ROOT = Path(os.environ.get("LANGUAGESPEC_ASSETS", Path.home() / "languagespec-assets"))
BENCHMARK_WORDS = ("robolab", "libero", "humanoidbench")  # never in training scenes or asset paths


def yaw_quat(yaw: float) -> np.ndarray:
    # [4] wxyz rotation about world z
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def pose7(xyz: np.ndarray | list[float], yaw: float = 0.0) -> np.ndarray:
    return np.concatenate([np.asarray(xyz, np.float64).reshape(3), yaw_quat(yaw)])


def circle(xy: np.ndarray, r: float, n: int = 24) -> np.ndarray:
    # [n, 3] polygon approximating a circle at z = 0
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), np.zeros(n)], 1)


def rect(xy: np.ndarray, half_len: float, half_wid: float, yaw: float) -> np.ndarray:
    # [4, 3] corners of an oriented rectangle at z = 0
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], np.float64) * [half_len, half_wid]
    w = loc @ np.array([[c, s], [-s, c]]) + np.asarray(xy)[:2]
    return np.concatenate([w, np.zeros((4, 1))], 1)


def region(id: str, poly: np.ndarray, z: float = 0.0, yaw: float = 0.0, height: float = 1.0) -> Entity:
    # polygon region; pose at the centroid, extent the bounding half-sizes
    xy = poly[:, :2].mean(0)
    ext = np.r_[np.abs(poly[:, :2] - xy).max(0), height / 2]
    return Entity(id, id, "region", pose7([xy[0], xy[1], z], yaw), ext, np.asarray(poly, np.float64), None)


def box_region(id: str, center: np.ndarray, half: np.ndarray, yaw: float = 0.0) -> Entity:
    # oriented 3D box region (in_region falls back to inside() when points is None)
    return Entity(id, id, "region", pose7(center, yaw), np.asarray(half, np.float64), None, None)


def path(id: str, pts: np.ndarray) -> Entity:
    pts = np.asarray(pts, np.float64)
    return Entity(id, id, "path", pose7(pts[0]), None, pts, None)


def start(xyz: np.ndarray | list[float], yaw: float) -> Entity:
    return Entity("start", "start", "region", pose7(xyz, yaw), np.array([0.25, 0.25, 0.25]), None, None)


class World:
    """Static MJCF world (no robot): the SimEnv attaches the robot at the `start` entity."""

    def __init__(self, name: str, floor_half_m: float = 200.0):
        self.name = name
        self.assets: list[str] = []
        self.geoms: list[str] = []
        self.bodies: list[str] = []
        self.floor_half_m = floor_half_m

    @staticmethod
    def _g(name: str, type: str, size: list[float], pos: list[float], yaw: float = 0.0, rgba: str = "0.6 0.6 0.6 1",
           extra: str = "") -> str:
        q = yaw_quat(yaw)
        return (f'<geom name={quoteattr(name)} type="{type}" size="{" ".join(f"{v:.4g}" for v in size)}" '
                f'pos="{" ".join(f"{v:.6g}" for v in pos)}" quat="{" ".join(f"{v:.6g}" for v in q)}" '
                f'rgba="{rgba}"{extra}/>')

    def geom(self, name: str, type: str, size: list[float], pos: list[float], yaw: float = 0.0,
             rgba: str = "0.6 0.6 0.6 1") -> None:
        self.geoms.append(self._g(name, type, size, pos, yaw, rgba))

    def mocap_box(self, name: str, half: list[float], pos: list[float], yaw: float) -> None:
        # moving obstacle: the sim sets mocap_pos/quat from the entity trajectory
        q = yaw_quat(yaw)
        g = self._g(name, "box", half, [0, 0, 0], 0.0, "0.8 0.3 0.2 1")
        self.bodies.append(f'<body name={quoteattr(name)} mocap="true" pos="{" ".join(f"{v:.6g}" for v in pos)}" '
                           f'quat="{" ".join(f"{v:.6g}" for v in q)}">{g}</body>')

    def wall(self, prefix: str, pts: np.ndarray, half_thick: float = 0.1, half_h: float = 0.5,
             start_k: int = 0) -> int:
        # one thin box per polyline segment pts [N, 2+]; returns the next geom index
        k = start_k
        for a, b in zip(pts[:-1, :2], pts[1:, :2]):
            d = b - a
            L = float(np.hypot(*d))
            if L < 1e-6:
                continue
            m = (a + b) / 2
            self.geom(f"{prefix}:{k}", "box", [L / 2 + half_thick, half_thick, half_h], [m[0], m[1], half_h],
                      float(np.arctan2(d[1], d[0])))
            k += 1
        return k

    def xml(self, floor: str | None = None) -> str:
        # floor: a ready <geom> for the ground (e.g. an hfield), else a plane named "floor"
        fl = floor or self._g("floor", "plane", [self.floor_half_m, self.floor_half_m, 0.1], [0, 0, 0],
                              rgba="0.3 0.3 0.3 1")
        asset = f"<asset>{''.join(self.assets)}</asset>" if self.assets else ""
        return (f'<mujoco model={quoteattr(self.name)}>{asset}<worldbody>'
                f'<light pos="0 0 50" dir="0 0 -1" directional="true"/>{fl}'
                f'{"".join(self.geoms)}{"".join(self.bodies)}</worldbody></mujoco>')


def robot(family: str, name: str, seed: int) -> RobotSpec:
    from spec2mpc.robots.v2 import load_robot  # Task 3 registry

    return load_robot(family, name, seed)


def members(family: str) -> list[str]:
    from spec2mpc.robots.v2 import list_robots

    return list_robots(family)


def task_id(source: str, task: str, scene_seed: int) -> str:
    return f"{source}/{task}/{scene_seed}"


def parse_task_id(tid: str) -> tuple[str, str, int]:
    source, task, seed = tid.split("/")
    return source, task, int(seed)


def card(source: str, task: str, scene_seed: int, rb: RobotSpec, command: str, goal: str, predicates: list[dict],
         graded: list[dict], horizon_s: float) -> TaskCard:
    tid = task_id(source, task, scene_seed)
    pub = TaskPublic(tid, rb.family, rb, scene_seed, command, goal)
    return TaskCard(pub, TaskTest(tid, predicates, graded, float(horizon_s)), "train", source)


def scene_of(entities: list[Entity]) -> Scene:
    ids = [e.id for e in entities]
    assert len(ids) == len(set(ids)), "duplicate entity ids"
    return Scene(entities, None, 0.0)


def pick(rng: np.random.Generator, xs: list | tuple):
    return xs[int(rng.integers(len(xs)))]


def resample(pts: np.ndarray, step: float, closed: bool = False) -> np.ndarray:
    # polyline [N, D] resampled at fixed arc length along its first two coordinates -> [M, D]
    p = np.vstack([pts, pts[:1]]) if closed else pts
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p[:, :2], axis=0).T))]
    q = np.arange(0.0, s[-1] if closed else s[-1] + 1e-9, step)
    return np.stack([np.interp(q, s, p[:, j]) for j in range(p.shape[1])], 1)


def arclen(pts: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(pts[:, :2], axis=0).T)))
