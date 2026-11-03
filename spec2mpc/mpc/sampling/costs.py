"""One JAX cost per VOCAB term, evaluated on a rollout's feature sequence.

Objects enter only as targets and clearance geometry (no object dynamics). A feature sequence is a dict of
arrays with a leading time axis of length H + 1 (index 0 is the current state):
  fpos [H+1, F, 3], fmat [H+1, F, 3, 3]   robot frames (RobotSpec.frames order)
  qpos [H+1, nq], qvel [H+1, nv], ctrl [H+1, nu], force [H+1, nu]
  base_pos [H+1, 3], base_mat [H+1, 3, 3], com [H+1, 3], t [H+1]
Each builder returns (fn, dyn): fn(feat, dyn) -> [H] per-step cost; dyn is a pytree of arrays refreshed from the
scene every control step by Ref.dyn, so entity motion between steps needs no recompile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from spec2mpc.core import Entity, Scene, Term

CostFn = Callable[[dict, Any], jnp.ndarray]  # (feat, dyn) -> [H]
AXES = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1), "-x": (-1, 0, 0), "-y": (0, -1, 0), "-z": (0, 0, -1)}


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    # q [4] wxyz -> [3, 3]
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@dataclass
class Ref:
    # a resolved pointer: a robot frame (frame >= 0, entity None), a static entity, or an entity attached to frame
    name: str
    frame: int  # index into RobotSpec.frames, -1 for a free entity
    entity: str | None
    off_pos: np.ndarray | None = None  # [3] entity position in the frame (attached only)
    off_mat: np.ndarray | None = None  # [3, 3]

    def dyn(self, scene: Scene) -> dict:
        # arrays the jitted cost reads for this pointer; raises KeyError when perception dropped the entity
        if self.entity is None:
            return {}
        e = scene.get(self.entity)
        ext = np.zeros(3) if e.extent is None else np.asarray(e.extent, np.float64)
        d = {"ext": ext}
        if self.frame < 0:
            d["pos"], d["mat"] = np.asarray(e.pose[:3], np.float64), quat_to_mat(e.pose[3:])
        else:
            d["pos"], d["mat"] = self.off_pos, self.off_mat
        return d

    def pos(self, feat: dict, d: dict) -> jnp.ndarray:  # -> [H+1, 3]
        if self.frame < 0:
            return jnp.broadcast_to(d["pos"], feat["fpos"].shape[:1] + (3,))
        p = feat["fpos"][:, self.frame]
        if self.entity is None:
            return p
        return p + jnp.einsum("tij,j->ti", feat["fmat"][:, self.frame], d["pos"])

    def mat(self, feat: dict, d: dict) -> jnp.ndarray:  # -> [H+1, 3, 3]
        if self.frame < 0:
            return jnp.broadcast_to(d["mat"], feat["fmat"].shape[:1] + (3, 3))
        r = feat["fmat"][:, self.frame]
        return r if self.entity is None else jnp.einsum("tij,jk->tik", r, d["mat"])


def resolve(name: str, frames: list[str], scene: Scene, fpos: np.ndarray, fmat: np.ndarray) -> Ref:
    # robot frame names win; an entity whose parent is a robot frame rides on that frame (held objects)
    if name in frames:
        return Ref(name, frames.index(name), None)
    e: Entity = scene.get(name)
    if e.parent is not None and e.parent in frames:
        j = frames.index(e.parent)
        r = fmat[j]
        off = r.T @ (np.asarray(e.pose[:3], np.float64) - fpos[j])
        return Ref(name, j, name, off, r.T @ quat_to_mat(e.pose[3:]))
    return Ref(name, -1, name)


def box_dist(p: jnp.ndarray, c: jnp.ndarray, r: jnp.ndarray, ext: jnp.ndarray) -> jnp.ndarray:
    # distance from points p [..., 3] to boxes centred c [..., 3], rotation r [..., 3, 3], half-sizes ext [3]
    local = jnp.einsum("...ji,...j->...i", r, p - c)
    q = jnp.abs(local) - ext
    return jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) + jnp.minimum(jnp.max(q, axis=-1), 0.0)


def _norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(x * x, axis=-1) + 1e-12)


def _rot_err(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    # [H, 3, 3] x2 -> [H], 0 when aligned, 2 at 180 degrees: (3 - trace(a^T b)) / 2
    return 0.5 * (3.0 - jnp.einsum("tij,tij->t", a, b))


def _speed(feat: dict, dt: float) -> jnp.ndarray:
    # base linear speed by finite differences, [H]
    return _norm(jnp.diff(feat["base_pos"], axis=0)) / dt


# ---- term builders: (term, ctx) -> (fn, dyn) ----

class Ctx:
    # what the builders need: resolved pointers, robot facts, and cost settings
    def __init__(self, frames: list[str], scene: Scene, fpos: np.ndarray, fmat: np.ndarray, cfg: dict,
                 ctrl_dt: float, feet: list[int], planar: bool, nominal_height: float):
        self.frames, self.scene, self.fpos, self.fmat = frames, scene, fpos, fmat
        self.w, self.cfg, self.dt = cfg["weights"], cfg, ctrl_dt
        self.feet, self.planar, self.nominal_height = feet, planar, nominal_height

    def ref(self, name: str) -> Ref:
        return resolve(name, self.frames, self.scene, self.fpos, self.fmat)


def _pose_to(t: Term, c: Ctx) -> tuple[CostFn, list[Ref]]:
    a, b = c.ref(t.args[0]), c.ref(t.args[1])
    off = np.array([t.params.get("x_offset_m", 0.0), t.params.get("y_offset_m", 0.0), t.params.get("z_offset_m", 0.0)])
    w = t.params.get("w", 1.0)
    wp, wr = w * t.params.get("w_pos", c.w["pose_pos"]), w * t.params.get("w_rot", c.w["pose_rot"])
    term_w = c.w["terminal"]

    def fn(feat: dict, d: list) -> jnp.ndarray:
        pa, pb = a.pos(feat, d[0])[1:], b.pos(feat, d[1])[1:] + off  # [H, 3]
        e = wp * _norm(pa - pb) + wr * _rot_err(a.mat(feat, d[0])[1:], b.mat(feat, d[1])[1:])
        return e.at[-1].multiply(1.0 + term_w)
    return fn, [a, b]


def _relative_pose(t: Term, c: Ctx) -> tuple[CostFn, list[Ref]]:
    # hold a fixed relative to b, as it is at stage start
    a, b = c.ref(t.args[0]), c.ref(t.args[1])
    w = t.params.get("w", 1.0)
    wp, wr = w * t.params.get("w_pos", c.w["relative_pos"]), w * t.params.get("w_rot", c.w["relative_rot"])

    def fn(feat: dict, d: list) -> jnp.ndarray:
        pa, pb, ra, rb = a.pos(feat, d[0]), b.pos(feat, d[1]), a.mat(feat, d[0]), b.mat(feat, d[1])
        rel_p = jnp.einsum("tji,tj->ti", rb, pa - pb)  # a in b's frame, [H+1, 3]
        rel_r = jnp.einsum("tji,tjk->tik", rb, ra)
        return wp * _norm(rel_p[1:] - rel_p[0]) + wr * _rot_err(rel_r[1:], jnp.broadcast_to(rel_r[0], rel_r[1:].shape))
    return fn, [a, b]


def _approach_axis(t: Term, c: Ctx) -> tuple[CostFn, list[Ref]]:
    # come at b along the world axis: stay on the line through b and close to the standoff point dist_m before b
    a, b = c.ref(t.args[0]), c.ref(t.args[1])
    ax = jnp.asarray(AXES[t.args[2]], jnp.float32)
    dist, w = t.params.get("dist_m", 0.0), t.params.get("w", 1.0) * c.w["approach"]

    def fn(feat: dict, d: list) -> jnp.ndarray:
        r = a.pos(feat, d[0])[1:] - b.pos(feat, d[1])[1:]  # [H, 3]
        perp = r - (r @ ax)[:, None] * ax
        return w * (_norm(perp) + _norm(r + dist * ax))
    return fn, [a, b]


def _follow_path(t: Term, c: Ctx) -> tuple[CostFn, list[Ref]]:
    a = c.ref(t.args[0])
    e = c.scene.get(t.args[1])
    pts = np.asarray(e.points, np.float64)  # [N, 3]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = max(float(s[-1]), 1e-6)
    wc, wl = t.params.get("w_contour", 1.0) * c.w["path_contour"], t.params.get("w_lag", 0.0)
    wpr = t.params.get("w_progress", 1.0) * c.w["path_progress"]
    v_ref = t.params.get("v_ref_frac")
    dims = 2 if c.planar else 3

    def fn(feat: dict, d: list) -> jnp.ndarray:
        p = a.pos(feat, d[0])[:, :dims]  # [H+1, dims]
        q = jnp.asarray(pts[:, :dims])
        dd = _norm(p[:, None, :] - q[None])  # [H+1, N]
        k = jnp.argmin(dd, axis=1)
        sk = jnp.asarray(s)[k]
        ds = jnp.diff(sk)
        cost = wc * jnp.min(dd, axis=1)[1:] - wpr * ds / L + wl * jnp.maximum(-ds, 0.0)  # lag: moving backwards
        if v_ref is not None:
            cost = cost + c.w["path_speed"] * (_speed(feat, c.dt) - d[1]["vmax"] * v_ref) ** 2
        return cost.at[-1].add(wpr * (1.0 - sk[-1] / L))
    return fn, [a, _Const({"vmax": np.asarray(c.cfg.get("vmax", 1.0))})]


def _velocity_to(t: Term, c: Ctx) -> tuple[CostFn, list[Ref]]:
    # body velocity of frame a in its heading frame (vx forward, vy left, vz up) and yaw rate
    a = c.ref(t.args[0])
    want = np.array([t.params.get("vx_mps", 0.0), t.params.get("vy_mps", 0.0), t.params.get("vz_mps", 0.0)])
    wz = t.params.get("wz_radps", 0.0)
    w = t.params.get("w", 1.0) * c.w["velocity"]

    def fn(feat: dict, d: list) -> jnp.ndarray:
        p, r = a.pos(feat, d[0]), a.mat(feat, d[0])
        v = jnp.diff(p, axis=0) / c.dt  # [H, 3] world
        yaw = jnp.arctan2(r[:, 1, 0], r[:, 0, 0])  # [H+1]
        cy, sy = jnp.cos(yaw[1:]), jnp.sin(yaw[1:])
        vb = jnp.stack([cy * v[:, 0] + sy * v[:, 1], -sy * v[:, 0] + cy * v[:, 1], v[:, 2]], -1)
        dyaw = jnp.angle(jnp.exp(1j * jnp.diff(yaw))) / c.dt
        return w * (jnp.sum((vb - want) ** 2, axis=-1) + (dyaw - wz) ** 2)
    return fn, [a]


def _contact(t: Term, c: Ctx) -> tuple[CostFn, list[Ref]]:
    # reach b's surface; the force target becomes a small push past it (objects have no dynamics here)
    a, b = c.ref(t.args[0]), c.ref(t.args[1])
    w = t.params.get("w", 1.0) * c.w["contact"]
    depth = 1e-4 * t.params.get("force_N", 0.0)

    def fn(feat: dict, d: list) -> jnp.ndarray:
        pa = a.pos(feat, d[0])[1:]
        pb, rb = b.pos(feat, d[1])[1:], b.mat(feat, d[1])[1:]
        ext = d[1].get("ext", jnp.zeros(3))
        return w * jnp.abs(box_dist(pa, pb, rb, ext) + depth)
    return fn, [a, b]


def _gait(t: Term, c: Ctx) -> tuple[CostFn, list[Ref]]:
    # foot height follows a periodic schedule; pairs alternate for trot and pace, all feet down for stand
    freq = t.params.get("freq_hz", c.cfg["gait"]["default_freq_hz"])
    w, h = t.params.get("w", 1.0) * c.w["gait"], c.cfg["gait"]["step_height_m"]
    n = len(c.feet)
    pattern = t.args[0]
    if pattern == "stand" or n == 0:
        phase = np.zeros(n)
        h = 0.0
    elif pattern == "trot":
        phase = np.array([0.0, 0.5, 0.5, 0.0][:n]) if n == 4 else np.arange(n) * 0.5
    elif pattern == "pace":
        phase = np.array([0.0, 0.5, 0.0, 0.5][:n]) if n == 4 else np.arange(n) * 0.5
    elif pattern == "bound":
        phase = np.array([0.0, 0.0, 0.5, 0.5][:n]) if n == 4 else np.arange(n) * 0.5
    else:  # walk: one foot at a time
        phase = np.arange(n) / max(n, 1)
    feet = np.asarray(c.feet, np.int32)

    def fn(feat: dict, d: list) -> jnp.ndarray:
        if n == 0:
            return jnp.zeros(feat["t"].shape[0] - 1)
        z = feat["fpos"][1:, feet, 2]  # [H, n]
        ph = 2 * jnp.pi * (freq * feat["t"][1:, None] + phase[None])
        target = d[0]["ground"] + h * jnp.maximum(jnp.sin(ph), 0.0)
        return w * jnp.sum((z - target) ** 2, axis=-1)
    ground = float(np.min(c.fpos[feet, 2])) if n else 0.0
    return fn, [_Const({"ground": np.asarray(ground)})]


def _balance(t: Term, c: Ctx) -> tuple[CostFn, list[Ref]]:
    # centre of mass over the feet, base upright, base near its standing height
    w = t.params.get("w", 1.0)
    margin = t.params.get("margin_m", 0.0)
    feet = np.asarray(c.feet, np.int32)
    wb, wu, wh = w * c.w["balance"], w * c.w["upright"], w * c.w["height"]

    def fn(feat: dict, d: list) -> jnp.ndarray:
        com = feat["com"][1:, :2]
        sup = feat["fpos"][1:][:, feet, :2].mean(axis=1) if len(feet) else feat["base_pos"][1:, :2]
        off = jnp.maximum(_norm(com - sup) - margin, 0.0)
        up = 1.0 - feat["base_mat"][1:, 2, 2]
        dh = jnp.minimum(feat["base_pos"][1:, 2] - d[0]["h"], 0.0)
        return wb * off + wu * up + wh * dh ** 2
    return fn, [_Const({"h": np.asarray(c.nominal_height)})]


def _clearance(t: Term, c: Ctx) -> tuple[CostFn, list[Ref]]:
    # soft clearance of the named frames from the named entity boxes (margin already clipped to the floor)
    body, cloud = t.args
    fr = np.arange(len(c.frames)) if body == "robot" else np.array([c.frames.index(body)])
    ents = [e for e in c.scene.entities if e.kind in ("object", "fixture")] if cloud == "scene" \
        else [c.scene.get(cloud)]
    ids = [e.id for e in ents]
    margin, w = t.params.get("margin_m", 0.0), t.params.get("w", 1.0) * c.w["clearance"]

    def fn(feat: dict, d: list) -> jnp.ndarray:
        if not ids:
            return jnp.zeros(feat["t"].shape[0] - 1)
        p = feat["fpos"][1:][:, fr]  # [H, f, 3]
        dist = box_dist(p[:, :, None], d[0]["pos"][None, None], d[0]["mat"][None, None], d[0]["ext"][None, None])
        return w * jnp.sum(jnp.maximum(margin - dist, 0.0) ** 2, axis=(1, 2))
    return fn, [_Boxes(ids)]


def _smooth(t: Term, c: Ctx) -> tuple[CostFn, list[Ref]]:
    w = t.params.get("w", 1.0) * c.w["smooth"]

    def fn(feat: dict, d: list) -> jnp.ndarray:
        return w * jnp.sum(jnp.diff(feat["ctrl"], axis=0) ** 2, axis=-1)
    return fn, []


class _Const:
    # a dyn source with fixed arrays
    def __init__(self, d: dict):
        self.d, self.entity = d, None

    def dyn(self, scene: Scene) -> dict:
        return self.d


class _Boxes:
    # stacked boxes of several entities: pos [K, 3], mat [K, 3, 3], ext [K, 3]
    def __init__(self, ids: list[str]):
        self.ids, self.entity = ids, (ids[0] if ids else None)

    def dyn(self, scene: Scene) -> dict:
        es = [scene.get(i) for i in self.ids]
        if not es:
            return {"pos": np.zeros((0, 3)), "mat": np.zeros((0, 3, 3)), "ext": np.zeros((0, 3))}
        return {"pos": np.stack([e.pose[:3] for e in es]).astype(np.float64),
                "mat": np.stack([quat_to_mat(e.pose[3:]) for e in es]),
                "ext": np.stack([np.zeros(3) if e.extent is None else e.extent for e in es]).astype(np.float64)}


# gripper and limit act through the solver (fixed actuators, hard limits), not as costs
BUILDERS: dict[str, Callable[[Term, Ctx], tuple[CostFn, list]]] = {
    "pose_to": _pose_to, "relative_pose": _relative_pose, "approach_axis": _approach_axis,
    "follow_path": _follow_path, "velocity_to": _velocity_to, "contact": _contact, "gait": _gait,
    "balance": _balance, "clearance": _clearance, "smooth": _smooth,
}
NON_COST_TERMS = frozenset({"gripper", "limit"})


def entity_ids(src: Any) -> list[str]:
    # scene entity ids a dyn source reads (for the dropped-entity check)
    if isinstance(src, _Boxes):
        return list(src.ids)
    return [src.entity] if getattr(src, "entity", None) else []
