import math
from dataclasses import dataclass, fields, replace
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.batch import Batched
from spec2mpc.sim.paths import RefPath, path_lookup, project_global

TASK_LAP, TASK_GOTO = 0, 1


@dataclass
class Scene(Batched):
    path: RefPath
    obstacles_xy: Tensor  # [B, O, 2]
    obstacles_r: Tensor  # [B, O] (padding r = 0)
    ped_xy0: Tensor  # [B, P, 2]
    ped_vel: Tensor  # [B, P, 2]
    ped_r: Tensor  # [B, P] (padding r = 0)
    task: Tensor  # [B] int, 0 = lap, 1 = goto
    n_laps: Tensor  # [B] int
    goal_tol_m: Tensor  # [B]
    seed: Tensor  # [B] int64

    @property
    def batch_size(self) -> int:
        return self.path.batch_size

    def index(self, idx: Union[Tensor, slice]) -> "Scene":
        return replace(self, **{f.name: getattr(self, f.name).index(idx) if f.name == "path" else getattr(self, f.name)[idx] for f in fields(self)})

    def to(self, device: Any) -> "Scene":
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    @classmethod
    def cat(cls, items: Sequence["Scene"]) -> "Scene":
        # items: list of [B_i, ...] -> [sum B_i, ...]; obstacle/pedestrian slots are padded to the batch max
        o = max(i.obstacles_r.shape[1] for i in items)
        p = max(i.ped_r.shape[1] for i in items)

        def pad(x: Tensor, n: int) -> Tensor:
            return torch.cat([x, torch.zeros(x.shape[0], n - x.shape[1], *x.shape[2:], dtype=x.dtype, device=x.device)], dim=1)

        kw = {}
        for f in fields(cls):
            if f.name == "path":
                kw["path"] = RefPath.cat([i.path for i in items])
                continue
            n = o if f.name.startswith("obstacles") else p if f.name.startswith("ped") else None
            kw[f.name] = torch.cat([pad(getattr(i, f.name), n) if n is not None else getattr(i, f.name) for i in items])
        return cls(**kw)


def pedestrians_at(scene: Scene, t_s: Union[Tensor, float]) -> Tensor:
    # t_s: [B] or float -> xy [B, P, 2]
    # Constant velocity in the path (Frenet) frame: velocity at t=0 is split into along/normal components,
    # the lateral coordinate reflects off +-(half_width - r) and s reflects (open) or wraps (closed).
    path = scene.path
    xy0 = scene.ped_xy0
    t = torch.as_tensor(t_s, dtype=xy0.dtype, device=xy0.device)
    t = t.expand(xy0.shape[0]) if t.dim() == 0 else t
    t = t[:, None]  # [B, 1]
    s0 = project_global(path, xy0)  # [B, P]
    p0, h0, _ = path_lookup(path, s0)
    tang = torch.stack([torch.cos(h0), torch.sin(h0)], dim=-1)
    nrm = torch.stack([-torch.sin(h0), torch.cos(h0)], dim=-1)
    d0 = ((xy0 - p0) * nrm).sum(-1)
    vs = (scene.ped_vel * tang).sum(-1)
    vd = (scene.ped_vel * nrm).sum(-1)
    a = (path.half_width_m[:, None] - scene.ped_r).clamp(min=1e-3)  # [B, P]
    u = torch.remainder(d0 + vd * t + a, 4 * a)
    d = torch.where(u < 2 * a, u, 4 * a - u) - a
    length = path.length_m[:, None]
    s_lin = s0 + vs * t
    u_s = torch.remainder(s_lin, 2 * length)
    s_open = torch.where(u_s < length, u_s, 2 * length - u_s)
    s = torch.where(path.closed[:, None], torch.remainder(s_lin, length), s_open)
    p, h, _ = path_lookup(path, s)
    n = torch.stack([-torch.sin(h), torch.cos(h)], dim=-1)
    return p + d[..., None] * n


def true_clearance(scene: Scene, xy: Tensor, t_s: Union[Tensor, float]) -> Tensor:
    # xy [B, K, 2] -> [B, K] signed distance to the nearest occupied region (negative inside)
    path = scene.path
    s = project_global(path, xy)
    p, _, _ = path_lookup(path, s)
    clr = path.half_width_m[:, None] - (xy - p).norm(dim=-1)  # [B, K]

    def circles(c: Tensor, r: Tensor) -> Tensor:
        d = (xy[:, :, None] - c[:, None]).norm(dim=-1) - r[:, None]  # [B, K, N]
        return torch.where((r > 0)[:, None], d, torch.full_like(d, float("inf"))).amin(dim=-1)

    clr = torch.minimum(clr, circles(scene.obstacles_xy, scene.obstacles_r))
    return torch.minimum(clr, circles(pedestrians_at(scene, t_s), scene.ped_r))


class _Rand:
    def __init__(self, seed: int):
        self.g = torch.Generator().manual_seed(seed)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * float(torch.rand(1, generator=self.g))

    def randint(self, lo: int, hi: int) -> int:  # inclusive
        return int(torch.randint(lo, hi + 1, (1,), generator=self.g))


def _second_derivs(pts: np.ndarray, t: np.ndarray, closed: bool) -> np.ndarray:
    # pts [N, 2], t [N] (+ closing knot t[N] if closed) -> second derivatives [N, 2] of the cubic spline
    n = len(pts)
    h = np.diff(t)  # [N-1] open, [N] closed
    a = np.zeros((n, n))
    rhs = np.zeros((n, 2))
    if closed:
        y = np.concatenate([pts, pts[:1]])
        for i in range(n):
            hm, hp = h[(i - 1) % n], h[i]
            a[i, (i - 1) % n] += hm / 6
            a[i, i] += (hm + hp) / 3
            a[i, (i + 1) % n] += hp / 6
            prev = y[n - 1] if i == 0 else y[i - 1]
            rhs[i] = (y[i + 1] - y[i]) / hp - (y[i] - prev) / hm
    else:
        a[0, 0] = a[-1, -1] = 1.0
        for i in range(1, n - 1):
            a[i, i - 1], a[i, i], a[i, i + 1] = h[i - 1] / 6, (h[i - 1] + h[i]) / 3, h[i] / 6
            rhs[i] = (pts[i + 1] - pts[i]) / h[i] - (pts[i] - pts[i - 1]) / h[i - 1]
    return np.linalg.solve(a, rhs)


def _spline_dense(pts: np.ndarray, closed: bool, fine_ds_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # pts [N, 2] -> dense position [F, 2], first derivative [F, 2], second derivative [F, 2] in chord parameter
    knots = np.concatenate([pts, pts[:1]]) if closed else pts
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(knots, axis=0), axis=1))])
    m = _second_derivs(pts, t, closed)
    if closed:
        m = np.concatenate([m, m[:1]])
        y = knots
    else:
        y = pts
    tf = np.linspace(0.0, t[-1], int(math.ceil(t[-1] / fine_ds_m)) + 1)
    i = np.clip(np.searchsorted(t, tf, side="right") - 1, 0, len(t) - 2)
    h = (t[i + 1] - t[i])[:, None]
    x0, x1 = (t[i + 1] - tf)[:, None], (tf - t[i])[:, None]
    ca, cb = y[i] / h - m[i] * h / 6, y[i + 1] / h - m[i + 1] * h / 6
    pos = m[i] * x0**3 / (6 * h) + m[i + 1] * x1**3 / (6 * h) + ca * x0 + cb * x1
    d1 = -m[i] * x0**2 / (2 * h) + m[i + 1] * x1**2 / (2 * h) - ca + cb
    d2 = m[i] * x0 / h + m[i + 1] * x1 / h
    return pos, d1, d2


def _resample(
    pts: np.ndarray, closed: bool, target_len_m: float, ds_m: float, fine_ds_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    # -> xy [n, 2], s [n], heading [n], curvature [n], length_m; uniform arc-length spacing ds_m,
    # length rescaled to an integer multiple of ds_m near target_len_m
    pos, d1, d2 = _spline_dense(pts, closed, fine_ds_m)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    n = max(int(round(target_len_m / ds_m)), 2)
    c = n * ds_m / arc[-1]
    s_new = np.arange(n if closed else n + 1) * ds_m
    q = s_new / c
    xy = np.stack([np.interp(q, arc, pos[:, 0]), np.interp(q, arc, pos[:, 1])], axis=1) * c
    heading = np.interp(q, arc, np.unwrap(np.arctan2(d1[:, 1], d1[:, 0])))
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / (d1**2).sum(1) ** 1.5
    curv = np.interp(q, arc, kappa) / c
    heading = np.arctan2(np.sin(heading), np.cos(heading))
    return xy, s_new, heading, curv, n * ds_m


def _lap_path(r: _Rand, cfg: DictConfig, ds_m: float) -> Optional[Tuple[Tuple[np.ndarray, ...], float]]:
    lc = cfg.lap
    k = r.randint(*lc.n_ctrl)
    jit = lc.angle_jitter_frac * 2 * math.pi / k
    phi = np.array([2 * math.pi * i / k + r.uniform(-jit, jit) for i in range(k)])
    rad = np.array([1.0 + lc.radial_jitter * r.uniform(-1, 1) for _ in range(k)])
    pts = np.stack([rad * np.cos(phi), rad * np.sin(phi)], axis=1)
    hw = r.uniform(*lc.half_width_m)
    xy, s, hd, cu, length = _resample(pts, True, r.uniform(*lc.length_m), ds_m, cfg.spline_fine_ds_m)
    if np.abs(cu).max() > 1.0 / cfg.min_turn_radius_floor_m:
        return None
    stride = max(1, int(round(lc.overlap_stride_m / ds_m)))
    q, sq = xy[::stride], s[::stride]
    darc = np.abs(sq[:, None] - sq[None])
    darc = np.minimum(darc, length - darc)
    dist = np.linalg.norm(q[:, None] - q[None], axis=-1)
    if ((darc > lc.overlap_arc_hw_factor * hw) & (dist < lc.overlap_dist_hw_factor * hw + lc.overlap_extra_m)).any():
        return None
    return (xy, s, hd, cu), hw


def _goto_path(r: _Rand, cfg: DictConfig, ds_m: float) -> Optional[Tuple[Tuple[np.ndarray, ...], float]]:
    gc = cfg.goto
    k = r.randint(*gc.n_ctrl)
    th, p = r.uniform(-math.pi, math.pi), np.zeros(2)
    pts = [p]
    for _ in range(k - 1):
        th += r.uniform(-gc.turn_max_rad, gc.turn_max_rad)
        p = p + np.array([math.cos(th), math.sin(th)])
        pts.append(p)
    hw = r.uniform(*gc.half_width_m)
    xy, s, hd, cu, length = _resample(np.stack(pts), False, r.uniform(*gc.length_m), ds_m, cfg.spline_fine_ds_m)
    if np.abs(cu).max() > 1.0 / max(cfg.min_turn_radius_floor_m, hw):  # keeps the corridor normal well defined
        return None
    return (xy, s, hd, cu), hw


def _place(r: _Rand, gc: DictConfig, xy: np.ndarray, s: np.ndarray, hd: np.ndarray, hw: float, rad: float,
           path_clear_m: float = 0.0) -> Optional[np.ndarray]:
    # path_clear_m: required gap between the circle's edge and the centerline (static obstacles only)
    for _ in range(gc.placement_attempts):
        sv = r.uniform(0.0, float(s[-1]))
        i = min(int(np.searchsorted(s, sv)), len(s) - 1)
        d = r.uniform(-(hw - rad), hw - rad)
        c = xy[i] + d * np.array([-math.sin(hd[i]), math.cos(hd[i])])
        if np.linalg.norm(c - xy[0]) > gc.start_clear_m and np.linalg.norm(c - xy[-1]) > gc.goal_clear_m \
                and np.linalg.norm(xy - c, axis=1).min() >= rad + path_clear_m:
            return c
    return None


def _one_scene(seed: int, cfg: DictConfig, task: str) -> dict:
    sc = cfg.scene
    r = _Rand(seed)
    maker = _lap_path if task == "lap" else _goto_path
    for _ in range(sc.max_attempts):
        out = maker(r, sc, cfg.ds_m)
        if out is not None:
            break
    else:
        raise RuntimeError(f"no valid {task} path after {sc.max_attempts} attempts")
    (xy, s, hd, cu), hw = out
    gc = sc.goto
    n_o, n_p = gc.n_obstacles[1], gc.n_peds[1]
    obs, obs_r = np.zeros((n_o, 2)), np.zeros(n_o)
    ped, ped_v, ped_r = np.zeros((n_p, 2)), np.zeros((n_p, 2)), np.zeros(n_p)
    if task == "goto":
        for j in range(r.randint(*gc.n_obstacles)):
            rad = r.uniform(*gc.obstacle_r_m)
            c = _place(r, gc, xy, s, hd, hw, rad, float(gc.path_clear_m))
            if c is not None:
                obs[j], obs_r[j] = c, rad
        for j in range(r.randint(*gc.n_peds)):
            c = _place(r, gc, xy, s, hd, hw, gc.ped_r_m)
            ang, sp = r.uniform(0, 2 * math.pi), r.uniform(*gc.ped_speed_mps)
            if c is not None:
                ped[j], ped_r[j], ped_v[j] = c, gc.ped_r_m, sp * np.array([math.cos(ang), math.sin(ang)])
    return dict(xy=xy, s=s, hd=hd, cu=cu, hw=hw, obs=obs, obs_r=obs_r, ped=ped, ped_v=ped_v, ped_r=ped_r)


def generate_scenes(n: int, cfg: DictConfig, gen: torch.Generator, task: str) -> Scene:
    # -> Scene with batch n; each scene is a deterministic function of its int64 seed (drawn from gen)
    assert task in ("lap", "goto")
    seeds = torch.randint(0, 2**62, (n,), generator=gen, dtype=torch.int64)
    recs = [_one_scene(int(sd), cfg, task) for sd in seeds]
    m = max(len(x["xy"]) for x in recs)

    def pad(a: np.ndarray) -> np.ndarray:
        return np.concatenate([a, np.repeat(a[-1:], m - len(a), axis=0)]) if len(a) < m else a

    def f(x: Any) -> Tensor:
        return torch.as_tensor(np.stack(x), dtype=torch.float32)

    closed = task == "lap"
    path = RefPath(
        xy=f([pad(x["xy"]) for x in recs]),
        s=f([pad(x["s"]) for x in recs]),
        heading=f([pad(x["hd"]) for x in recs]),
        curvature=f([pad(x["cu"]) for x in recs]),
        length_m=torch.tensor([x["s"][-1] + (cfg.ds_m if closed else 0.0) for x in recs], dtype=torch.float32),
        closed=torch.full((n,), closed, dtype=torch.bool),
        half_width_m=torch.tensor([x["hw"] for x in recs], dtype=torch.float32),
    )
    tc = cfg.scene.lap if closed else cfg.scene.goto
    return Scene(
        path=path,
        obstacles_xy=f([x["obs"] for x in recs]),
        obstacles_r=f([x["obs_r"] for x in recs]),
        ped_xy0=f([x["ped"] for x in recs]),
        ped_vel=f([x["ped_v"] for x in recs]),
        ped_r=f([x["ped_r"] for x in recs]),
        task=torch.full((n,), TASK_LAP if closed else TASK_GOTO, dtype=torch.int64),
        n_laps=torch.full((n,), cfg.scene.n_laps if closed else tc.n_laps, dtype=torch.int64),
        goal_tol_m=torch.full((n,), tc.goal_tol_m, dtype=torch.float32),
        seed=seeds,
    )
