from __future__ import annotations

from dataclasses import replace
from typing import Dict

import torch
from torch import Tensor

from spec2mpc.robots.kart import KartParams
from spec2mpc.robots.kart_dynamics import GRAVITY_MPS2
from spec2mpc.sim.paths import path_ds_m, path_lookup
from spec2mpc.sim.scenes import TASK_LAP, Scene
from spec2mpc.sim.trajectory import DONE_COMPLETED, Trajectory

METRIC_NAMES: tuple[str, ...] = (
    "completion", "progress_frac", "lap_time_s", "completion_time_s",
    "speed_mean_mps", "speed_p95_mps", "speed_max_mps",
    "a_lat_rms_g", "a_lat_p99_g", "a_lat_max_g", "a_long_rms_g", "a_long_p99_g",
    "jerk_rms_mps3", "min_clearance_m", "time_below_margin_s",
    "boundary_violations", "violation_integral_ms", "solver_failures",
    "energy_Wh", "energy_per_m_Whpm", "path_error_rms_m",
)
"""SI episode metrics; this is the rubric expression namespace."""


def _masked_quantile(x: Tensor, valid: Tensor, q: float) -> Tensor:
    # x [B, T], valid [B, T] -> [B]
    return torch.nanquantile(x.masked_fill(~valid, float("nan")), q, dim=1)


def _masked_mean(x: Tensor, valid: Tensor) -> Tensor:
    # x [B, T], valid [B, T] -> [B]
    return torch.where(valid, x, 0.0).sum(1) / valid.sum(1).clamp(min=1)


def _masked_rms(x: Tensor, valid: Tensor) -> Tensor:
    # x [B, T], valid [B, T] -> [B]
    return _masked_mean(x**2, valid).sqrt()


def episode_metrics(traj: Trajectory, p: KartParams, scene: Scene, m_floor_m: float) -> Dict[str, Tensor]:
    # traj [B, T, ...], p/scene [B] -> {name: [B]} in METRIC_NAMES order
    v, dt = traj.valid, traj.dt_s  # [B, T]
    # invalid steps may hold NaN/inf: zero them so no statistic can see them
    traj = replace(
        traj,
        x=torch.where(v[..., None], traj.x, 0.0), s=torch.where(v, traj.s, 0.0),
        a_long=torch.where(v, traj.a_long, 0.0), a_lat=torch.where(v, traj.a_lat, 0.0),
        clearance_true=torch.where(v, traj.clearance_true, 0.0), p_drive_W=torch.where(v, traj.p_drive_W, 0.0),
    )
    has = v.any(1)
    n_valid = v.sum(1)
    last = (n_valid - 1).clamp(min=0)[:, None]  # [B, 1]
    s_last = torch.gather(traj.s, 1, last)[:, 0]
    length = scene.path.length_m
    n_laps = scene.n_laps.to(length.dtype)

    speed = torch.hypot(traj.x[..., 3], traj.x[..., 4])
    a_lat, a_long = traj.a_lat.abs() / GRAVITY_MPS2, traj.a_long.abs() / GRAVITY_MPS2
    jv = v[:, 1:] & v[:, :-1]
    jerk = torch.hypot(traj.a_long.diff(dim=1), traj.a_lat.diff(dim=1)) / dt  # [B, T-1]

    c = traj.clearance_true
    neg = c < 0
    entries = neg[:, 1:] & ~neg[:, :-1] & jv

    energy_Wh = torch.where(v, traj.p_drive_W, 0.0).sum(1) * dt / 3600.0
    travelled = (s_last - traj.s[:, 0]).clamp(min=path_ds_m(scene.path))

    ref_xy, _, _ = path_lookup(scene.path, traj.s)  # [B, T, 2]
    path_err = (traj.x[..., :2] - ref_xy).norm(dim=-1)

    max_laps = int(scene.n_laps.max().item())
    t_lap, n_done = torch.zeros_like(length), torch.zeros_like(length)
    for k in range(1, max_laps + 1):
        hit = (traj.s >= k * length[:, None]) & v  # [B, T]
        ok = hit.any(1) & (k <= scene.n_laps)
        t_k = torch.gather(traj.t_s, 1, hit.int().argmax(1, keepdim=True))[:, 0]
        t_lap = torch.where(ok, t_k, t_lap)
        n_done = n_done + ok
    lap_time = torch.where((n_done > 0) & (scene.task == TASK_LAP), t_lap / n_done.clamp(min=1), torch.nan)

    m = {
        "completion": ((traj.done_reason == DONE_COMPLETED) & has).to(length.dtype),
        "progress_frac": (s_last / (length * n_laps)).clamp(0.0, 1.0),
        "lap_time_s": lap_time,
        "completion_time_s": traj.completion_time_s,
        "speed_mean_mps": _masked_mean(speed, v),
        "speed_p95_mps": _masked_quantile(speed, v, 0.95),
        "speed_max_mps": speed.masked_fill(~v, -torch.inf).amax(1),
        "a_lat_rms_g": _masked_rms(a_lat, v),
        "a_lat_p99_g": _masked_quantile(a_lat, v, 0.99),
        "a_lat_max_g": a_lat.masked_fill(~v, -torch.inf).amax(1),
        "a_long_rms_g": _masked_rms(a_long, v),
        "a_long_p99_g": _masked_quantile(a_long, v, 0.99),
        "jerk_rms_mps3": _masked_rms(jerk, jv),
        "min_clearance_m": c.masked_fill(~v, torch.inf).amin(1),
        "time_below_margin_s": ((c < m_floor_m) & v).sum(1) * dt,
        "boundary_violations": entries.sum(1).to(length.dtype),
        "violation_integral_ms": torch.where(v, torch.relu(-c), 0.0).sum(1) * dt,
        "solver_failures": (~traj.solver_ok & v).sum(1).to(length.dtype),
        "energy_Wh": energy_Wh,
        "energy_per_m_Whpm": energy_Wh / travelled,
        "path_error_rms_m": _masked_rms(path_err, v),
    }
    nan_ok = ("lap_time_s", "completion_time_s")
    m["completion_time_s"] = torch.where(has, m["completion_time_s"], torch.nan)
    return {k: m[k] if k in nan_ok else torch.where(has, m[k], 0.0) for k in METRIC_NAMES}
