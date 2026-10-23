from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from spec2mpc.robots.kart import KartParams
from spec2mpc.robots.kart_dynamics import GRAVITY_MPS2
from spec2mpc.sim.scenes import Scene

PHI_NAMES: tuple[str, ...] = (
    "completion", "progress_frac", "completion_time", "lap_time",
    "speed_mean", "speed_p95", "speed_max",
    "a_lat_rms", "a_lat_p99", "a_lat_max", "a_long_rms", "a_long_p99",
    "jerk_rms", "min_clearance", "time_below_margin",
    "boundary_violations", "violation_integral", "solver_failures",
    "energy_per_m", "path_error_rms",
)
"""Robot-agnostic trajectory features; no card thresholds."""


def phi(metrics: Dict[str, Tensor], p: KartParams, scene: Scene, timeout_s: Tensor) -> Tensor:
    # metrics {name: [B]}, p/scene [B], timeout_s [B] -> [B, len(PHI_NAMES)]
    m = metrics
    g = GRAVITY_MPS2

    def time(x: Tensor) -> Tensor:
        return torch.nan_to_num(x / timeout_s, nan=1.0)

    f = {
        "completion": m["completion"],
        "progress_frac": m["progress_frac"],
        "completion_time": time(m["completion_time_s"]),
        "lap_time": time(m["lap_time_s"]),
        "speed_mean": m["speed_mean_mps"] / p.v_max_mps,
        "speed_p95": m["speed_p95_mps"] / p.v_max_mps,
        "speed_max": m["speed_max_mps"] / p.v_max_mps,
        "a_lat_rms": m["a_lat_rms_g"] * g / p.a_lat_max_mps2,
        "a_lat_p99": m["a_lat_p99_g"] * g / p.a_lat_max_mps2,
        "a_lat_max": m["a_lat_max_g"] * g / p.a_lat_max_mps2,
        "a_long_rms": m["a_long_rms_g"] * g / p.a_long_max_mps2,
        "a_long_p99": m["a_long_p99_g"] * g / p.a_long_max_mps2,
        "jerk_rms": m["jerk_rms_mps3"] / p.a_long_max_mps2,  # per second
        "min_clearance": m["min_clearance_m"] / p.radius_m,
        "time_below_margin": time(m["time_below_margin_s"]),
        "boundary_violations": torch.log1p(m["boundary_violations"]),
        "violation_integral": m["violation_integral_ms"] / (p.radius_m * timeout_s),
        "solver_failures": torch.log1p(m["solver_failures"]),
        "energy_per_m": m["energy_per_m_Whpm"] / (p.max_drive_force_N / 3600.0),
        "path_error_rms": m["path_error_rms_m"] / scene.path.half_width_m,
    }
    out = torch.stack([f[k] for k in PHI_NAMES], dim=1)
    return torch.nan_to_num(out, nan=0.0)
