from dataclasses import dataclass
from typing import Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.batch import Batched

COST_NAMES = ("w_progress", "w_contour", "w_lag", "w_speed_ref", "w_accel", "w_jerk", "w_effort", "w_clearance", "w_slack")
CONSTRAINT_NAMES = ("c_vmax", "c_alat", "c_along", "c_margin")
THETA_NAMES = COST_NAMES + CONSTRAINT_NAMES + ("v_ref",)
D_THETA = len(THETA_NAMES)


@dataclass
class ThetaBatch(Batched):
    # values: [B, D_THETA]; delta for costs, fraction for c_vmax/c_alat/c_along/v_ref, meters for c_margin
    values: Tensor

    def get(self, name: str) -> Tensor:
        # -> [B]
        return self.values[:, THETA_NAMES.index(name)]


def default_theta(B: int, mpc_cfg: DictConfig) -> ThetaBatch:
    # -> values [B, D_THETA]
    row = [0.0] * len(COST_NAMES) + [float(mpc_cfg.constraints.nominal[n]) for n in CONSTRAINT_NAMES + ("v_ref",)]
    return ThetaBatch(values=torch.tensor(row, dtype=torch.float32).repeat(B, 1))


def theta_bounds(mpc_cfg: DictConfig) -> Tuple[Tensor, Tensor]:
    # -> lo, hi [D_THETA] float32: the canonical Theta box
    d = float(mpc_cfg.delta_max)
    lo = [-d] * len(COST_NAMES)
    hi = [d] * len(COST_NAMES)
    for c in ("c_vmax", "c_alat", "c_along"):
        lo.append(float(mpc_cfg.floors[c]))
        hi.append(float(mpc_cfg.max[c]))
    lo.append(float(mpc_cfg.floors.m_floor_m))
    hi.append(float(mpc_cfg.max.c_margin_m))
    lo.append(float(mpc_cfg.v_ref_min))
    hi.append(1.0)
    return torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32)


def normalize_theta(values: Tensor, mpc_cfg: DictConfig) -> Tensor:
    # values [..., D_THETA] in the Theta box -> [..., D_THETA] in [-1, 1]
    lo, hi = (b.to(values.device) for b in theta_bounds(mpc_cfg))
    return 2.0 * (values - lo) / (hi - lo) - 1.0


def denormalize_theta(u: Tensor, mpc_cfg: DictConfig) -> Tensor:
    # u [..., D_THETA] (clamped to [-1, 1]) -> values [..., D_THETA] in the Theta box
    lo, hi = (b.to(u.device) for b in theta_bounds(mpc_cfg))
    return lo + (hi - lo) * (u.clamp(-1.0, 1.0) + 1.0) / 2.0


def sobol_theta(n: int, mpc_cfg: DictConfig, seed: int) -> ThetaBatch:
    # -> values [n, D_THETA], scrambled Sobol mapped to the Theta bounds
    u = torch.quasirandom.SobolEngine(D_THETA, scramble=True, seed=seed).draw(n)
    lo_t, hi_t = theta_bounds(mpc_cfg)
    return ThetaBatch(values=(lo_t + (hi_t - lo_t) * u).to(torch.float32))
