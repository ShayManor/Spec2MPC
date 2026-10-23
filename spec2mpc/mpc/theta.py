from dataclasses import dataclass

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


def sobol_theta(n: int, mpc_cfg: DictConfig, seed: int) -> ThetaBatch:
    # -> values [n, D_THETA], scrambled Sobol mapped to the Theta bounds
    u = torch.quasirandom.SobolEngine(D_THETA, scramble=True, seed=seed).draw(n)
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
    lo_t, hi_t = torch.tensor(lo), torch.tensor(hi)
    return ThetaBatch(values=(lo_t + (hi_t - lo_t) * u).to(torch.float32))
