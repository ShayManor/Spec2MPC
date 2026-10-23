from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.batch import Batched
from spec2mpc.mpc.theta import COST_NAMES, ThetaBatch
from spec2mpc.robots.kart import KartParams


@dataclass
class MPCWeights(Batched):
    # every field [B]
    q_progress: Tensor
    q_contour: Tensor
    q_lag: Tensor
    q_speed_ref: Tensor
    r_accel: Tensor
    r_jerk: Tensor
    r_effort: Tensor
    q_clearance: Tensor
    rho_slack: Tensor
    v_cap_mps: Tensor
    a_lat_cap_mps2: Tensor
    a_long_cap_mps2: Tensor
    margin_m: Tensor
    v_ref_mps: Tensor


_COST_FIELDS = ("q_progress", "q_contour", "q_lag", "q_speed_ref", "r_accel", "r_jerk", "r_effort", "q_clearance", "rho_slack")


def adapt(theta: ThetaBatch, p: KartParams, mpc_cfg: DictConfig) -> MPCWeights:
    # theta.values [B, D_THETA], p fields [B] -> MPCWeights fields [B]; torch.clamp keeps gradients where unclipped
    d = float(mpc_cfg.delta_max)
    fl, mx = mpc_cfg.floors, mpc_cfg.max
    costs = {}
    for field, name in zip(_COST_FIELDS, COST_NAMES):
        delta = torch.clamp(theta.get(name), -d, d)
        costs[field] = float(mpc_cfg.w_nom[name]) * torch.exp(delta)
    costs["rho_slack"] = torch.clamp(costs["rho_slack"], min=float(fl.rho_slack_min))
    c_vmax = torch.clamp(theta.get("c_vmax"), float(fl.c_vmax), float(mx.c_vmax))
    c_alat = torch.clamp(theta.get("c_alat"), float(fl.c_alat), float(mx.c_alat))
    c_along = torch.clamp(theta.get("c_along"), float(fl.c_along), float(mx.c_along))
    v_ref = torch.clamp(theta.get("v_ref"), float(mpc_cfg.v_ref_min), 1.0)
    return MPCWeights(
        **costs,
        v_cap_mps=c_vmax * p.v_max_mps,
        a_lat_cap_mps2=c_alat * p.a_lat_max_mps2,
        a_long_cap_mps2=c_along * p.a_long_max_mps2,
        margin_m=torch.clamp(theta.get("c_margin"), float(fl.m_floor_m), float(mx.c_margin_m)),
        v_ref_mps=v_ref * p.v_max_mps,
    )
