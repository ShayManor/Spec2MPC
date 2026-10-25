from typing import Tuple

import torch
from torch import Tensor

from spec2mpc.robots.kart import KartParams

KART_STATE = ("X", "Y", "psi", "vx", "vy", "omega", "delta", "F")
KART_INPUT = ("delta_rate", "F_rate")
NX = len(KART_STATE)
NU = len(KART_INPUT)
GRAVITY_MPS2 = 9.81


def _lateral_force(alpha: Tensor, c_lin: Tensor, fz: Tensor, mu: Tensor, p: KartParams, tire_model: str) -> Tensor:
    # alpha, c_lin, fz, mu: [B] -> [B]
    fmax = mu * fz
    if tire_model == "linear":
        return torch.minimum(torch.maximum(c_lin * alpha, -fmax), fmax)
    if tire_model == "pacejka":
        b = p.pac_B * alpha
        return fmax * torch.sin(p.pac_C * torch.atan(b - p.pac_E * (b - torch.atan(b))))
    raise ValueError(f"unknown tire_model {tire_model!r}")


def _forces(x: Tensor, p: KartParams, mu_scale: Tensor, mass_scale: Tensor, tire_model: str, vx_slip: Tensor,
            brake_vx_eps_mps: float):
    # x: [B,NX]; mu_scale, mass_scale, vx_slip: [B] -> (m, fx, fyf, fyr) each [B]
    vx, vy, om, delta, F = x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7]
    m = p.mass_kg * mass_scale
    mu = p.mu_nominal * mu_scale
    fz_f = m * GRAVITY_MPS2 * p.lr_m / p.wheelbase_m
    fz_r = m * GRAVITY_MPS2 * p.lf_m / p.wheelbase_m
    alpha_f = delta - torch.atan2(vy + p.lf_m * om, vx_slip)
    alpha_r = -torch.atan2(vy - p.lr_m * om, vx_slip)
    fyf = _lateral_force(alpha_f, p.cf_Npr, fz_f, mu, p, tire_model)
    fyr = _lateral_force(alpha_r, p.cr_Npr, fz_r, mu, p, tire_model)
    vfac = torch.tanh(vx / brake_vx_eps_mps)
    fx = (torch.where(F >= 0, F, F * vfac)
          - p.c_rr * m * GRAVITY_MPS2 * vfac - p.c_drag * vx * torch.abs(vx))
    return m, fx, fyf, fyr


def kart_xdot(x: Tensor, u: Tensor, p: KartParams, mu_scale: Tensor, mass_scale: Tensor,
              tire_model: str, v_blend: Tuple[float, float], kin_relax_s: float,
              brake_vx_eps_mps: float) -> Tensor:
    # x: [B,NX], u: [B,NU], mu_scale/mass_scale: [B] -> [B,NX]
    v_lo, v_hi = v_blend
    psi, vx, vy, om, delta = x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6]
    m, fx, fyf, fyr = _forces(x, p, mu_scale, mass_scale, tire_model, torch.clamp(vx, min=v_lo),
                                brake_vx_eps_mps)
    cd, sd = torch.cos(delta), torch.sin(delta)
    iz = p.yaw_inertia_kgm2 * mass_scale

    vx_dyn = (fx - fyf * sd) / m + vy * om
    vy_dyn = (fyr + fyf * cd) / m - vx * om
    om_dyn = (p.lf_m * fyf * cd - p.lr_m * fyr) / iz

    # kinematic bicycle: vy = lr/L vx tan(delta), omega = vx tan(delta)/L, differentiated
    ax = fx / m
    d_vx_tan = (ax * torch.tan(delta) + vx / (cd * cd) * u[:, 0]) / p.wheelbase_m
    tan_d = torch.tan(delta)
    vx_kin = ax
    vy_kin = p.lr_m * d_vx_tan + (p.lr_m * vx * tan_d / p.wheelbase_m - vy) / kin_relax_s
    om_kin = d_vx_tan + (vx * tan_d / p.wheelbase_m - om) / kin_relax_s

    w = torch.clamp((vx - v_lo) / (v_hi - v_lo), 0.0, 1.0)
    blend = lambda k, d: (1.0 - w) * k + w * d
    cp, sp = torch.cos(psi), torch.sin(psi)
    return torch.stack([
        vx * cp - vy * sp, vx * sp + vy * cp, om,
        blend(vx_kin, vx_dyn), blend(vy_kin, vy_dyn), blend(om_kin, om_dyn),
        u[:, 0], u[:, 1],
    ], dim=1)


def kart_step(x: Tensor, u: Tensor, p: KartParams, dt: float, mu_scale: Tensor, mass_scale: Tensor,
              tire_model: str, v_blend: Tuple[float, float], kin_relax_s: float,
              brake_vx_eps_mps: float) -> Tensor:
    # x: [B,NX], u: [B,NU] -> [B,NX]; RK4 with rate-limited u and clamped actuator states
    u = torch.stack([
        torch.minimum(torch.maximum(u[:, 0], -p.max_steer_rate_radps), p.max_steer_rate_radps),
        torch.minimum(torch.maximum(u[:, 1], -p.max_force_rate_Nps), p.max_force_rate_Nps),
    ], dim=1)
    f = lambda s: kart_xdot(s, u, p, mu_scale, mass_scale, tire_model, v_blend, kin_relax_s, brake_vx_eps_mps)
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    xn = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    delta = torch.minimum(torch.maximum(xn[:, 6], -p.max_steer_rad), p.max_steer_rad)
    F = torch.minimum(torch.maximum(xn[:, 7], -p.max_brake_force_N), p.max_drive_force_N)
    return torch.cat([xn[:, :6], delta[:, None], F[:, None]], dim=1)


def kart_accels(x: Tensor, p: KartParams, mu_scale: Tensor, mass_scale: Tensor,
                tire_model: str, brake_vx_eps_mps: float) -> Tuple[Tensor, Tensor]:
    # x: [B,NX] -> (a_long [B], a_lat [B]) body-frame, tire forces only (gravity/inertial coupling excluded)
    delta = x[:, 6]
    m, fx, fyf, fyr = _forces(x, p, mu_scale, mass_scale, tire_model, torch.clamp(x[:, 3], min=1e-3),
                                brake_vx_eps_mps)
    return (fx - fyf * torch.sin(delta)) / m, (fyr + fyf * torch.cos(delta)) / m
