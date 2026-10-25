from dataclasses import dataclass, fields
from typing import Tuple, Union

import torch
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.batch import Batched
from spec2mpc.mpc.adapter import MPCWeights
from spec2mpc.robots.kart import KartParams
from spec2mpc.robots.kart_dynamics import NX, kart_accels, kart_step
from spec2mpc.sim.occupancy import GridSpec, sdf_lookup
from spec2mpc.sim.paths import RefPath, path_lookup, project_to_path


@dataclass
class MPCCState(Batched):
    z: Tensor  # [B, N, 3] warm start (delta_rate, F_rate, v_s), SI units
    s: Tensor  # [B] progress at the last solve (projection of x_hat onto the path)
    u_prev: Tensor  # [B, 2] last returned u0, reference for the stage-0 jerk term
    m1: Tensor  # [B, N, 3] Adam first moment, shifted with z
    m2: Tensor  # [B, N, 3] Adam second moment, shifted with z
    n_steps: Tensor  # [B] Adam steps taken so far (bias correction)

    @classmethod
    def init(cls, B: int, N: int, device: Union[str, torch.device]) -> "MPCCState":
        z = lambda *s: torch.zeros(*s, device=device)
        return cls(z(B, N, 3), z(B), z(B, 2), z(B, N, 3), z(B, N, 3), z(B))


@dataclass
class SolveResult(Batched):
    u0: Tensor  # [B, 2]
    v_s0: Tensor  # [B]
    ok: Tensor  # [B] bool
    cost: Tensor  # [B]
    grad_norm: Tensor  # [B], gradient norm in normalized decision units at the returned iterate (NaN if none)
    pred_x: Tensor  # [B, N+1, NX]


class MPCC:
    def __init__(self, mpc_cfg: DictConfig, tire_model: str):
        self.N = int(mpc_cfg.horizon)
        self.dt = float(mpc_cfg.dt_s)
        self.v_blend = tuple(float(v) for v in mpc_cfg.v_blend_mps)
        self.kin_relax_s = float(mpc_cfg.kin_relax_s)
        self.brake_vx_eps_mps = float(mpc_cfg.brake_vx_eps_mps)
        self.band_m = float(mpc_cfg.clearance_band_m)
        sc = mpc_cfg.solver
        self.lr = float(sc.lr)
        self.betas = (float(sc.betas[0]), float(sc.betas[1]))
        self.eps = float(sc.eps)
        self.project_window_m = float(sc.project_window_m)
        self.persist_moments = bool(sc.persist_moments)
        # RK4 at dt_s is unstable for the stiff low-speed tire mode (lambda ~ -(cf + cr) / (m vx)); substeps fix it
        self.substeps = int(sc.model_substeps)
        self.tire_model = tire_model
        # fuses the eager rollout (~10k tiny ops per pass); compiling takes minutes, so only worth it for long runs
        self._rollout_fn = torch.compile(self._rollout, dynamic=False) if bool(sc.compile) else self._rollout

    def _rollout(self, u: Tensor, x0: Tensor, p: KartParams) -> Tensor:
        # u [B, N, 2], x0 [B, NX] -> x [B, N+1, NX]; nominal friction and mass (the controller does not know xi)
        ones = torch.ones(x0.shape[0], dtype=x0.dtype, device=x0.device)
        h = self.dt / self.substeps
        xs = [x0]
        for k in range(self.N):
            x = xs[-1]
            for _ in range(self.substeps):
                x = kart_step(x, u[:, k], p, h, ones, ones, self.tire_model, self.v_blend,
                              self.kin_relax_s, self.brake_vx_eps_mps)
            xs.append(x)
        return torch.stack(xs, dim=1)

    def _project(self, z: Tensor, x0: Tensor, p: KartParams) -> Tensor:
        # z [B, N, 3], x0 [B, NX] -> [B, N, 3]: rates within rate limits, integrated actuator states within
        # their bounds (so the model's clamp in kart_step never cuts the gradient), v_s in [0, v_max]
        cols = []
        for j, (rl, lo, hi) in enumerate((
            (p.max_steer_rate_radps, -p.max_steer_rad, p.max_steer_rad),
            (p.max_force_rate_Nps, -p.max_brake_force_N, p.max_drive_force_N),
        )):
            rl, lo, hi = rl[:, None], lo[:, None], hi[:, None]
            a0 = x0[:, 6 + j : 7 + j]  # [B, 1]
            rate = torch.minimum(torch.maximum(z[..., j], -rl), rl)
            traj = a0 + self.dt * torch.cumsum(rate, dim=1)  # [B, N]
            clip = torch.minimum(torch.maximum(traj, lo), hi)
            prev = torch.cat([a0, clip[:, :-1]], dim=1)
            fixed = torch.minimum(torch.maximum((clip - prev) / self.dt, -rl), rl)
            changed = clip != traj
            changed = changed | torch.cat([torch.zeros_like(changed[:, :1]), changed[:, :-1]], dim=1)
            cols.append(torch.where(changed, fixed, rate))
        cols.append(torch.minimum(z[..., 2].clamp(min=0.0), p.v_max_mps[:, None]))
        return torch.stack(cols, dim=-1)

    def _cost(self, z: Tensor, x0: Tensor, s0: Tensor, u_prev: Tensor, w: MPCWeights, p: KartParams,
              p_rep: KartParams, path: RefPath, sdf: Tensor, sdf_origin: Tensor, frame_idx: Tensor,
              spec: GridSpec) -> Tuple[Tensor, Tensor]:
        # z [B, N, 3], x0 [B, NX], s0 [B], u_prev [B, 2] -> (cost [B], x [B, N+1, NX])
        B, N, dt = z.shape[0], self.N, self.dt
        x_all = self._rollout_fn(z[..., :2], x0, p)  # [B, N+1, NX]
        x = x_all[:, 1:]  # [B, N, NX] stage k cost is on x_{k+1}
        v_s = z[..., 2]
        s = s0[:, None] + dt * torch.cumsum(v_s, dim=1)  # [B, N] = s_{k+1}
        ref, hd, _ = path_lookup(path, s)
        dx, dy = x[..., 0] - ref[..., 0], x[..., 1] - ref[..., 1]
        sh, ch = torch.sin(hd), torch.cos(hd)
        e_c = sh * dx - ch * dy
        e_l = -ch * dx - sh * dy

        onesn = torch.ones(B * N, dtype=x0.dtype, device=x0.device)
        a_long, a_lat = kart_accels(x.reshape(B * N, NX), p_rep, onesn, onesn, self.tire_model, self.brake_vx_eps_mps)
        a_long, a_lat = a_long.reshape(B, N), a_lat.reshape(B, N)

        u = z[..., :2]
        rl = torch.stack([p.max_steer_rate_radps, p.max_force_rate_Nps], dim=-1)[:, None]  # [B, 1, 2]
        du = (u - torch.cat([u_prev[:, None], u[:, :-1]], dim=1)) / rl
        d = sdf_lookup(sdf, sdf_origin, spec, x[..., :2], frame_idx)  # [B, N]
        r_need = (p.radius_m + w.margin_m)[:, None]

        c = lambda f: f[:, None]
        stage = (c(w.q_contour) * e_c ** 2 + c(w.q_lag) * e_l ** 2 - c(w.q_progress) * v_s * dt
                 + c(w.q_speed_ref) * (x[..., 3] - c(w.v_ref_mps)) ** 2
                 + c(w.r_accel) * ((a_long / c(p.a_long_max_mps2)) ** 2 + (a_lat / c(p.a_lat_max_mps2)) ** 2)
                 + c(w.r_jerk) * (du ** 2).sum(-1)
                 + c(w.r_effort) * ((x[..., 6] / c(p.max_steer_rad)) ** 2 + (x[..., 7] / c(p.max_drive_force_N)) ** 2)
                 + c(w.q_clearance) * torch.relu(r_need + self.band_m - d) ** 2)
        g = torch.stack([x[..., 3] - c(w.v_cap_mps), a_lat.abs() - c(w.a_lat_cap_mps2),
                         a_long.abs() - c(w.a_long_cap_mps2), r_need - d], dim=-1).relu()  # [B, N, 4]
        stage = stage + c(w.rho_slack) * (g + g ** 2).sum(-1)
        return stage.sum(dim=1), x_all

    def solve(self, x_hat: Tensor, state: MPCCState, w: MPCWeights, p: KartParams, path: RefPath, sdf: Tensor,
              sdf_origin: Tensor, sdf_age_s: float, spec: GridSpec, n_iters: int) -> SolveResult:
        # x_hat [B, NX], sdf [B, F, n, n], sdf_origin [B, 2] -> SolveResult; updates state in place
        B, N, dev = x_hat.shape[0], self.N, x_hat.device
        with torch.no_grad():
            bad = ~torch.isfinite(x_hat).all(dim=1)
            for f in fields(w):
                bad = bad | ~torch.isfinite(getattr(w, f.name))
            x0 = torch.where(bad[:, None], torch.zeros_like(x_hat), x_hat)
            w = MPCWeights(**{f.name: torch.where(bad, torch.zeros_like(getattr(w, f.name)), getattr(w, f.name))
                              for f in fields(w)})
            s_prev = torch.where(torch.isfinite(state.s), state.s, torch.zeros_like(state.s))
            s0 = project_to_path(path, x0[:, :2], s_prev, self.project_window_m)
            s0 = torch.where(bad | ~torch.isfinite(s0), s_prev, s0)
            u_prev = state.u_prev

            offs = torch.tensor(spec.frame_offsets_s, dtype=x0.dtype, device=dev)
            t_k = sdf_age_s + self.dt * torch.arange(1, N + 1, dtype=x0.dtype, device=dev)
            frame_idx = (t_k[:, None] - offs[None]).abs().argmin(dim=1)  # [N]
            p_rep = p.index(torch.arange(B, device=dev).repeat_interleave(N))
            scale = torch.stack([p.max_steer_rate_radps, p.max_force_rate_Nps, p.v_max_mps], dim=-1)[:, None]  # [B,1,3]

            z = torch.where(torch.isfinite(state.z), state.z, torch.zeros_like(state.z))
            zn = self._project(z, x0, p) / scale
            args = (x0, s0, u_prev, w, p, p_rep, path, sdf, sdf_origin, frame_idx, spec)
            if self.persist_moments:
                fin = lambda t: torch.where(torch.isfinite(t), t, torch.zeros_like(t))
                m1, m2, t0 = fin(state.m1), fin(state.m2), fin(state.n_steps)[:, None, None]
            else:
                m1, m2, t0 = torch.zeros_like(zn), torch.zeros_like(zn), torch.zeros(B, 1, 1, dtype=zn.dtype, device=dev)
            b1, b2 = self.betas
            best_zn = zn.clone()
            best_c = torch.full((B,), float("inf"), dtype=x0.dtype, device=dev)
            best_x = x0[:, None].expand(B, N + 1, NX).clone()
            grad_norm = torch.full((B,), float("nan"), dtype=x0.dtype, device=dev)
            diverged = torch.zeros(B, dtype=torch.bool, device=dev)

            for it in range(n_iters + 1):
                zr = zn.detach().requires_grad_(True)
                with torch.enable_grad():
                    cost, x_pred = self._cost(zr * scale, *args)
                    (g,) = torch.autograd.grad(torch.where(bad, torch.zeros_like(cost), cost).sum(), zr)
                cost, x_pred = cost.detach(), x_pred.detach()
                # any non-finite cost, iterate or gradient during the solve marks the env failed, not only at the end
                diverged = diverged | ~torch.isfinite(cost) | ~torch.isfinite(zn).flatten(1).all(dim=1) \
                    | ~torch.isfinite(g).flatten(1).all(dim=1)
                better = cost < best_c  # NaN never improves
                best_c = torch.where(better, cost, best_c)
                best_zn = torch.where(better[:, None, None], zn, best_zn)
                best_x = torch.where(better[:, None, None], x_pred, best_x)
                grad_norm = torch.where(better, g.flatten(1).norm(dim=1), grad_norm)  # at the returned iterate
                if it == n_iters:
                    break
                m1 = b1 * m1 + (1 - b1) * g
                m2 = b2 * m2 + (1 - b2) * g * g
                t = t0 + (it + 1)
                step = (m1 / (1 - b1 ** t)) / ((m2 / (1 - b2 ** t)).sqrt() + self.eps)
                # a diverged env keeps its last finite iterate, so NaN never reaches path_lookup's gather indices
                zn = torch.where(diverged[:, None, None], zn, self._project((zn - self.lr * step) * scale, x0, p) / scale)

            z = best_zn * scale
            ok = torch.isfinite(best_c) & torch.isfinite(z).flatten(1).all(dim=1) & ~bad & ~diverged
            z = torch.where(ok[:, None, None], z, torch.zeros_like(z))
            u0 = z[:, 0, :2].clone()
            v_s0 = z[:, 0, 2].clone()
            shift = lambda a: torch.where(ok[:, None, None], torch.cat([a[:, 1:], a[:, -1:]], dim=1), torch.zeros_like(a))
            state.z = shift(z)
            state.m1, state.m2 = shift(m1), shift(m2)
            state.n_steps = torch.where(ok, t0[:, 0, 0] + n_iters, torch.zeros_like(t0[:, 0, 0]))
            state.s = s0
            state.u_prev = u0.clone()
            return SolveResult(u0=u0, v_s0=v_s0, ok=ok, cost=best_c, grad_norm=grad_norm, pred_x=best_x)
