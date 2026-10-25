import math
from dataclasses import dataclass
from typing import Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.batch import Batched
from spec2mpc.mpc.adapter import adapt
from spec2mpc.mpc.mpcc import MPCC, MPCCState
from spec2mpc.mpc.theta import ThetaBatch
from spec2mpc.robots.kart import KartParams, Robot, RuntimeConfig
from spec2mpc.robots.kart_dynamics import NU, NX, kart_accels, kart_step
from spec2mpc.sim.conditions import Conditions
from spec2mpc.sim.occupancy import GridSpec
from spec2mpc.sim.paths import project_global, project_to_path
from spec2mpc.sim.perception import PerceptionState, perceive, sample_perception_noise
from spec2mpc.sim.scenes import TASK_LAP, Scene, true_clearance
from spec2mpc.sim.trajectory import DONE_COMPLETED, DONE_FAILURE, DONE_TIMEOUT, Trajectory

_EPS_T_S = 1e-6


@dataclass
class SolverStats(Batched):
    # per-episode means over valid steps whose solve was ok (NaN if there are none)
    cost_mean: Tensor  # [B] SolveResult.cost
    grad_norm_mean: Tensor  # [B] SolveResult.grad_norm


def episode_timeout_s(scene: Scene, p: KartParams, sim_cfg: DictConfig) -> Tensor:
    # scene, p [B] -> [B]: length * n_laps / (frac * v_max), capped per task
    e = sim_cfg.episode
    lap = scene.task == TASK_LAP
    cap = torch.where(lap, torch.full_like(p.v_max_mps, float(e.timeout_cap_lap_s)),
                      torch.full_like(p.v_max_mps, float(e.timeout_cap_goto_s)))
    t = scene.path.length_m * scene.n_laps.to(p.v_max_mps.dtype) / (float(e.timeout_v_frac) * p.v_max_mps)
    return torch.minimum(t, cap)


def _initial_state(scene: Scene, cond: Conditions) -> Tensor:
    # -> x0 [B, NX]: on the path at s = 0 with (lateral, heading, speed) offsets; delta = F = 0
    path = scene.path
    h0 = path.heading[:, 0]
    nrm = torch.stack([-torch.sin(h0), torch.cos(h0)], dim=-1)  # [B, 2]
    x = torch.zeros(h0.shape[0], NX, dtype=path.xy.dtype, device=path.xy.device)
    x[:, :2] = path.xy[:, 0] + cond.init_offset[:, 0:1] * nrm
    x[:, 2] = h0 + cond.init_offset[:, 1]
    x[:, 3] = cond.init_offset[:, 2]
    return x


def run_episodes(robot: Robot, scene: Scene, theta: ThetaBatch, cond: Conditions, sim_cfg: DictConfig,
                 mpc_cfg: DictConfig, gen: torch.Generator, tire_model: str = "linear") -> Trajectory:
    # robot/scene/theta/cond [B] -> Trajectory [B, T] at the control rate; tire_model is the plant's and the MPC's
    return run_episodes_with_stats(robot, scene, theta, cond, sim_cfg, mpc_cfg, gen, tire_model)[0]


def run_episodes_with_stats(robot: Robot, scene: Scene, theta: ThetaBatch, cond: Conditions, sim_cfg: DictConfig,
                            mpc_cfg: DictConfig, gen: torch.Generator,
                            tire_model: str = "linear") -> Tuple[Trajectory, SolverStats]:
    # same as run_episodes, plus per-episode solver stats [B]
    p, rt = robot.params, robot.runtime
    assert isinstance(rt, RuntimeConfig), "rollout needs one RuntimeConfig for the whole batch"
    B, dev = theta.batch_size, theta.values.device
    plant_dt = float(sim_cfg.plant_dt_s)
    n_sub = round(1.0 / (plant_dt * rt.control_rate_hz))
    assert n_sub >= 1 and math.isclose(n_sub * plant_dt * rt.control_rate_hz, 1.0), "control rate must divide 1/plant_dt"
    cr, pr = float(rt.control_rate_hz), float(rt.perception_rate_hz)
    assert 0.0 < pr <= cr, "perception rate must not exceed the control rate"
    dt = n_sub * plant_dt
    sub_per_stage = round(float(mpc_cfg.dt_s) / plant_dt)  # plant substeps per MPC stage
    stages_per_step = max(1, round(dt / float(mpc_cfg.dt_s)))  # MPC stages elapsed per control step
    e = sim_cfg.episode
    spec = GridSpec.from_cfg(sim_cfg.grid)
    path = scene.path
    length = path.length_m
    closed = path.closed
    lap = scene.task == TASK_LAP
    v_blend = tuple(float(v) for v in mpc_cfg.v_blend_mps)
    kin_relax_s, brake_eps = float(mpc_cfg.kin_relax_s), float(mpc_cfg.brake_vx_eps_mps)
    mu, ms = cond.friction_scale, cond.mass_scale
    max_fail = int(sim_cfg.max_consecutive_solver_failures)
    vy_max, om_max = float(sim_cfg.spin_vy_max_mps), float(sim_cfg.spin_omega_max_radps)
    win = float(e.progress_window_m)
    check_every = int(e.done_check_every)

    timeout = episode_timeout_s(scene, p, sim_cfg)  # [B]
    T = int(math.ceil(float(timeout.max()) / dt - _EPS_T_S)) + 1
    target_s = scene.n_laps.to(length.dtype) * length  # [B]

    # perception and controller
    noise = sample_perception_noise(robot.perception, gen)
    max_lat = int(torch.round(robot.perception.latency_s.max() * rt.perception_rate_hz))
    pseed = int(torch.randint(0, 2**62, (1,), generator=gen))
    pstate = PerceptionState.init(B, spec, max_lat, pseed, dev)
    mpc = MPCC(mpc_cfg, tire_model)
    w = adapt(theta, p, mpc_cfg)
    mstate = MPCCState.init(B, mpc.N, dev)

    # actuator delay: ring of past commands at plant-step resolution, read per env at its own delay
    d_steps = torch.round((p.actuator_delay_s + cond.latency_s) / plant_dt).long()  # [B]
    ring_len = int(d_steps.max()) + 1
    ring = torch.zeros(B, ring_len, NU, device=dev)
    ar = torch.arange(B, device=dev)
    g_step = 0

    x = _initial_state(scene, cond)
    s_w = project_global(path, x[:, None, :2])[:, 0]  # [B] wrapped / clamped progress
    s_unw = torch.where(closed, torch.remainder(s_w + 0.5 * length, length) - 0.5 * length, s_w)
    mstate.s = s_w.clone()

    f32 = dict(dtype=x.dtype, device=dev)
    xs = torch.zeros(B, T, NX, **f32)
    us = torch.zeros(B, T, NU, **f32)
    ss, a_lo, a_la, clr, pdr = (torch.zeros(B, T, **f32) for _ in range(5))
    oks = torch.zeros(B, T, dtype=torch.bool, device=dev)
    valid = torch.zeros(B, T, dtype=torch.bool, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    reason = torch.full((B,), DONE_TIMEOUT, dtype=torch.long, device=dev)
    t_done = torch.full((B,), float("nan"), **f32)
    consec = torch.zeros(B, dtype=torch.long, device=dev)
    loc_err = torch.zeros(B, 2, **f32)
    sdf = origin = None
    age = 0
    n_steps = T
    cost_sum, gn_sum, n_ok = (torch.zeros(B, **f32) for _ in range(3))

    for k in range(T):
        t = k * dt
        if k > 0:
            s_new = project_to_path(path, x[:, :2], s_w, win)
            ds = s_new - s_w
            ds = torch.where(closed, torch.remainder(ds + 0.5 * length, length) - 0.5 * length, ds)
            s_unw = torch.where(done, s_unw, s_unw + ds)  # terminated envs keep their final progress
            s_w = torch.where(done, s_w, s_new)

        al, at = kart_accels(x, p, mu, ms, tire_model, brake_eps)
        xs[:, k], ss[:, k], a_lo[:, k], a_la[:, k] = x, s_unw, al, at
        clr[:, k] = true_clearance(scene, x[:, None, :2], t)[:, 0] - p.radius_m
        pdr[:, k] = (x[:, 7].clamp(min=0.0) * x[:, 3]).clamp(min=0.0)
        valid[:, k] = ~done

        # perception updates on control steps where floor(k * pr / cr) increments (every cr/pr steps when integral)
        if k == 0 or math.floor(k * pr / cr) != math.floor((k - 1) * pr / cr):
            x_hat, sdf, origin = perceive(scene, x, t, robot.perception, noise, pstate, spec, s_w,
                                          rt.perception_rate_hz)
            loc_err = x_hat[:, :2] - x[:, :2]
            age = 0
        else:
            # between updates the estimate keeps the last update's localization error
            x_hat = torch.cat([x[:, :2] + loc_err, x[:, 2:]], dim=1)
        res = mpc.solve(x_hat, mstate, w, p, path, sdf, origin, age * dt, spec, rt.solver_iters)
        age += 1
        us[:, k], oks[:, k] = res.u0, res.ok
        consec = torch.where(res.ok, torch.zeros_like(consec), consec + 1)
        use = valid[:, k] & res.ok & torch.isfinite(res.cost) & torch.isfinite(res.grad_norm)
        cost_sum = cost_sum + torch.where(use, res.cost, torch.zeros_like(res.cost))
        gn_sum = gn_sum + torch.where(use, res.grad_norm, torch.zeros_like(res.grad_norm))
        n_ok = n_ok + use.to(n_ok.dtype)

        # goto finishes when progress is within goal_tol of the end (a finish line, like laps); the kart cannot
        # close a lateral offset at the path end without driving past it, so a radius test parks it beside the goal
        completed = s_unw >= target_s - scene.goal_tol_m
        failed = (consec >= max_fail) | (x[:, 4].abs() > vy_max) | (x[:, 5].abs() > om_max)
        timed_out = t >= timeout - _EPS_T_S
        new = ~done & (completed | failed | timed_out)
        reason = torch.where(new, torch.where(completed, DONE_COMPLETED, torch.where(failed, DONE_FAILURE, DONE_TIMEOUT)),
                             reason)
        t_done = torch.where(new & completed, torch.full_like(t_done, t), t_done)
        done = done | new
        if k == T - 1:
            break
        if (k + 1) % check_every == 0 and bool(done.all()):
            n_steps = k + 1
            break

        active = ~done[:, None]
        # a control period longer than one MPC stage follows the plan stage by stage; the solve already shifted
        # the warm start by one stage, so z[:, j - 1] is stage j
        for i in range(n_sub):
            j = min(i // sub_per_stage, mstate.z.shape[1])
            u_cmd = res.u0 if j == 0 else torch.where(res.ok[:, None], mstate.z[:, j - 1, :2], res.u0)
            ring[:, g_step % ring_len] = u_cmd
            u_app = ring[ar, (g_step - d_steps) % ring_len]  # [B, NU]
            xn = kart_step(x, u_app, p, plant_dt, mu, ms, tire_model, v_blend, kin_relax_s, brake_eps)
            x = torch.where(active, xn, x)
            g_step += 1
        # shift the warm start by the remaining stages that elapsed during the control period
        for _ in range(stages_per_step - 1):
            mstate.z, mstate.m1, mstate.m2 = [torch.cat([a[:, 1:], a[:, -1:]], dim=1)
                                              for a in (mstate.z, mstate.m1, mstate.m2)]

    sl = slice(0, n_steps)
    t_s = (torch.arange(n_steps, **f32) * dt)[None].expand(B, n_steps).contiguous()
    s_last = s_unw
    laps = torch.where(lap, torch.minimum(torch.floor(s_last / length).clamp(min=0).long(), scene.n_laps),
                       (reason == DONE_COMPLETED).long())
    traj = Trajectory(
        t_s=t_s, x=xs[:, sl], u=us[:, sl], s=ss[:, sl], a_long=a_lo[:, sl], a_lat=a_la[:, sl],
        clearance_true=clr[:, sl], solver_ok=oks[:, sl], valid=valid[:, sl], p_drive_W=pdr[:, sl],
        done_reason=reason, completion_time_s=t_done, laps_done=laps, dt_s=dt,
    )
    return traj, SolverStats(cost_mean=cost_sum / n_ok, grad_norm_mean=gn_sum / n_ok)  # 0/0 -> NaN
