import math
from dataclasses import replace

import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.mpc.theta import THETA_NAMES, default_theta
from spec2mpc.robots.kart import RuntimeConfig, load_fixed_kart
from spec2mpc.sim.conditions import Conditions, sample_conditions
from spec2mpc.sim.rollout import episode_timeout_s, run_episodes, run_episodes_with_stats
from spec2mpc.sim.scenes import generate_scenes
from spec2mpc.sim.trajectory import DONE_COMPLETED, DONE_FAILURE, DONE_TIMEOUT

B = 4
TRACK_LEN_M = 60.0


@pytest.fixture(scope="module")
def sim_cfg():
    cfg = OmegaConf.load("configs/sim/default.yaml")
    cfg.scene.lap.length_m = [TRACK_LEN_M, TRACK_LEN_M]
    cfg.scene.lap.half_width_m = [3.0, 3.5]
    cfg.scene.n_laps = 1
    cfg.grid.res_m = 0.25  # n = 80 keeps the CPU EDT cheap
    return cfg


@pytest.fixture(scope="module")
def short_cfg(sim_cfg):
    # same config, every episode times out after 0.6 s
    cfg = OmegaConf.create(OmegaConf.to_container(sim_cfg))
    cfg.episode.timeout_cap_lap_s = 0.6
    return cfg


@pytest.fixture(scope="module")
def mpc_cfg():
    return OmegaConf.load("configs/mpc/kart.yaml")


@pytest.fixture(scope="module")
def kart_cfg():
    return OmegaConf.load("configs/robots/kart.yaml")


@pytest.fixture(scope="module")
def runtime_mod():
    return RuntimeConfig(control_rate_hz=20.0, solver_iters=10, perception_rate_hz=10.0)


@pytest.fixture(scope="module")
def evc(kart_cfg, runtime_mod):
    return load_fixed_kart(kart_cfg, "evc", runtime_mod).index(torch.zeros(B, dtype=torch.long))


@pytest.fixture(scope="module")
def scene(sim_cfg):
    return generate_scenes(B, sim_cfg, torch.Generator().manual_seed(3), "lap")


def _clean(robot):
    # perfect perception: no noise, bias, latency, flips, or prediction error; full view
    pr = robot.perception
    z = torch.zeros_like(pr.range_m)
    perception = replace(pr, range_m=z + 40.0, fov_rad=z + 2 * math.pi, loc_noise_std_m=z, loc_bias_m=z,
                         latency_s=z, p_false_occ=z, p_false_free=z, pred_error_m=z)
    return replace(robot, perception=perception)


def _nominal_cond(n: int) -> Conditions:
    one, zero = torch.ones(n), torch.zeros(n)
    return Conditions(friction_scale=one, mass_scale=one, latency_s=zero, init_offset=torch.zeros(n, 3),
                      seed=torch.zeros(n, dtype=torch.int64))


@pytest.fixture(scope="module")
def setup(evc, scene, mpc_cfg, short_cfg):
    cond = sample_conditions(B, short_cfg, torch.Generator().manual_seed(1))
    return evc, scene, default_theta(B, mpc_cfg), cond, short_cfg, mpc_cfg


def test_sample_conditions_ranges(sim_cfg):
    c = sample_conditions(512, sim_cfg, torch.Generator().manual_seed(0))
    cc = sim_cfg.conditions
    for v, r in ((c.friction_scale, cc.friction_scale), (c.mass_scale, cc.mass_scale), (c.latency_s, cc.latency_s),
                 (c.init_offset[:, 0], cc.init_lateral_m), (c.init_offset[:, 1], cc.init_heading_rad),
                 (c.init_offset[:, 2], cc.init_speed_mps)):
        assert v.min() >= r[0] and v.max() <= r[1]
        assert v.max() - v.min() > 0.8 * (r[1] - r[0])
    assert c.seed.dtype == torch.int64 and c.seed.unique().numel() == 512
    d = sample_conditions(512, sim_cfg, torch.Generator().manual_seed(0))
    assert torch.equal(c.init_offset, d.init_offset)


def test_episode_timeout(evc, scene, sim_cfg):
    t = episode_timeout_s(scene, evc.params, sim_cfg)
    expect = scene.path.length_m * scene.n_laps / (0.15 * evc.params.v_max_mps)
    assert torch.allclose(t, torch.clamp(expect, max=300.0))
    goto = replace(scene, task=torch.ones_like(scene.task), path=replace(scene.path, length_m=scene.path.length_m * 100))
    assert torch.allclose(episode_timeout_s(goto, evc.params, sim_cfg), torch.full((B,), 120.0))


def test_bit_reproducible(setup):
    a = run_episodes(*setup, gen=torch.Generator().manual_seed(0))
    b = run_episodes(*setup, gen=torch.Generator().manual_seed(0))
    assert torch.equal(a.x, b.x) and torch.equal(a.solver_ok, b.solver_ok) and torch.equal(a.u, b.u)
    # contracts
    T = a.x.shape[1]
    assert a.t_s.shape == (B, T) and a.u.shape == (B, T, 2) and a.dt_s == pytest.approx(0.05)
    assert a.valid[:, 0].all()
    assert (a.done_reason == DONE_TIMEOUT).all() and a.completion_time_s.isnan().all()
    assert (a.p_drive_W >= 0).all()
    assert (a.s[:, -1] > a.s[:, 0]).all()  # the karts move forward
    c, st = run_episodes_with_stats(*setup, gen=torch.Generator().manual_seed(1))
    assert not torch.equal(a.x, c.x)  # perception noise comes from gen
    assert st.cost_mean.shape == (B,) and torch.isfinite(st.cost_mean).all()
    assert (st.grad_norm_mean > 0).all() and torch.isfinite(st.grad_norm_mean).all()


def test_non_integer_perception_ratio(kart_cfg, scene, mpc_cfg, short_cfg):
    # 25 Hz control with 10 Hz perception (sampled in data generation): updates every 2 or 3 control steps
    rt = RuntimeConfig(control_rate_hz=25.0, solver_iters=10, perception_rate_hz=10.0)
    robot = load_fixed_kart(kart_cfg, "evc", rt).index(torch.zeros(B, dtype=torch.long))
    cond = sample_conditions(B, short_cfg, torch.Generator().manual_seed(1))
    tr = run_episodes(robot, scene, default_theta(B, mpc_cfg), cond, short_cfg, mpc_cfg,
                      gen=torch.Generator().manual_seed(0))
    assert tr.dt_s == pytest.approx(0.04) and torch.isfinite(tr.x).all()
    assert (tr.s[:, -1] > tr.s[:, 0]).all()


def test_latency_shifts_response(evc, scene, mpc_cfg, short_cfg):
    robot = _clean(evc)
    robot = replace(robot, params=replace(robot.params, actuator_delay_s=torch.zeros(B)))
    cond = _nominal_cond(B)
    cond_lat = replace(cond, latency_s=torch.full((B,), 0.06))
    th = default_theta(B, mpc_cfg)
    g = lambda: torch.Generator().manual_seed(0)
    a = run_episodes(robot, scene, th, cond, short_cfg, mpc_cfg, gen=g())
    b = run_episodes(robot, scene, th, cond_lat, short_cfg, mpc_cfg, gen=g())
    first = lambda tr: (tr.x[..., 7].abs() > 1e-3).float().argmax(dim=1)  # first control step with F != 0
    assert (first(a) == 1).all()  # command at t = 0 shows up by t = dt
    assert (first(b) == 2).all()  # 60 ms delay: still F = 0 at t = 50 ms
    assert torch.equal(a.u[:, 0], b.u[:, 0])  # same first decision, only its effect is delayed


def test_terminated_envs_frozen(evc, scene, mpc_cfg, short_cfg):
    # per-env timeouts differ through v_max: 60 m / (0.15 v_max) capped at 0.6 s, so some envs stop earlier
    v_max = torch.tensor([1500.0, 700.0, 17.5, 3000.0])
    robot = replace(evc, params=replace(evc.params, v_max_mps=v_max))
    tr = run_episodes(robot, scene, default_theta(B, mpc_cfg), _nominal_cond(B), short_cfg, mpc_cfg,
                      gen=torch.Generator().manual_seed(0))
    n_valid = tr.valid.sum(dim=1)
    timeout = episode_timeout_s(scene, robot.params, short_cfg)
    assert torch.equal(n_valid, torch.ceil(timeout / tr.dt_s - 1e-6).long() + 1)
    assert n_valid.unique().numel() > 1
    for b in range(B):
        k = int(n_valid[b])
        assert tr.valid[b, :k].all() and not tr.valid[b, k:].any()  # prefix mask
        assert (tr.x[b, k:] == tr.x[b, k - 1]).all()  # frozen after termination
        assert (tr.s[b, k:] == tr.s[b, k - 1]).all()


def test_goto_completes_at_finish_line_with_lateral_offset(evc, sim_cfg, mpc_cfg, short_cfg):
    # 1.5 m beside the centreline within goal_tol of the path end: finished, as a lap is at s >= L, even though the
    # kart is more than goal_tol from the goal point (it cannot close that offset without driving past the end)
    scene = generate_scenes(B, sim_cfg, torch.Generator().manual_seed(3), "goto")
    scene = replace(scene, path=replace(scene.path, length_m=scene.goal_tol_m.clone()))
    cond = replace(_nominal_cond(B), init_offset=torch.tensor([[1.5, 0.0, 0.0]]).repeat(B, 1))
    tr = run_episodes(_clean(evc), scene, default_theta(B, mpc_cfg), cond, short_cfg, mpc_cfg,
                      gen=torch.Generator().manual_seed(0))
    assert (tr.done_reason == DONE_COMPLETED).all()
    assert (tr.completion_time_s == 0.0).all()


@pytest.mark.slow
def test_default_theta_completes_nominal_lap(evc, scene, mpc_cfg, sim_cfg):
    robot = _clean(evc)
    tr = run_episodes(robot, scene, default_theta(B, mpc_cfg), _nominal_cond(B), sim_cfg, mpc_cfg,
                      gen=torch.Generator().manual_seed(0))
    assert (tr.done_reason == DONE_COMPLETED).all()
    assert (tr.laps_done == 1).all()
    assert torch.isfinite(tr.completion_time_s).all()
    last = tr.valid.sum(dim=1) - 1
    assert torch.allclose(tr.completion_time_s, tr.t_s[torch.arange(B), last])
    assert (tr.s[torch.arange(B), last] >= TRACK_LEN_M).all()


@pytest.mark.slow
def test_default_theta_completes_lap_with_max_false_occupancy(evc, scene, mpc_cfg, sim_cfg, kart_cfg):
    # false-occupied flips at the top of the configured range, scaled to the production grid's phantom density per m^2
    prod_res_m = float(OmegaConf.load("configs/sim/default.yaml").grid.res_m)
    p = float(kart_cfg.perception.p_false_occ[1]) * (float(sim_cfg.grid.res_m) / prod_res_m) ** 2
    robot = _clean(evc)
    robot = replace(robot, perception=replace(robot.perception, p_false_occ=torch.full((B,), p)))
    tr = run_episodes(robot, scene, default_theta(B, mpc_cfg), _nominal_cond(B), sim_cfg, mpc_cfg,
                      gen=torch.Generator().manual_seed(0))
    # full view at the range maximum: flicker slows one lap past its timeout (ideas8), none crash or touch a wall
    assert (tr.done_reason == DONE_COMPLETED).sum() >= B - 1
    assert (tr.done_reason != DONE_FAILURE).all()
    assert ((tr.clearance_true >= 0) | ~tr.valid).all()


@pytest.mark.slow
def test_zero_progress_does_not_complete(evc, scene, mpc_cfg, sim_cfg):
    robot = _clean(evc)
    th = default_theta(B, mpc_cfg)
    v = th.values.clone()
    v[:, THETA_NAMES.index("w_progress")] = -3.0
    v[:, THETA_NAMES.index("w_speed_ref")] = -3.0
    tr = run_episodes(robot, scene, replace(th, values=v), _nominal_cond(B), sim_cfg, mpc_cfg,
                      gen=torch.Generator().manual_seed(0))
    assert (tr.done_reason != DONE_COMPLETED).all()
    assert tr.completion_time_s.isnan().all()
    assert (tr.done_reason != DONE_FAILURE).all()


def test_slow_control_rate_follows_plan_stages(kart_cfg, scene, mpc_cfg, short_cfg, monkeypatch):
    # 10 Hz control with 0.05 s MPC stages: each control period applies stage 0, then stage 1, and the next
    # solve starts from a warm start shifted by both stages
    import spec2mpc.sim.rollout as ro
    rt = RuntimeConfig(control_rate_hz=10.0, solver_iters=10, perception_rate_hz=10.0)
    robot = _clean(load_fixed_kart(kart_cfg, "evc", rt).index(torch.zeros(B, dtype=torch.long)))
    robot = replace(robot, params=replace(robot.params, actuator_delay_s=torch.zeros(B)))
    solves, applied = [], []
    solve, step = ro.MPCC.solve, ro.kart_step

    def rec_solve(self, x0, state, *a, **k):
        z_in = state.z.clone()
        res = solve(self, x0, state, *a, **k)
        solves.append((z_in, res, state.z.clone()))
        return res

    def rec_step(x, u, *a, **k):
        applied.append(u.clone())
        return step(x, u, *a, **k)

    monkeypatch.setattr(ro.MPCC, "solve", rec_solve)
    monkeypatch.setattr(ro, "kart_step", rec_step)
    run_episodes(robot, scene, default_theta(B, mpc_cfg), _nominal_cond(B), short_cfg, mpc_cfg,
                 gen=torch.Generator().manual_seed(0))
    n_sub, per_stage = 10, 5
    _, res, z_out = solves[0]
    assert res.ok.all()
    for i in range(n_sub):
        want = res.u0 if i < per_stage else z_out[:, 0, :2]
        assert torch.equal(applied[i], want)
    assert not torch.equal(z_out[:, 0, :2], res.u0)  # the plan changes between stages, so the test can fail
    assert torch.equal(solves[1][0], torch.cat([z_out[:, 1:], z_out[:, -1:]], dim=1))
