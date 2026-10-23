import dataclasses

import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.features.metrics import METRIC_NAMES, episode_metrics
from spec2mpc.features.phi import PHI_NAMES, phi
from spec2mpc.robots.kart import load_fixed_kart
from spec2mpc.sim.paths import path_lookup
from spec2mpc.sim.scenes import generate_scenes
from spec2mpc.sim.trajectory import DONE_COMPLETED, DONE_TIMEOUT, Trajectory

DT = 0.1
T = 200
M_FLOOR = 0.2
G = 9.81


@pytest.fixture(scope="module")
def scene():
    cfg = OmegaConf.load("configs/sim/default.yaml")
    return generate_scenes(2, cfg, torch.Generator().manual_seed(3), "lap")


@pytest.fixture(scope="module")
def kart(runtime_module):
    cfg = OmegaConf.load("configs/robots/kart.yaml")
    r = load_fixed_kart(cfg, "evc", runtime_module)
    return r.params.index(torch.zeros(2, dtype=torch.long))


@pytest.fixture(scope="module")
def runtime_module():
    from spec2mpc.robots.kart import RuntimeConfig

    return RuntimeConfig(control_rate_hz=10.0, solver_iters=10, perception_rate_hz=10.0)


def make_traj(B: int = 2, v: float = 5.0) -> Trajectory:
    t = (torch.arange(T) * DT).expand(B, T).clone()
    x = torch.zeros(B, T, 8)
    x[..., 0] = v * t
    x[..., 3] = v
    return Trajectory(
        t_s=t, x=x, u=torch.zeros(B, T, 2), s=v * t, a_long=torch.zeros(B, T),
        a_lat=torch.zeros(B, T), clearance_true=torch.full((B, T), 2.0),
        solver_ok=torch.ones(B, T, dtype=torch.bool), valid=torch.ones(B, T, dtype=torch.bool),
        p_drive_W=torch.full((B, T), 360.0), done_reason=torch.full((B,), DONE_TIMEOUT),
        completion_time_s=torch.full((B,), float("nan")), laps_done=torch.zeros(B, dtype=torch.long),
        dt_s=DT,
    )


def test_metric_keys(kart, scene):
    m = episode_metrics(make_traj(), kart, scene, M_FLOOR)
    assert tuple(m) == METRIC_NAMES
    assert all(v.shape == (2,) for v in m.values())


def test_constant_speed_metrics(kart, scene):
    m = episode_metrics(make_traj(), kart, scene, M_FLOOR)
    assert m["speed_mean_mps"][0].item() == pytest.approx(5.0)
    assert m["speed_max_mps"][0].item() == pytest.approx(5.0)
    assert m["a_lat_rms_g"][0].item() == pytest.approx(0.0, abs=1e-6)
    assert m["jerk_rms_mps3"][0].item() == pytest.approx(0.0, abs=1e-6)
    assert m["boundary_violations"][0].item() == 0
    assert m["min_clearance_m"][0].item() == pytest.approx(2.0)
    assert m["energy_Wh"][0].item() == pytest.approx(360.0 * T * DT / 3600)
    assert m["completion"][0].item() == 0 and torch.isnan(m["lap_time_s"][0])


def test_boundary_violation_counted_once_per_entry(kart, scene):
    tr = make_traj()
    tr.clearance_true[0, 50:60] = -0.5
    m = episode_metrics(tr, kart, scene, M_FLOOR)
    assert m["boundary_violations"].tolist() == [1, 0]
    assert m["violation_integral_ms"][0].item() == pytest.approx(0.5 * 10 * DT)
    assert m["min_clearance_m"][0].item() == pytest.approx(-0.5)
    assert m["time_below_margin_s"][0].item() == pytest.approx(10 * DT)
    tr.clearance_true[0, 80:85] = -0.1
    assert episode_metrics(tr, kart, scene, M_FLOOR)["boundary_violations"][0].item() == 2


def test_solver_failure_and_invalid_steps_ignored(kart, scene):
    tr = make_traj()
    tr.solver_ok[0, 10] = False
    tr.valid[1, 100:] = False
    tr.clearance_true[1, 150:] = -3.0
    tr.x[1, 150:, 3] = 99.0
    m = episode_metrics(tr, kart, scene, M_FLOOR)
    assert m["solver_failures"].tolist() == [1, 0]
    assert m["boundary_violations"][1].item() == 0
    assert m["speed_max_mps"][1].item() == pytest.approx(5.0)
    assert m["energy_Wh"][1].item() == pytest.approx(360.0 * 100 * DT / 3600)


def test_lat_accel_in_g(kart, scene):
    tr = make_traj()
    tr.a_lat[:] = G
    m = episode_metrics(tr, kart, scene, M_FLOOR)
    assert m["a_lat_rms_g"][0].item() == pytest.approx(1.0)
    assert m["a_lat_max_g"][0].item() == pytest.approx(1.0)


def test_lap_time_multi_lap(kart, scene):
    L = scene.path.length_m
    v = (L / 5.0).max().item()  # fastest: lap in 5 s; others take longer
    sc = dataclasses.replace(scene, n_laps=torch.full((2,), 2))
    tr = make_traj(v=v)
    tr.done_reason[:] = DONE_COMPLETED
    m = episode_metrics(tr, kart, sc, M_FLOOR)
    expected = L / v
    assert torch.allclose(m["lap_time_s"], expected, atol=DT + 1e-4)
    assert (m["completion"] == 1).all()
    assert (m["progress_frac"] <= 1.0).all()


def test_phi_fixed_order_and_finite(kart, scene):
    tr = make_traj()
    tr.clearance_true[0, 50:60] = -0.5
    tr.solver_ok[1, 3] = False
    m = episode_metrics(tr, kart, scene, M_FLOOR)
    f = phi(m, kart, scene, torch.full((2,), T * DT))
    assert f.shape == (2, len(PHI_NAMES)) and torch.isfinite(f).all()
    assert f[0, PHI_NAMES.index("speed_mean")].item() == pytest.approx(5.0 / kart.v_max_mps[0].item())
    assert f[0, PHI_NAMES.index("lap_time")].item() == 1.0


def _poison(tr: Trajectory, b: int, sl: slice) -> None:
    tr.valid[b, sl] = False
    tr.x[b, sl] = float("nan")
    tr.x[b, sl, 3] = float("inf")
    tr.a_long[b, sl] = float("inf")
    tr.a_lat[b, sl] = float("nan")
    tr.p_drive_W[b, sl] = float("nan")
    tr.clearance_true[b, sl] = float("-inf")
    tr.s[b, sl] = float("nan")
    tr.solver_ok[b, sl] = False


def test_nan_inf_at_invalid_steps_ignored(kart, scene):
    def build(poison: bool) -> Trajectory:
        tr = make_traj()
        tr.a_lat[:] = torch.sin(tr.t_s) * 3.0
        tr.a_long[:] = torch.cos(tr.t_s) * 2.0
        tr.valid[0, 120:] = False
        tr.valid[1, 60:] = False
        if poison:
            _poison(tr, 0, slice(120, T))
            _poison(tr, 1, slice(60, T))
        return tr

    clean = episode_metrics(build(False), kart, scene, M_FLOOR)
    dirty = episode_metrics(build(True), kart, scene, M_FLOOR)
    for k in METRIC_NAMES:
        if k in ("lap_time_s", "completion_time_s"):
            assert torch.isnan(dirty[k]).equal(torch.isnan(clean[k]))
            continue
        assert torch.isfinite(dirty[k]).all(), k
        assert torch.allclose(dirty[k], clean[k], atol=1e-5), k


def test_percentiles_and_rms_ignore_invalid(kart, scene):
    tr = make_traj()
    tr.a_lat[:] = G
    tr.valid[:, 100:] = False
    tr.a_lat[:, 100:] = 100 * G
    m = episode_metrics(tr, kart, scene, M_FLOOR)
    for k in ("a_lat_rms_g", "a_lat_p99_g", "a_lat_max_g"):
        assert m[k][0].item() == pytest.approx(1.0), k


def test_zero_valid_steps_safe(kart, scene):
    tr = make_traj()
    tr.valid[:] = False
    tr.done_reason[:] = DONE_COMPLETED
    m = episode_metrics(tr, kart, scene, M_FLOOR)
    for k in METRIC_NAMES:
        if k in ("lap_time_s", "completion_time_s"):
            assert torch.isnan(m[k]).all(), k
        else:
            assert (m[k] == 0).all(), k
    f = phi(m, kart, scene, torch.full((2,), T * DT))
    assert torch.isfinite(f).all() and f.abs().max() < 1e3


def test_goto_scene_lap_time_nan(kart):
    cfg = OmegaConf.load("configs/sim/default.yaml")
    sc = generate_scenes(2, cfg, torch.Generator().manual_seed(4), "goto")
    tr = make_traj()
    tr.s[:] = 1e6
    m = episode_metrics(tr, kart, sc, M_FLOOR)
    assert torch.isnan(m["lap_time_s"]).all()
    f = phi(m, kart, sc, torch.full((2,), T * DT))
    assert (f[:, PHI_NAMES.index("lap_time")] == 1.0).all()
