import math

import pytest
import torch

from spec2mpc.robots.kart import load_fixed_kart, sample_karts
from spec2mpc.robots.kart_dynamics import NU, NX, kart_accels, kart_step, kart_xdot

VB = (1.0, 3.0)
KR = 0.1
BE = 0.1
G = 9.81


@pytest.fixture
def p1(kart_cfg, runtime):
    return load_fixed_kart(kart_cfg, "evc", runtime).params


@pytest.fixture
def p8(kart_cfg, runtime, gen):
    return sample_karts(8, kart_cfg, runtime, gen).params


def state(vx=0.0, delta=0.0, F=0.0, B=1):
    x = torch.zeros(B, NX)
    x[:, 3], x[:, 6], x[:, 7] = vx, delta, F
    return x


def ones(B=1):
    return torch.ones(B)


def test_dims():
    assert (NX, NU) == (8, 2)


def test_straight_stays_straight(p1):
    vx = 8.0
    F = float(p1.c_rr[0] * p1.mass_kg[0] * G + p1.c_drag[0] * vx**2)
    x = state(vx=vx, F=F)
    for _ in range(200):
        x = kart_step(x, torch.zeros(1, NU), p1, 0.01, ones(), ones(), "linear", VB, KR, BE)
    assert abs(x[0, 1]) < 1e-5 and abs(x[0, 2]) < 1e-5


def test_low_speed_cornering_matches_kinematic(p1):
    vx = 2.0
    F = float(p1.c_rr[0] * p1.mass_kg[0] * G + p1.c_drag[0] * vx**2)
    x = state(vx=vx, delta=0.2, F=F)
    for _ in range(300):
        x = kart_step(x, torch.zeros(1, NU), p1, 0.01, ones(), ones(), "linear", VB, KR, BE)
    assert torch.isclose(x[0, 5], torch.tensor(vx * math.tan(0.2) / float(p1.wheelbase_m[0])), rtol=0.05)


def test_zero_speed_is_finite(p1):
    x = kart_step(state(vx=0.0), torch.tensor([[1.0, 500.0]]), p1, 0.01, ones(), ones(), "pacejka", VB, KR, BE)
    assert torch.isfinite(x).all()


@pytest.mark.parametrize("tire", ["linear", "pacejka"])
def test_lateral_force_saturates_low_friction(p1, tire):
    mu_scale = torch.tensor([0.5])
    F = float(p1.c_rr[0] * p1.mass_kg[0] * G + p1.c_drag[0] * 12.0**2)
    x = state(vx=12.0, delta=0.3, F=F)
    peak = 0.0
    for _ in range(100):
        x = kart_step(x, torch.zeros(1, NU), p1, 0.01, mu_scale, ones(), tire, VB, KR, BE)
        _, a_lat = kart_accels(x, p1, mu_scale, ones(), tire, BE)
        peak = max(peak, float(a_lat.abs()))
    assert 0.9 * 0.5 * float(p1.mu_nominal[0]) * G <= peak <= 0.5 * float(p1.mu_nominal[0]) * G * 1.02


def test_batched_equals_individual(p8):
    g = torch.Generator().manual_seed(1)
    x8 = torch.rand(8, NX, generator=g) * 0.2
    x8[:, 3] = torch.linspace(0.0, 14.0, 8)
    x8[:, 6] = 0.1
    x8[:, 7] = 200.0
    u8 = torch.randn(8, NU, generator=g)
    mu = 0.6 + 0.4 * torch.rand(8, generator=g)
    ms = 0.9 + 0.2 * torch.rand(8, generator=g)
    xb = kart_step(x8, u8, p8, 0.01, mu, ms, "pacejka", VB, KR, BE)
    for i in range(8):
        xi = kart_step(x8[i:i + 1], u8[i:i + 1], p8.index(slice(i, i + 1)), 0.01, mu[i:i + 1], ms[i:i + 1], "pacejka", VB, KR, BE)
        assert torch.allclose(xb[i:i + 1], xi, atol=1e-6)


def test_actuator_states_clamped(p1):
    x = kart_step(state(delta=0.0), torch.tensor([[1e3, 1e6]]), p1, 1.0, ones(), ones(), "linear", VB, KR, BE)
    assert x[0, 6] <= p1.max_steer_rad[0] and x[0, 7] <= p1.max_drive_force_N[0]
    x = kart_step(state(delta=0.0), torch.tensor([[-1e3, -1e6]]), p1, 1.0, ones(), ones(), "linear", VB, KR, BE)
    assert x[0, 6] >= -p1.max_steer_rad[0] and x[0, 7] >= -p1.max_brake_force_N[0]


def test_input_rate_clamped(p1):
    x = kart_step(state(), torch.tensor([[1e3, 1e6]]), p1, 0.001, ones(), ones(), "linear", VB, KR, BE)
    assert x[0, 6] <= p1.max_steer_rate_radps[0] * 0.001 * 1.001
    assert x[0, 7] <= p1.max_force_rate_Nps[0] * 0.001 * 1.001


@pytest.mark.parametrize("tire", ["linear", "pacejka"])
def test_gradient_wrt_u_finite(p1, tire):
    x = state(vx=6.0, delta=0.1, F=100.0)
    u = torch.tensor([[0.5, 300.0]], requires_grad=True)
    out = kart_step(x, u, p1, 0.02, ones(), ones(), tire, VB, KR, BE)
    (g,) = torch.autograd.grad(out.sum(), u)
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def _run(x, p, n, dt=0.01, u=None, tire="linear"):
    u = torch.zeros(1, NU) if u is None else u
    lo = 0.0
    for _ in range(n):
        x = kart_step(x, u, p, dt, ones(), ones(), tire, VB, KR, BE)
        lo = min(lo, float(x[0, 3]))
    return x, lo


def test_kinematic_yaw_rate_at_vx_half(p1):
    vx = 0.5
    F = float(p1.c_rr[0] * p1.mass_kg[0] * G + p1.c_drag[0] * vx**2)
    x, _ = _run(state(vx=vx, delta=0.2, F=F), p1, 300)
    assert torch.isclose(x[0, 5], torch.tensor(vx * math.tan(0.2) / float(p1.wheelbase_m[0])), rtol=0.05)


def test_stop_from_spin(p1):
    x = state(vx=0.5)
    x[0, 4], x[0, 5] = 0.3, 1.0
    x, _ = _run(x, p1, 500)
    assert abs(x[0, 4]) < 0.01 and abs(x[0, 5]) < 0.01


def test_brake_at_rest_does_not_reverse(p1):
    x, lo = _run(state(F=-300.0), p1, 100)
    assert lo > -0.05


def test_brake_from_speed_stops(p1):
    x, lo = _run(state(vx=10.0, F=-float(p1.max_brake_force_N[0])), p1, 1500)
    assert 0.0 <= float(x[0, 3]) <= 0.2 and lo > -0.05
