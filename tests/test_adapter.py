import math

import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.mpc.adapter import adapt
from spec2mpc.mpc.theta import D_THETA, THETA_NAMES, ThetaBatch, default_theta, sobol_theta
from spec2mpc.robots.kart import sample_karts


@pytest.fixture
def mpc_cfg():
    return OmegaConf.load("configs/mpc/kart.yaml")


@pytest.fixture
def p4(kart_cfg, runtime, gen):
    return sample_karts(4, kart_cfg, runtime, gen).params


def test_floors_cannot_be_crossed(p4, mpc_cfg):
    th = ThetaBatch(values=torch.full((4, D_THETA), -100.0))
    w = adapt(th, p4, mpc_cfg)
    assert (w.margin_m >= mpc_cfg.floors.m_floor_m).all()
    assert (w.rho_slack >= mpc_cfg.floors.rho_slack_min).all()
    assert (w.v_cap_mps >= mpc_cfg.floors.c_vmax * p4.v_max_mps - 1e-6).all()
    assert (w.a_lat_cap_mps2 >= mpc_cfg.floors.c_alat * p4.a_lat_max_mps2 - 1e-6).all()
    assert (w.a_long_cap_mps2 >= mpc_cfg.floors.c_along * p4.a_long_max_mps2 - 1e-6).all()
    assert (w.v_ref_mps >= mpc_cfg.v_ref_min * p4.v_max_mps - 1e-6).all()


def test_caps_never_exceed_physical_limits(p4, mpc_cfg):
    w = adapt(ThetaBatch(values=torch.full((4, D_THETA), 100.0)), p4, mpc_cfg)
    assert (w.v_cap_mps <= p4.v_max_mps).all() and (w.a_lat_cap_mps2 <= p4.a_lat_max_mps2).all()
    assert (w.a_long_cap_mps2 <= p4.a_long_max_mps2).all()
    assert (w.margin_m <= mpc_cfg.max.c_margin_m).all()


def test_delta_clipped_to_pm3(p4, mpc_cfg):
    w = adapt(ThetaBatch(values=torch.full((4, D_THETA), 10.0)), p4, mpc_cfg)
    assert torch.allclose(w.q_contour, torch.full((4,), mpc_cfg.w_nom.w_contour * math.exp(3)))


def test_default_theta_maps_to_nominal(p4, mpc_cfg):
    w = adapt(default_theta(4, mpc_cfg), p4, mpc_cfg)
    assert torch.allclose(w.q_lag, torch.full((4,), mpc_cfg.w_nom.w_lag))


def test_sobol_in_bounds_and_deterministic(mpc_cfg):
    a, b = sobol_theta(256, mpc_cfg, seed=1), sobol_theta(256, mpc_cfg, seed=1)
    assert torch.equal(a.values, b.values)
    assert (a.values[:, :9].abs() <= 3).all()
    f, m = mpc_cfg.floors, mpc_cfg.max
    bounds = {"c_vmax": (f.c_vmax, m.c_vmax), "c_alat": (f.c_alat, m.c_alat), "c_along": (f.c_along, m.c_along),
              "c_margin": (f.m_floor_m, m.c_margin_m), "v_ref": (mpc_cfg.v_ref_min, 1.0)}
    for name, (lo, hi) in bounds.items():
        col = a.get(name)
        assert (col >= lo).all() and (col <= hi).all()


def test_get_by_name(mpc_cfg):
    th = default_theta(3, mpc_cfg)
    assert D_THETA == len(THETA_NAMES) == 14
    assert torch.allclose(th.get("c_margin"), torch.full((3,), mpc_cfg.constraints.nominal.c_margin))


def test_adapt_differentiable(p4, mpc_cfg):
    th = default_theta(4, mpc_cfg)
    th.values.requires_grad_(True)
    w = adapt(th, p4, mpc_cfg)
    (w.q_contour.sum() + w.v_cap_mps.sum() + w.margin_m.sum()).backward()
    assert th.values.grad is not None and th.values.grad[:, 1].abs().min() > 0
    assert th.values.grad[:, THETA_NAMES.index("c_vmax")].abs().min() > 0
