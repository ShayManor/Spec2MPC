import math

import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.mpc.adapter import adapt
from spec2mpc.mpc.theta import (D_THETA, THETA_NAMES, ThetaBatch, default_theta, denormalize_theta, normalize_theta,
                                sobol_theta, theta_bounds)
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


SOBOL_8_SEED3 = torch.tensor([  # sobol_theta(8, kart mpc cfg, seed=3), captured before theta_bounds existed
    [-2.5892093181610107, -1.6528382301330566, 1.1951112747192383, -2.624485969543457, -0.7476286888122559, -2.817474603652954, -2.115769386291504, -0.9687595367431641, -1.3790624141693115, 0.34530895948410034, 0.7210382223129272, 0.45027920603752136, 1.6152719259262085, 0.6622426509857178],
    [2.3852968215942383, 1.8848786354064941, -0.7700269222259521, 0.8127045631408691, 0.46015334129333496, 0.5161027908325195, 0.46412014961242676, 2.3339943885803223, 2.129758358001709, 0.8931469917297363, 0.5206310749053955, 0.7495288252830505, 0.7960830926895142, 0.24224069714546204],
    [1.071401596069336, -0.37051868438720703, 2.5029640197753906, -0.20032858848571777, 1.7189335823059082, 1.7137975692749023, -0.057674407958984375, 1.4671964645385742, 1.457108497619629, 0.7042310237884521, 0.14162692427635193, 0.9357327222824097, 1.4828741550445557, 0.8273471593856812],
    [-0.8915691375732422, 0.5890092849731445, -2.8341495990753174, 1.6372013092041016, -1.6189582347869873, -0.918123722076416, 1.7810029983520508, -2.7269599437713623, -2.20625376701355, 0.15453585982322693, 0.8201106190681458, 0.2928582429885864, 0.5092313885688782, 0.310911625623703],
    [-0.09612202644348145, -1.471958875656128, -0.7322719097137451, 2.8195900917053223, -2.5064504146575928, -1.970564365386963, -1.4739121198654175, 0.627094030380249, 0.5144190788269043, 0.8370667099952698, 0.31060636043548584, 0.13737022876739502, 1.1688343286514282, 0.12148505449295044],
    [0.36668825149536133, 0.9407243728637695, 0.3069930076599121, -1.2447466850280762, 2.4072070121765137, 1.3623676300048828, 2.7989234924316406, -1.8751447200775146, -2.765212297439575, 0.5147666335105896, 0.9890698790550232, 0.8374369740486145, 0.34251803159713745, 0.5415412783622742],
    [1.68369722366333, -2.8015007972717285, -1.6648503541946411, 0.731968879699707, 1.4121389389038086, 2.3725643157958984, -2.409959316253662, -0.3059091567993164, -0.3997519016265869, 0.32411545515060425, 0.5520643591880798, 0.5666379332542419, 1.936453104019165, 0.43132254481315613],
    [-1.7910151481628418, 2.283083438873291, 1.9965739250183105, -1.9319034814834595, -1.1253955364227295, -0.2587149143218994, 1.1099481582641602, 1.6828570365905762, 2.6491661071777344, 0.6480319499969482, 0.3516644835472107, 0.4369538426399231, 0.9556550979614258, 0.9479283690452576],
], dtype=torch.float32)


def test_theta_normalize_roundtrip(mpc_cfg):
    v = sobol_theta(64, mpc_cfg, seed=0).values
    assert torch.allclose(denormalize_theta(normalize_theta(v, mpc_cfg), mpc_cfg), v, atol=1e-5)
    lo, hi = theta_bounds(mpc_cfg)
    assert lo.shape == hi.shape == (D_THETA,) and lo.dtype == hi.dtype == torch.float32
    assert torch.equal(denormalize_theta(torch.full((1, D_THETA), 5.0), mpc_cfg)[0], hi)
    assert torch.equal(denormalize_theta(torch.full((1, D_THETA), -5.0), mpc_cfg)[0], lo)
    u = normalize_theta(v, mpc_cfg)
    assert (u >= -1).all() and (u <= 1).all()


def test_sobol_unchanged_by_refactor(mpc_cfg):  # pinned against values captured before the edit
    assert torch.equal(sobol_theta(8, mpc_cfg, seed=3).values, SOBOL_8_SEED3)
