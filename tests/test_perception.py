import math

import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.robots.kart import PerceptionProfile
from spec2mpc.robots.kart_dynamics import NX
from spec2mpc.sim.occupancy import GridSpec, cell_centers, rasterize, sdf_lookup, signed_distance
from spec2mpc.sim.perception import PerceptionNoise, PerceptionState, perceive, sample_perception_noise, window_half_m
from spec2mpc.sim.scenes import generate_scenes

RATE_HZ = 10.0
B = 4


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load("configs/sim/default.yaml")


@pytest.fixture(scope="module")
def spec(cfg):
    return GridSpec.from_cfg(cfg.grid)


@pytest.fixture(scope="module")
def scene(cfg):
    return generate_scenes(B, cfg, torch.Generator().manual_seed(2), "goto")


@pytest.fixture(scope="module")
def x_true(scene):
    x = torch.zeros(scene.batch_size, NX)
    x[:, :2] = scene.path.xy[:, 5]
    x[:, 2] = 0.3
    x[:, 3] = 2.0
    return x


def profile(b: int = B, **kw) -> PerceptionProfile:
    d = dict(range_m=1e3, fov_rad=2 * math.pi, loc_noise_std_m=0.0, loc_bias_m=0.0, latency_s=0.0, p_false_occ=0.0, p_false_free=0.0, pred_error_m=0.0)
    d.update(kw)
    has_pred = d.pop("has_prediction", True)
    t = {k: torch.full((b,), float(v)) for k, v in d.items()}
    return PerceptionProfile(**t, has_prediction=torch.full((b,), has_pred, dtype=torch.bool))


def run(scene, x, prof, spec, t_s=0.0, seed=0, state=None, max_lat=0, gen_seed=1):
    b = scene.batch_size
    noise = sample_perception_noise(prof, torch.Generator().manual_seed(gen_seed))
    state = state or PerceptionState.init(b, spec, max_lat, seed, torch.device("cpu"))
    s0 = torch.zeros(b)
    return perceive(scene, x, t_s, prof, noise, state, spec, s0, RATE_HZ), noise, state


def test_perfect_profile_equals_truth(scene, x_true, spec):
    (x_hat, sdf, origin), _, _ = run(scene, x_true, profile(), spec)
    half = window_half_m(spec)
    truth = signed_distance(rasterize(scene, x_true[:, :2], torch.zeros(B), spec, (torch.zeros(B), half)), spec.res_m, chunk_bytes=spec.edt_chunk_bytes)
    assert torch.equal(sdf, truth)
    assert torch.equal(x_hat, x_true) and torch.equal(origin, x_true[:, :2])
    assert sdf.dtype == torch.float32 and sdf.shape == (B, len(spec.frame_offsets_s), spec.n, spec.n)


def test_no_prediction_repeats_frame0(scene, x_true, spec):
    (_, sdf, _), _, _ = run(scene, x_true, profile(has_prediction=False, p_false_occ=0.05), spec)
    for f in range(1, sdf.shape[1]):
        assert torch.equal(sdf[:, f], sdf[:, 0])


def test_flip_rate(scene, x_true, spec):
    empty = scene.index(slice(0, B))
    empty = type(empty)(**{**empty.__dict__, "path": type(empty.path)(**{**empty.path.__dict__, "half_width_m": torch.full((B,), 1e3)})})
    empty.obstacles_r.zero_()
    empty.ped_r.zero_()
    (_, sdf, _), _, _ = run(empty, x_true, profile(range_m=8.0, p_false_occ=0.05), spec)
    n = spec.n
    c = (torch.arange(n) + 0.5) * spec.res_m - spec.size_m / 2
    r = torch.sqrt(c[None, :] ** 2 + c[:, None] ** 2)
    inside = r <= 8.0
    frac = (sdf[:, 0] <= 0)[:, inside].float().mean()
    assert 0.04 <= frac <= 0.06


def test_latency_delays_frames(scene, x_true, spec):
    sc = scene.index(slice(0, 2))
    sc.obstacles_r.zero_()
    sc.ped_r.zero_()
    sc.ped_r[:, 0] = 0.3
    sc.ped_xy0[:, 0] = sc.path.xy[:, 10]
    sc.ped_vel[:, 0] = 0.0
    sc.ped_vel[:, 0, 0] = 3.0
    x = x_true[:2]
    ref_state = PerceptionState.init(2, spec, 0, 0, torch.device("cpu"))
    lat_state = PerceptionState.init(2, spec, 3, 0, torch.device("cpu"))
    ref_p, lat_p = profile(2), profile(2, latency_s=0.2)
    refs, lats = [], []
    for k in range(5):
        refs.append(run(sc, x, ref_p, spec, t_s=0.1 * k, state=ref_state)[0][1])
        lats.append(run(sc, x, lat_p, spec, t_s=0.1 * k, state=lat_state, max_lat=3)[0][1])
    assert not torch.equal(refs[0], refs[2])
    for k in range(5):
        assert torch.equal(lats[k], refs[max(k - 2, 0)])


def test_zero_range_all_free(scene, x_true, spec):
    (_, sdf, _), _, _ = run(scene, x_true, profile(range_m=0.0), spec)
    assert (sdf > 0).all()


def test_fov_limits_visibility(scene, x_true, spec):
    (_, sdf_full, _), _, _ = run(scene, x_true, profile(), spec)
    (_, sdf_narrow, _), _, _ = run(scene, x_true, profile(fov_rad=0.5), spec)
    assert (sdf_narrow[:, 0] <= 0).sum() < (sdf_full[:, 0] <= 0).sum()


def test_full_false_free_runs(scene, x_true, spec):
    (_, sdf, _), _, _ = run(scene, x_true, profile(p_false_free=1.0), spec)
    assert torch.isfinite(sdf).all() and (sdf > 0).all()


def test_localization_shift(scene, x_true, spec):
    prof = profile(loc_bias_m=0.5)
    (x_hat, sdf, origin), noise, _ = run(scene, x_true, prof, spec)
    assert torch.allclose(noise.bias_xy.norm(dim=-1), torch.full((B,), 0.5), atol=1e-5)
    assert torch.allclose(x_hat[:, :2], x_true[:, :2] + noise.bias_xy) and torch.equal(origin, x_hat[:, :2])
    assert torch.equal(x_hat[:, 2:], x_true[:, 2:])
    half = window_half_m(spec)
    truth = signed_distance(rasterize(scene, x_true[:, :2], torch.zeros(B), spec, (torch.zeros(B), half)), spec.res_m, chunk_bytes=spec.edt_chunk_bytes)
    assert torch.equal(sdf, truth)


def test_reproducible_given_seed(scene, x_true, spec):
    prof = profile(loc_noise_std_m=0.1, p_false_occ=0.02, p_false_free=0.1, pred_error_m=0.5, range_m=9.0, fov_rad=3.0)
    a = run(scene, x_true, prof, spec, seed=7)[0]
    b = run(scene, x_true, prof, spec, seed=7)[0]
    c = run(scene, x_true, prof, spec, seed=8)[0]
    assert all(torch.equal(p, q) for p, q in zip(a, b))
    assert not torch.equal(a[1], c[1])


def test_latency_over_max_raises(scene, x_true, spec):
    with pytest.raises(ValueError):
        run(scene, x_true, profile(latency_s=0.5), spec, max_lat=2)


def test_no_occupied_outside_view(scene, x_true, spec):
    prof = profile(range_m=6.0, fov_rad=1.5, p_false_occ=0.2)
    (x_hat, sdf, origin), _, _ = run(scene, x_true, prof, spec)
    rel = cell_centers(spec, torch.zeros(B, 2))
    dpsi = torch.remainder(torch.atan2(rel[..., 1], rel[..., 0]) - x_hat[:, 2, None, None] + math.pi, 2 * math.pi) - math.pi
    outside = (rel.norm(dim=-1) > 6.0) | (dpsi.abs() > 0.75 + 1e-4)
    # outside the view every cell is free; sdf there is positive only if no occupied cell touches it, so check cells themselves
    assert (sdf[:, 0][outside] > 0).all()
    assert (sdf[:, 0] <= 0).any()


def test_pred_error_changes_future_frames_only(scene, x_true, spec):
    sc = scene.index(slice(0, 2))
    sc.obstacles_r.zero_()
    sc.ped_r.zero_()
    sc.ped_r[:, 0] = 0.3
    sc.ped_xy0[:, 0] = sc.path.xy[:, 10]
    sc.ped_vel[:, 0] = 0.0
    sc.ped_vel[:, 0, 0] = 1.0
    x = x_true[:2]
    (_, a, _), _, _ = run(sc, x, profile(2, pred_error_m=0.0), spec)
    (_, c, _), _, _ = run(sc, x, profile(2, pred_error_m=1.0), spec)
    assert torch.equal(a[:, 0], c[:, 0])
    assert not torch.equal(a[:, 1:], c[:, 1:])


def test_localization_shift_direction(scene, spec):
    sc = scene.index(slice(0, 1))
    sc.path.half_width_m.fill_(1e3)
    sc.ped_r.zero_()
    sc.obstacles_r.zero_()
    sc.obstacles_r[:, 0] = 0.3
    sc.obstacles_xy[:, 0] = torch.tensor([3.0, 0.0])
    x = torch.zeros(1, NX)
    prof = profile(1)
    noise = PerceptionNoise(bias_xy=torch.tensor([[0.5, 0.0]]), seed=torch.zeros(1, dtype=torch.int64))
    state = PerceptionState.init(1, spec, 0, 0, torch.device("cpu"))
    x_hat, sdf, origin = perceive(sc, x, 0.0, prof, noise, state, spec, torch.zeros(1), RATE_HZ)
    assert torch.allclose(origin, torch.tensor([[0.5, 0.0]]))
    at = lambda px: sdf_lookup(sdf, origin, spec, torch.tensor([[[px, 0.0]]]), torch.tensor([0]))[0, 0]
    assert at(3.5) < 0 and at(3.0) > 0
