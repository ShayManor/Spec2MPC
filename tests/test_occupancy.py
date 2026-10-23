import math

import pytest
import torch
from omegaconf import OmegaConf
from scipy.ndimage import distance_transform_edt as edt

from spec2mpc.sim.occupancy import GridSpec, cell_centers, clearance_grid, rasterize, sdf_lookup, signed_distance
from spec2mpc.sim.scenes import generate_scenes, pedestrians_at, true_clearance


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load("configs/sim/default.yaml")


@pytest.fixture(scope="module")
def spec(cfg):
    return GridSpec.from_cfg(cfg.grid)


@pytest.fixture(scope="module")
def scene_goto(cfg):
    return generate_scenes(8, cfg, torch.Generator().manual_seed(2), "goto")


@pytest.fixture(scope="module")
def scene_one_ped(cfg):
    sc = generate_scenes(2, cfg, torch.Generator().manual_seed(3), "goto")
    sc.obstacles_r.zero_()
    sc.ped_r.zero_()
    sc.ped_r[:, 0] = 0.3
    mid = sc.path.xy[:, sc.path.xy.shape[1] // 4]
    sc.ped_xy0[:, 0] = mid
    sc.ped_vel[:, 0] = 0.0
    sc.ped_vel[:, 0, 0] = 1.0
    return sc


def window(sc, spec):
    b = sc.batch_size
    return torch.zeros(b), spec.size_m * math.sqrt(2) + sc.path.length_m.max().item()


def test_spec_from_cfg(spec):
    assert spec.n == 200 and spec.frame_offsets_s == (0.0, 0.5, 1.0, 1.5)


def test_edt_matches_scipy():
    g = torch.Generator().manual_seed(0)
    occ = torch.rand(3, 2, 64, 64, generator=g) < 0.02
    ours = signed_distance(occ, 0.1, chunk_bytes=2.0e9)
    ref = torch.zeros_like(ours)
    for b in range(3):
        for f in range(2):
            o = occ[b, f].numpy()
            ref[b, f] = torch.from_numpy(0.1 * (edt(~o) - edt(o))).float()
    assert torch.allclose(ours, ref, atol=1e-4)


def test_edt_chunking_is_exact():
    g = torch.Generator().manual_seed(1)
    occ = torch.rand(5, 2, 32, 32, generator=g) < 0.03
    assert torch.equal(signed_distance(occ, 0.1, chunk_bytes=2.0e9), signed_distance(occ, 0.1, chunk_bytes=1.0))


def test_single_cell_distance():
    occ = torch.zeros(1, 1, 50, 50, dtype=torch.bool)
    occ[0, 0, 25, 25] = True
    sdf = signed_distance(occ, 0.1, chunk_bytes=2.0e9)
    assert math.isclose(sdf[0, 0, 25, 35].item(), 1.0, abs_tol=1e-5)


def test_empty_grid_is_far():
    assert (signed_distance(torch.zeros(1, 1, 20, 20, dtype=torch.bool), 0.1, chunk_bytes=2.0e9) > 1.9).all()


def test_cell_centers_layout(spec):
    c = cell_centers(spec, torch.tensor([[1.0, 2.0]]))
    assert c.shape == (1, spec.n, spec.n, 2)
    assert torch.allclose(c[0, 0, 0], torch.tensor([1.0 - 10 + 0.05, 2.0 - 10 + 0.05]), atol=1e-5)
    assert torch.allclose(c[0, 3, 7], torch.tensor([1.0 - 10 + 7.5 * 0.1, 2.0 - 10 + 3.5 * 0.1]), atol=1e-5)


def test_clearance_matches_true_clearance(scene_goto, spec):
    origin = scene_goto.path.xy[:, 0]
    t = torch.zeros(scene_goto.batch_size)
    clr = clearance_grid(scene_goto, origin, t, spec, window(scene_goto, spec))
    cells = cell_centers(spec, origin).reshape(scene_goto.batch_size, -1, 2)
    for f, off in enumerate(spec.frame_offsets_s):
        ref = true_clearance(scene_goto, cells, t + off).reshape(-1, spec.n, spec.n)
        assert (clr[:, f] - ref).abs().max() <= spec.res_m / 2


def test_window_is_exact_for_valid_scene(scene_goto, spec):
    origin = scene_goto.path.xy[:, 0]
    t = torch.zeros(scene_goto.batch_size)
    full = clearance_grid(scene_goto, origin, t, spec, window(scene_goto, spec))
    win = clearance_grid(scene_goto, origin, t, spec, (torch.zeros_like(t), spec.size_m * math.sqrt(2)))
    assert torch.allclose(full, win, atol=1e-4)


def test_rasterize_is_clearance_nonpositive(scene_goto, spec):
    origin = scene_goto.path.xy[:, 0]
    t = torch.zeros(scene_goto.batch_size)
    w = window(scene_goto, spec)
    occ = rasterize(scene_goto, origin, t, spec, w)
    assert occ.shape == (8, 4, spec.n, spec.n) and occ.dtype == torch.bool
    assert torch.equal(occ, clearance_grid(scene_goto, origin, t, spec, w) <= 0)
    assert occ.any() and (~occ).any()


def test_lookup_agrees_with_analytic_clearance(scene_goto, spec):
    b = scene_goto.batch_size
    origin = scene_goto.path.xy[:, 0]
    t = torch.zeros(b)
    w = window(scene_goto, spec)
    sdf = signed_distance(rasterize(scene_goto, origin, t, spec, w), spec.res_m, chunk_bytes=spec.edt_chunk_bytes)
    g = torch.Generator().manual_seed(5)
    xy = origin[:, None] + (torch.rand(b, 64, 2, generator=g) - 0.5) * 16.0
    frame_idx = torch.arange(64) % 4
    got = sdf_lookup(sdf, origin, spec, xy, frame_idx)
    err = 0.0
    for f in range(4):
        k = frame_idx == f
        ref = true_clearance(scene_goto, xy[:, k], t + spec.frame_offsets_s[f])
        near = ref.abs() < 2.0  # SDF sees only in-grid occupancy; points lie within 8 m of the centre, edge at 10 m
        err = max(err, (got[:, k] - ref)[near].abs().max().item())
    assert err <= spec.res_m


def test_lookup_exact_at_cell_center(spec):
    origin = torch.tensor([[3.0, -4.0]])
    sdf = torch.zeros(1, 4, spec.n, spec.n)
    sdf[0, 2, 17, 140] = 7.0
    xy = cell_centers(spec, origin)[:, 17, 140][:, None]  # [1, 1, 2]
    out = sdf_lookup(sdf, origin, spec, xy, torch.tensor([2]))
    assert math.isclose(out.item(), 7.0, abs_tol=1e-4)
    assert sdf_lookup(sdf, origin, spec, xy, torch.tensor([1])).abs().item() < 1e-6


def test_lookup_gradient_finite(scene_goto, spec):
    origin = scene_goto.path.xy[:, 0]
    t = torch.zeros(scene_goto.batch_size)
    sdf = signed_distance(rasterize(scene_goto, origin, t, spec, window(scene_goto, spec)), spec.res_m, chunk_bytes=spec.edt_chunk_bytes)
    xy = (origin[:, None] + torch.randn(8, 5, 2) * 3.0).requires_grad_(True)
    out = sdf_lookup(sdf, origin, spec, xy, torch.tensor([0, 1, 2, 3, 3]))
    out.sum().backward()
    assert torch.isfinite(xy.grad).all() and xy.grad.abs().sum() > 0


def test_moving_pedestrian_shifts_across_frames(scene_one_ped, spec):
    sc = scene_one_ped
    origin = sc.path.xy[:, sc.path.xy.shape[1] // 4]
    t = torch.zeros(sc.batch_size)
    occ = rasterize(sc, origin, t, spec, window(sc, spec))
    cells = cell_centers(spec, origin)
    ped_only = []
    for f in range(4):
        p = pedestrians_at(sc, t + spec.frame_offsets_s[f])[:, 0]
        d = (cells - p[:, None, None]).norm(dim=-1)
        m = d < 0.25
        assert occ[:, f][m].all()
        if f > 0:
            old = (cells - ped_only[-1][:, None, None]).norm(dim=-1) < 0.1
            assert not occ[:, f][old].any()
        ped_only.append(p)
    assert (ped_only[3] - ped_only[0]).norm(dim=-1).min() > 0.1


def test_edt_chunk_not_dividing_frames():
    g = torch.Generator().manual_seed(7)
    occ = torch.rand(3, 4, 32, 32, generator=g) < 0.03  # 12 frames
    per_frame = 32**3 * 4 * 2
    assert torch.equal(signed_distance(occ, 0.1, chunk_bytes=5 * per_frame), signed_distance(occ, 0.1, chunk_bytes=2.0e9))


def test_lap_window_wraps(cfg, spec):
    sc = generate_scenes(4, cfg, torch.Generator().manual_seed(4), "lap")
    idx = torch.round(sc.path.length_m / 0.25).long() - 2
    origin = sc.path.xy[torch.arange(4), idx]  # near s = L
    t = torch.zeros(4)
    s_c = sc.path.length_m - 0.5
    win = clearance_grid(sc, origin, t, spec, (s_c, spec.size_m * math.sqrt(2)))
    cells = cell_centers(spec, origin).reshape(4, -1, 2)
    ref = true_clearance(sc, cells, 0.0).reshape(-1, spec.n, spec.n)
    assert (win[:, 0] - ref).abs().max() <= spec.res_m / 2
