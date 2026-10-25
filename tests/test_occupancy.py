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
    # the geometry tests below are written for 0.1 m cells; the default config uses 0.2 m
    return GridSpec.from_cfg(OmegaConf.merge(cfg.grid, {"res_m": 0.1}))


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


def test_spec_from_cfg(cfg):
    spec = GridSpec.from_cfg(cfg.grid)
    assert spec.n == 100 and spec.frame_offsets_s == (0.0, 0.5, 1.0, 1.5)


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


def test_lookup_nan_query_isolated(spec):
    # a NaN query (diverged MPC prediction) returns NaN there; grid_sample's x86 CPU backward segfaults on NaN grids
    origin = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
    sdf = torch.randn(2, 4, spec.n, spec.n, generator=torch.Generator().manual_seed(4))
    xy = torch.tensor([[[0.5, 0.5], [float("nan"), 0.0]], [[1.3, 2.7], [0.4, 1.1]]], requires_grad=True)
    out = sdf_lookup(sdf, origin, spec, xy, torch.tensor([0, 2]))
    assert torch.isnan(out[0, 1]) and torch.isfinite(out[0, 0]) and torch.isfinite(out[1]).all()
    out[1].sum().backward()
    assert torch.isfinite(xy.grad).all() and xy.grad[1].abs().sum() > 0


# ---- equivalence against the pre-optimisation implementation (copied verbatim) ----

from dataclasses import replace

from spec2mpc.sim.occupancy import _circles
from spec2mpc.sim.paths import path_ds_m
from spec2mpc.sim.scenes import Scene


def _ref_corridor_dist(scene, cells, s_center, half_window_m):
    path = scene.path
    ds = path_ds_m(path)
    w = int(math.ceil(2 * half_window_m / ds)) + 1
    n_open = torch.round(path.length_m / ds).long() + 1  # [Bc]
    n_closed = torch.round(path.length_m / ds).long()
    i0 = torch.floor((s_center - half_window_m) / ds).long()[:, None] + torch.arange(w, device=cells.device)  # [Bc, W]

    def pick(i):
        idx = torch.where(path.closed[:, None], torch.remainder(i, n_closed[:, None]), i.clamp(min=0).minimum(n_open[:, None] - 1))
        return torch.gather(path.xy, 1, idx[..., None].expand(-1, -1, 2))  # [Bc, W, 2]

    a, b = pick(i0), pick(i0 + 1)
    ref = cells.mean(dim=1, keepdim=True)
    c, a, b = cells - ref, a - ref, b - ref
    e = b - a  # [Bc, W, 2]
    e2 = (e * e).sum(-1)  # [Bc, W]
    inv = torch.where(e2 > 1e-12, 1.0 / e2.clamp(min=1e-12), torch.zeros_like(e2))
    ce = torch.bmm(c, e.transpose(1, 2))  # [Bc, N, W]
    ae = (a * e).sum(-1)[:, None]
    ca = torch.bmm(c, a.transpose(1, 2))
    cc = (c * c).sum(-1)[..., None]
    aa = (a * a).sum(-1)[:, None]
    t = ((ce - ae) * inv[:, None]).clamp(0.0, 1.0)
    d2 = cc + aa - 2 * ca - 2 * t * (ce - ae) + t * t * e2[:, None]
    return d2.amin(dim=-1).clamp(min=0.0).sqrt()


def _ref_clearance_grid(scene, origin_xy, t_s, spec, path_window):
    s_center, half = path_window
    b, n, nf = origin_xy.shape[0], spec.n, len(spec.frame_offsets_s)
    peds = [pedestrians_at(scene, t_s + off) for off in spec.frame_offsets_s]  # F x [B, P, 2]
    w = int(math.ceil(2 * half / path_ds_m(scene.path))) + 1
    per_env = n * n * (w + scene.obstacles_r.shape[1] + scene.ped_r.shape[1] + 4) * 4 * 6
    chunk = max(1, int(spec.edt_chunk_bytes // per_env))
    out = torch.empty(b, nf, n, n, dtype=origin_xy.dtype, device=origin_xy.device)
    for lo in range(0, b, chunk):
        sl = slice(lo, min(lo + chunk, b))
        sc = scene.index(sl)
        cells = cell_centers(spec, origin_xy[sl]).reshape(-1, n * n, 2)  # [Bc, N, 2]
        clr = sc.path.half_width_m[:, None] - _ref_corridor_dist(sc, cells, s_center[sl], half)
        clr = torch.minimum(clr, _circles(cells, sc.obstacles_xy, sc.obstacles_r))
        for f in range(nf):
            out[sl, f] = torch.minimum(clr, _circles(cells, peds[f][sl], sc.ped_r)).reshape(-1, n, n)
    return out


def _ref_sq_edt_1d(f, axis_len):
    k = torch.arange(axis_len, device=f.device, dtype=f.dtype)
    sq = (k[:, None] - k[None]) ** 2  # [n(i), n(k)]
    return (sq[None, :, :, None] + f[:, None]).amin(dim=2)


def _ref_signed_distance(occ, res_m, *, chunk_bytes):
    b, nf, n, _ = occ.shape
    flat = occ.reshape(b * nf, n, n)
    inf = float(4 * n * n)
    chunk = max(1, int(chunk_bytes // (n * n * n * 4 * 2)))
    out = torch.empty(b * nf, n, n, dtype=torch.float32, device=occ.device)
    for lo in range(0, b * nf, chunk):
        o = flat[lo : lo + chunk]
        f0 = torch.stack([torch.where(o, 0.0, inf), torch.where(o, inf, 0.0)], dim=1).reshape(-1, n, n)
        g = _ref_sq_edt_1d(f0, n)
        d2 = _ref_sq_edt_1d(g.transpose(1, 2), n).transpose(1, 2)
        d = d2.reshape(-1, 2, n, n).sqrt() * res_m
        out[lo : lo + chunk] = d[:, 0] - d[:, 1]
    sdf = out.reshape(b, nf, n, n)
    any_occ = occ.flatten(2).any(-1)[..., None, None]
    all_occ = occ.flatten(2).all(-1)[..., None, None]
    far = n * res_m
    sdf = torch.where(~any_occ, torch.full_like(sdf, far), sdf)
    return torch.where(all_occ, torch.full_like(sdf, -far), sdf)


def _equivalence_case(cfg, b, seed, res_m, device, frames=None):
    # random lap + goto batch, poses near the path with lateral noise, s_center jittered (incl. past both ends)
    g = torch.Generator().manual_seed(seed)
    sc = Scene.cat([generate_scenes(b - b // 2, cfg, g, "lap"), generate_scenes(b // 2, cfg, g, "goto")])
    spec = GridSpec.from_cfg(OmegaConf.merge(cfg.grid, {"res_m": res_m}))
    if frames is not None:
        spec = replace(spec, frame_offsets_s=frames)
    u = torch.rand(b, generator=g)
    s = u * sc.path.length_m
    idx = torch.round(s / 0.25).long().clamp(max=sc.path.xy.shape[1] - 1)
    origin = sc.path.xy[torch.arange(b), idx] + torch.randn(b, 2, generator=g) * 2.0
    s_c = s + torch.randn(b, generator=g) * 3.0
    s_c[0] = -5.0
    s_c[-1] = sc.path.length_m[-1] + 5.0
    t = torch.rand(b, generator=g) * 30.0
    sc, origin, s_c, t = sc.to(device), origin.to(device), s_c.to(device), t.to(device)
    w = (s_c, spec.size_m * math.sqrt(2))
    new = clearance_grid(sc, origin, t, spec, w)
    ref = _ref_clearance_grid(sc, origin, t, spec, w)
    assert torch.equal(new <= 0, ref <= 0)  # occ, the gate
    assert torch.equal(new, ref)
    assert torch.equal(rasterize(sc, origin, t, spec, w), ref <= 0)
    occ = ref <= 0
    assert torch.equal(signed_distance(occ, spec.res_m, chunk_bytes=spec.edt_chunk_bytes),
                       _ref_signed_distance(occ, spec.res_m, chunk_bytes=spec.edt_chunk_bytes))


@pytest.mark.parametrize("b,seed", [(2, 0), (6, 1)])
def test_equivalence_to_reference(cfg, b, seed):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _equivalence_case(cfg, b, seed, 0.2, dev)


def test_edt_equivalence_random_occupancy():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(11)
    for p in (0.0, 0.01, 0.2, 0.9, 1.0):
        occ = (torch.rand(3, 4, 40, 40, generator=g) < p).to(dev)
        assert torch.equal(signed_distance(occ, 0.2, chunk_bytes=2.0e9), _ref_signed_distance(occ, 0.2, chunk_bytes=2.0e9))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU variant")
@pytest.mark.parametrize("b,seed,res_m,frames", [(64, 2, 0.2, None), (256, 3, 0.2, None), (128, 4, 0.2, (0.5, 1.0, 1.5)),
                                                 (48, 5, 0.1, None), (200, 6, 0.2, None)])
def test_equivalence_to_reference_gpu(cfg, b, seed, res_m, frames):
    _equivalence_case(cfg, b, seed, res_m, torch.device("cuda"), frames)
