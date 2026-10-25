import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.sim.paths import RefPath, path_lookup, project_global, project_to_path
from spec2mpc.sim.scenes import Scene, generate_scenes, pedestrians_at, true_clearance

B = 16


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load("configs/sim/default.yaml")


@pytest.fixture(scope="module")
def scenes_lap(cfg):
    return generate_scenes(B, cfg, torch.Generator().manual_seed(1), "lap")


@pytest.fixture(scope="module")
def scenes_goto(cfg):
    return generate_scenes(B, cfg, torch.Generator().manual_seed(2), "goto")


def n_valid(p: RefPath, ds_m: float) -> torch.Tensor:
    n = torch.round(p.length_m / ds_m).long()
    return torch.where(p.closed, n, n + 1)  # [B]


def last_valid_point(p: RefPath, ds_m: float) -> torch.Tensor:
    idx = n_valid(p, ds_m) - 1
    return p.xy[torch.arange(p.batch_size), idx]  # [B, 2]


def test_lap_path_closed_and_uniform(scenes_lap, cfg):
    p = scenes_lap.path
    gap = (p.xy[:, 0] - last_valid_point(p, cfg.ds_m)).norm(dim=-1)
    assert (gap < 2 * cfg.ds_m).all()
    assert ((p.length_m >= 80) & (p.length_m <= 250)).all()
    assert p.closed.all()
    for b in range(B):
        n = int(n_valid(p, cfg.ds_m)[b])
        seg = (p.xy[b, 1:n] - p.xy[b, : n - 1]).norm(dim=-1)
        assert torch.allclose(p.s[b, 1:n] - p.s[b, : n - 1], torch.full_like(seg, cfg.ds_m), atol=1e-4)
        assert seg.max() <= cfg.ds_m + 1e-4 and seg.min() > 0.9 * cfg.ds_m  # chord ~ arc at gentle curvature
        assert (p.xy[b, n:] == p.xy[b, n - 1]).all()  # padding repeats last point


def test_lap_geometry_limits(scenes_lap, cfg):
    p = scenes_lap.path
    assert p.curvature.abs().max() <= 1.0 / cfg.scene.min_turn_radius_floor_m + 1e-6
    assert ((p.half_width_m >= 2.5) & (p.half_width_m <= 5.0)).all()
    assert (scenes_lap.task == 0).all() and (scenes_lap.n_laps == cfg.scene.n_laps).all()


def test_lap_no_self_overlap(scenes_lap, cfg):
    p = scenes_lap.path
    for b in range(B):
        n = int(n_valid(p, cfg.ds_m)[b])
        xy, s, hw, L = p.xy[b, :n], p.s[b, :n], float(p.half_width_m[b]), float(p.length_m[b])
        ds = (s[:, None] - s[None]).abs()
        ds = torch.minimum(ds, L - ds)
        d = torch.cdist(xy, xy)
        assert not ((ds > 4 * hw) & (d < 2 * hw + 1.0)).any()


def test_lookup_wraps_on_closed(scenes_lap):
    p = scenes_lap.path
    a, _, _ = path_lookup(p, torch.zeros(B, 1))
    b, _, _ = path_lookup(p, p.length_m[:, None])
    assert torch.allclose(a, b, atol=1e-3)
    c, _, _ = path_lookup(p, p.length_m[:, None] * 2.0 + 3.0)
    d, _, _ = path_lookup(p, torch.full((B, 1), 3.0))
    assert torch.allclose(c, d, atol=1e-3)


def test_lookup_hits_nodes_and_heading_matches_tangent(scenes_lap, cfg):
    p = scenes_lap.path
    s = torch.arange(1, 20).float()[None].repeat(B, 1) * cfg.ds_m
    xy, h, k = path_lookup(p, s)
    assert torch.allclose(xy, p.xy[:, 1:20], atol=1e-5)
    tang = p.xy[:, 2:21] - p.xy[:, :19]
    assert torch.cos(torch.atan2(tang[..., 1], tang[..., 0]) - h).min() > 0.99


def test_lookup_open_clamps(scenes_goto):
    p = scenes_goto.path
    a, _, _ = path_lookup(p, torch.full((B, 1), -5.0))
    b, _, _ = path_lookup(p, torch.zeros(B, 1))
    c, _, _ = path_lookup(p, p.length_m[:, None] + 10.0)
    d, _, _ = path_lookup(p, p.length_m[:, None])
    assert torch.allclose(a, b) and torch.allclose(c, d)
    assert not p.closed.any()


def test_lookup_differentiable_in_s(scenes_lap):
    p = scenes_lap.path
    s = (torch.rand(B, 3, generator=torch.Generator().manual_seed(5)) * p.length_m[:, None]).requires_grad_()
    xy, h, _ = path_lookup(p, s)
    xy.sum().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    # d xy / d s is the unit tangent: |grad of x| and |grad of y| terms
    s2 = s.detach().clone().requires_grad_()
    xy2, h2, _ = path_lookup(p, s2)
    gx, = torch.autograd.grad(xy2[..., 0].sum(), s2)
    assert torch.allclose(gx, torch.cos(h2), atol=0.1)  # chord slope vs interpolated heading: ~kappa*ds/2


def test_project_recovers_s(scenes_lap, cfg):
    s = torch.rand(B, generator=torch.Generator().manual_seed(3)) * scenes_lap.path.length_m
    xy, _, _ = path_lookup(scenes_lap.path, s[:, None])
    got = project_to_path(scenes_lap.path, xy[:, 0], s + 1.0, 5.0)
    err = (got - s).abs()
    err = torch.minimum(err, scenes_lap.path.length_m - err)
    assert (err < cfg.ds_m).all()


def test_project_open_path(scenes_goto, cfg):
    p = scenes_goto.path
    s = torch.rand(B, generator=torch.Generator().manual_seed(4)) * p.length_m
    xy, _, _ = path_lookup(p, s[:, None])
    got = project_to_path(p, xy[:, 0], s - 1.0, 5.0)
    assert ((got - s).abs() < cfg.ds_m).all()
    assert ((got >= 0) & (got <= p.length_m)).all()


def test_clearance_sign(scenes_goto):
    on_center, _, _ = path_lookup(scenes_goto.path, torch.zeros(B, 1))
    assert (true_clearance(scenes_goto, on_center, 0.0) > 0).all()  # start is clear by construction


def test_clearance_analytic(scenes_goto, scenes_lap):
    p = scenes_lap.path
    c, h, _ = path_lookup(p, torch.full((B, 1), 10.0))
    normal = torch.stack([-torch.sin(h), torch.cos(h)], dim=-1)
    for off in (0.0, 1.0, -1.5):
        clr = true_clearance(scenes_lap, c + off * normal, 0.0)
        assert torch.allclose(clr, scenes_lap.path.half_width_m[:, None] - abs(off), atol=0.05)
    # lap scenes have no obstacles: far outside is negative
    assert (true_clearance(scenes_lap, c + 100.0 * normal, 0.0) < 0).all()
    # circle: place a point at 0.5 m outside a real obstacle
    sc = scenes_goto
    has = (sc.obstacles_r > 0).any(dim=1).nonzero()
    assert len(has) > 0
    b = int(has[0])
    o = int((sc.obstacles_r[b] > 0).nonzero()[0])
    q = sc.obstacles_xy[b, o] + torch.tensor([sc.obstacles_r[b, o] + 0.5, 0.0])
    clr = true_clearance(sc.index(torch.tensor([b])), q[None, None], 0.0)
    assert clr.item() <= 0.5 + 1e-4


def test_pedestrians_stay_in_corridor(scenes_goto, cfg):
    sc = scenes_goto
    assert (sc.ped_r > 0).any()
    real = sc.ped_r > 0  # [B, P]
    hw = sc.path.half_width_m[:, None]
    for t in torch.linspace(0, 60, 31):
        xy = pedestrians_at(sc, float(t))  # [B, P, 2]
        assert xy.shape == sc.ped_xy0.shape
        B_, P_ = real.shape
        flat = xy.reshape(B_, P_, 2)
        s_guess = torch.zeros(B_ * P_)
        rep = sc.path.index(torch.arange(B_).repeat_interleave(P_))
        s = project_to_path(rep, flat.reshape(-1, 2), s_guess, sc.path.length_m.max().item())
        p_xy, _, _ = path_lookup(rep, s[:, None])
        off = (flat.reshape(-1, 2) - p_xy[:, 0]).norm(dim=-1).reshape(B_, P_)
        assert (off[real] <= (hw - sc.ped_r + 0.05).expand_as(off)[real]).all()


def test_pedestrians_deterministic_and_tensor_time(scenes_goto):
    a = pedestrians_at(scenes_goto, 7.3)
    b = pedestrians_at(scenes_goto, torch.full((B,), 7.3))
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(pedestrians_at(scenes_goto, 7.3), a)


def test_goto_contents(scenes_goto, cfg):
    sc = scenes_goto
    assert (sc.task == 1).all()
    assert ((sc.path.length_m >= 30) & (sc.path.length_m <= 120)).all()
    assert ((sc.path.half_width_m >= 3) & (sc.path.half_width_m <= 8)).all()
    start = sc.path.xy[:, 0]
    for xy, r in ((sc.obstacles_xy, sc.obstacles_r), (sc.ped_xy0, sc.ped_r)):
        real = r > 0
        assert (real.sum(1) <= 6).all()
        assert ((xy - start[:, None]).norm(dim=-1)[real] > 5.0).all()
    r = sc.obstacles_r[sc.obstacles_r > 0]
    assert ((r >= 0.3) & (r <= 1.0)).all()
    sp = sc.ped_vel.norm(dim=-1)[sc.ped_r > 0]
    assert ((sp >= 0.5 - 1e-5) & (sp <= 1.5 + 1e-5)).all()
    assert (sc.goal_tol_m > 0).all()


def test_static_obstacles_off_path_and_in_corridor(cfg):
    # the planner routes around static obstacles: each clears the centerline by path_clear_m, and stays in the corridor
    sc = generate_scenes(64, cfg, torch.Generator().manual_seed(5), "goto")
    real = sc.obstacles_r > 0
    assert real.sum() > 32
    p, _, _ = path_lookup(sc.path, project_global(sc.path, sc.obstacles_xy))
    d = (sc.obstacles_xy - p).norm(dim=-1)  # [B, O] distance to the centerline
    clear = float(cfg.scene.goto.path_clear_m)
    tol = float(cfg.ds_m)  # the placement test uses the path's sample points
    assert (d - sc.obstacles_r >= clear - tol)[real].all()
    assert (d + sc.obstacles_r <= sc.path.half_width_m[:, None] + 1e-3)[real].all()


def test_generation_deterministic(cfg):
    a = generate_scenes(4, cfg, torch.Generator().manual_seed(9), "lap")
    b = generate_scenes(4, cfg, torch.Generator().manual_seed(9), "lap")
    c = generate_scenes(4, cfg, torch.Generator().manual_seed(10), "lap")
    assert torch.equal(a.path.xy, b.path.xy) and torch.equal(a.seed, b.seed)
    assert not torch.equal(a.seed, c.seed)


def test_scene_index_cat_to(scenes_lap, scenes_goto):
    both = Scene.cat([scenes_lap, scenes_goto])
    assert both.batch_size == 2 * B
    assert both.path.xy.shape[0] == 2 * B and both.path.xy.shape[1] >= scenes_lap.path.xy.shape[1]
    sub = both.index(torch.tensor([0, B]))
    assert sub.task.tolist() == [0, 1]
    a, _, _ = path_lookup(sub.path, torch.tensor([[1.0], [1.0]]))
    b0, _, _ = path_lookup(scenes_lap.path.index(slice(0, 1)), torch.tensor([[1.0]]))
    assert torch.allclose(a[0], b0[0], atol=1e-5)
    assert sub.to("cpu").batch_size == 2


def test_lap_seam_is_smooth(scenes_lap, cfg):
    p = scenes_lap.path
    last = n_valid(p, cfg.ds_m) - 1
    ar = torch.arange(B)
    dh = torch.atan2(torch.sin(p.heading[:, 0] - p.heading[ar, last]), torch.cos(p.heading[:, 0] - p.heading[ar, last]))
    assert (dh.abs() < 0.15).all()
    assert ((p.curvature[:, 0] - p.curvature[ar, last]).abs() < 0.05).all()


def test_pedestrians_in_corridor_curvy_paths(cfg):
    sc = generate_scenes(64, cfg, torch.Generator().manual_seed(21), "goto")
    sc = sc.index((sc.path.curvature.abs().amax(1) > 0.08).nonzero()[:, 0])
    assert sc.batch_size > 0 and (sc.ped_r > 0).any()
    real = sc.ped_r > 0
    hw = sc.path.half_width_m[:, None]
    for t in torch.linspace(0, 60, 31):
        xy = pedestrians_at(sc, float(t))
        s = project_global(sc.path, xy)
        p, _, _ = path_lookup(sc.path, s)
        off = (xy - p).norm(dim=-1)
        assert (off <= hw - sc.ped_r + 0.05)[real].all()
