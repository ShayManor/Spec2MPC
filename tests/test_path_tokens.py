import math

import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.robots.kart import sample_karts
from spec2mpc.sim.paths import RefPath
from spec2mpc.sim.scenes import Scene, generate_scenes
from spec2mpc.tokenize.path_tokens import PATH_TOKEN_FIELDS, path_tokens

DS_M = 0.25
N_TOKENS = 64
F = {n: i for i, n in enumerate(PATH_TOKEN_FIELDS)}


@pytest.fixture(scope="module")
def tok_cfg():
    return OmegaConf.load("configs/tokenize/path.yaml")


@pytest.fixture
def karts(kart_cfg, gen, runtime):
    return sample_karts(2, kart_cfg, runtime, gen).params


def make_scene(path: RefPath, obstacles=None) -> Scene:
    b = path.batch_size
    o = torch.zeros(b, 1, 2) if obstacles is None else obstacles[0]
    r = torch.zeros(b, 1) if obstacles is None else obstacles[1]
    return Scene(
        path=path, obstacles_xy=o, obstacles_r=r, ped_xy0=torch.zeros(b, 1, 2), ped_vel=torch.zeros(b, 1, 2),
        ped_r=torch.zeros(b, 1), task=torch.ones(b, dtype=torch.long), n_laps=torch.ones(b, dtype=torch.long),
        goal_tol_m=torch.zeros(b), seed=torch.zeros(b, dtype=torch.int64),
    )


def straight(b: int, half_width_m: float, length_m: float = 40.0) -> RefPath:
    m = int(round(length_m / DS_M)) + 1
    s = (torch.arange(m) * DS_M).expand(b, m).contiguous()
    xy = torch.stack([s, torch.zeros_like(s)], dim=-1)
    return RefPath(
        xy=xy, s=s, heading=torch.zeros(b, m), curvature=torch.zeros(b, m),
        length_m=torch.full((b,), length_m), closed=torch.zeros(b, dtype=torch.bool),
        half_width_m=torch.full((b,), half_width_m),
    )


def circle(b: int, length_m: float = 125.0) -> RefPath:
    n = int(round(length_m / DS_M))
    radius = length_m / (2 * math.pi)
    s = (torch.arange(n + 1) * DS_M).expand(b, n + 1).contiguous()
    a = s / radius
    xy = torch.stack([radius * torch.cos(a), radius * torch.sin(a)], dim=-1)
    return RefPath(
        xy=xy, s=s, heading=a + math.pi / 2, curvature=torch.full((b, n + 1), 1.0 / radius),
        length_m=torch.full((b,), length_m), closed=torch.ones(b, dtype=torch.bool),
        half_width_m=torch.full((b,), 3.0),
    )


def test_straight_corridor(karts, tok_cfg):
    p = straight(2, 3.0)
    out = path_tokens(p, make_scene(p), karts, tok_cfg)
    assert out.shape == (2, N_TOKENS, len(PATH_TOKEN_FIELDS))
    assert out[..., F["curvature_x_Rmin"]].abs().max() < 1e-5
    expect = (3.0 / karts.wheelbase_m)[:, None].expand(-1, N_TOKENS)
    assert torch.allclose(out[..., F["clear_left_over_L"]], expect, atol=0.05)
    assert torch.allclose(out[..., F["clear_right_over_L"]], expect, atol=0.05)
    assert (out[..., F["closed"]] == 0).all()


def test_circle_curvature(karts, tok_cfg):
    p = circle(2)
    out = path_tokens(p, make_scene(p), karts, tok_cfg)
    radius = 125.0 / (2 * math.pi)
    expect = karts.min_turn_radius_m / radius
    assert torch.allclose(out[..., F["curvature_x_Rmin"]], expect[:, None].expand(-1, N_TOKENS), rtol=1e-3)
    assert (out[..., F["closed"]] == 1).all() and (out[..., F["is_goal"]] == 0).all()
    assert torch.allclose(out[..., F["seg_len_over_L"]], (125.0 / N_TOKENS / karts.wheelbase_m)[:, None].expand(-1, N_TOKENS), rtol=1e-4)


def test_goal_flag_only_on_last_token_of_open_path(karts, tok_cfg):
    scenes = generate_scenes(2, OmegaConf.load("configs/sim/default.yaml"), torch.Generator().manual_seed(2), "goto")
    out = path_tokens(scenes.path, scenes, karts, tok_cfg)
    assert (out[:, -1, F["is_goal"]] == 1).all() and (out[:, :-1, F["is_goal"]] == 0).all()
    assert (out[..., F["closed"]] == 0).all()


def test_obstacle_limits_clearance_and_cap(karts, tok_cfg):
    p = straight(1, 30.0)  # wide corridor: the wall is beyond the cap
    karts1 = karts.index(slice(0, 1))
    obs = (torch.tensor([[[20.0, 4.0]]]), torch.tensor([[1.0]]))  # left of the centerline at x = 20
    out = path_tokens(p, make_scene(p, obs), karts1, tok_cfg)
    L = float(karts1.wheelbase_m)
    x = torch.linspace(0, 40.0, N_TOKENS)
    i = int((x - 20.0).abs().argmin())
    assert abs(float(out[0, i, F["clear_left_over_L"]]) * L - 3.0) < 0.2
    assert abs(float(out[0, 0, F["clear_left_over_L"]]) * L - tok_cfg.clear_cap_m) < 1e-4
    assert abs(float(out[0, i, F["clear_right_over_L"]]) * L - tok_cfg.clear_cap_m) < 1e-4


def test_circle_corridor_wall_stop(karts, tok_cfg):
    p = circle(2)
    out = path_tokens(p, make_scene(p), karts, tok_cfg)
    total = out[..., F["clear_left_over_L"]] + out[..., F["clear_right_over_L"]]
    expect = (2 * 3.0 / karts.wheelbase_m)[:, None].expand(-1, N_TOKENS)
    assert torch.allclose(total, expect, atol=0.05)


def test_modified_cfg_changes_output(karts, tok_cfg):
    p = straight(2, 30.0)
    cfg = OmegaConf.create({**OmegaConf.to_container(tok_cfg), "clear_cap_m": 2.0, "n_path_tokens": 16})
    out = path_tokens(p, make_scene(p), karts, cfg)
    assert out.shape == (2, 16, len(PATH_TOKEN_FIELDS))
    assert torch.allclose(out[..., F["clear_left_over_L"]], (2.0 / karts.wheelbase_m)[:, None].expand(-1, 16), atol=1e-4)


def test_too_few_tokens_raises(karts, tok_cfg):
    p = straight(2, 3.0)
    cfg = OmegaConf.create({**OmegaConf.to_container(tok_cfg), "n_path_tokens": 1})
    with pytest.raises(ValueError):
        path_tokens(p, make_scene(p), karts, cfg)
