import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.robots.kart import sample_karts
from spec2mpc.sim.storage import read_shard, write_shard
from spec2mpc.tokenize.robot_tokens import (D_ROBOT_TOK, F_MAX, N_ROBOT_TOK, ROBOT_TOKEN_TYPES, robot_columns,
                                            robot_tokens, robot_tokens_from_columns)
from test_storage import B, shard_inputs  # noqa: F401


@pytest.fixture(scope="module")
def tok_cfg():
    return OmegaConf.load("configs/robot_tokens/kart.yaml")


def test_tokens_match_written_shard(tmp_path, shard_inputs, tok_cfg):  # noqa: F811
    robot, scene, theta, cond, traj, metrics, phi, tok, ids = shard_inputs
    write_shard(tmp_path, "s", robot, scene, theta, cond, traj, metrics, phi, tok, store_hz=10.0, meta={}, **ids)
    table, _ = read_shard(tmp_path / "s")
    a = robot_tokens(robot, tok_cfg)
    cols = {k: torch.tensor(table.column(k).to_numpy()) for k in table.column_names
            if k.startswith(("kart_", "perc_", "rt_"))}
    assert set(robot_columns(robot)) == set(cols)
    b = robot_tokens_from_columns(cols, tok_cfg)
    assert torch.allclose(a, b, atol=1e-6) and a.shape == (B, N_ROBOT_TOK, D_ROBOT_TOK)


def test_in_range_values_within_unit_box(kart_cfg, runtime, gen, tok_cfg):
    t = robot_tokens(sample_karts(256, kart_cfg, runtime, gen), tok_cfg)
    vals, mask = t[..., N_ROBOT_TOK:N_ROBOT_TOK + F_MAX], t[..., N_ROBOT_TOK + F_MAX:]
    assert (vals[mask.bool()].abs() <= 1.0 + 1e-5).all()


def test_type_one_hot_and_mask(kart_cfg, runtime, gen, tok_cfg):
    t = robot_tokens(sample_karts(4, kart_cfg, runtime, gen), tok_cfg)
    assert D_ROBOT_TOK == 24 and ROBOT_TOKEN_TYPES[0] == "chassis"
    for i, name in enumerate(ROBOT_TOKEN_TYPES):
        assert torch.equal(t[:, i, :N_ROBOT_TOK], torch.eye(N_ROBOT_TOK)[i].expand(4, -1))
        assert (t[:, i, N_ROBOT_TOK + F_MAX:].sum(-1) == len(tok_cfg.tokens[name])).all()
        assert (t[:, i, N_ROBOT_TOK:N_ROBOT_TOK + F_MAX][t[:, i, N_ROBOT_TOK + F_MAX:] == 0] == 0).all()


def test_missing_column_named(kart_cfg, runtime, gen, tok_cfg):
    cols = robot_columns(sample_karts(2, kart_cfg, runtime, gen))
    del cols["kart_mass_kg"]
    with pytest.raises(KeyError, match="kart_mass_kg"):
        robot_tokens_from_columns(cols, tok_cfg)
