import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.features.metrics import METRIC_NAMES
from spec2mpc.features.phi import PHI_NAMES
from spec2mpc.mpc.theta import THETA_NAMES, sobol_theta
from spec2mpc.robots.kart import RuntimeConfig, sample_karts
from spec2mpc.sim.conditions import sample_conditions
from spec2mpc.sim.rollout import SolverStats
from spec2mpc.sim.scenes import Scene, generate_scenes
from spec2mpc.sim.storage import cond_from_table, metrics_from_table, read_shard, write_shard
from spec2mpc.sim.trajectory import Trajectory

B, T, XI = 32, 41, 16  # 41 steps at 50 Hz -> 9 stored steps at 10 Hz


@pytest.fixture(scope="module")
def shard_inputs():
    g = torch.Generator().manual_seed(0)
    sim_cfg = OmegaConf.load("configs/sim/default.yaml")
    mpc_cfg = OmegaConf.load("configs/mpc/kart.yaml")
    kart_cfg = OmegaConf.load("configs/robots/kart.yaml")
    rt = RuntimeConfig(control_rate_hz=50.0, solver_iters=10, perception_rate_hz=10.0)
    rep = torch.arange(B // XI).repeat_interleave(XI)  # [B] block index per episode
    robot = sample_karts(B // XI, kart_cfg, rt, g).index(rep)
    scene = Scene.cat([generate_scenes(1, sim_cfg, g, "lap"), generate_scenes(1, sim_cfg, g, "goto")]).index(rep)
    theta = sobol_theta(B // XI, mpc_cfg, 0).index(rep)
    cond = sample_conditions(B, sim_cfg, g)
    traj = Trajectory(
        t_s=torch.arange(T, dtype=torch.float32)[None].expand(B, T) * 0.02,
        x=torch.randn(B, T, 8, generator=g), u=torch.randn(B, T, 2, generator=g),
        s=torch.randn(B, T, generator=g), a_long=torch.randn(B, T, generator=g), a_lat=torch.randn(B, T, generator=g),
        clearance_true=torch.randn(B, T, generator=g), solver_ok=torch.rand(B, T, generator=g) > 0.1,
        valid=torch.arange(T)[None] < torch.randint(1, T + 1, (B, 1), generator=g),
        p_drive_W=torch.rand(B, T, generator=g), done_reason=torch.zeros(B, dtype=torch.long),
        completion_time_s=torch.full((B,), float("nan")), laps_done=torch.zeros(B, dtype=torch.long), dt_s=0.02,
    )
    metrics = {k: torch.randn(B, generator=g) for k in METRIC_NAMES}
    phi = torch.randn(B, len(PHI_NAMES), generator=g)
    tok = torch.randn(B, 64, 6, generator=g)
    cost = torch.rand(B, generator=g)
    cost[0] = float("nan")  # no ok solve in that episode
    stats = SolverStats(cost_mean=cost, grad_norm_mean=torch.rand(B, generator=g))
    ids = dict(episode_id=torch.arange(100, 100 + B), block_id=rep + 7, solver_stats=stats)
    return robot, scene, theta, cond, traj, metrics, phi, tok, ids


def test_roundtrip(tmp_path, shard_inputs):
    robot, scene, theta, cond, traj, metrics, phi, tok, ids = shard_inputs
    write_shard(tmp_path, "shard_000000", robot, scene, theta, cond, traj, metrics, phi, tok,
                store_hz=10.0, meta={"seed": 3, "device": "cpu"}, **ids)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shard_000000"]  # no temp dir left behind
    table, z = read_shard(tmp_path / "shard_000000")
    assert table.num_rows == B
    col = lambda n: table.column(n).to_numpy(zero_copy_only=False)
    assert np.array_equal(col("episode_id"), ids["episode_id"].numpy())
    assert np.array_equal(col("block_id"), ids["block_id"].numpy())
    assert list(col("robot_id")) == robot.robot_id
    assert np.array_equal(col("kart_mass_kg"), robot.params.mass_kg.numpy())
    assert np.array_equal(col("perc_has_prediction"), robot.perception.has_prediction.numpy())
    assert (col("rt_control_rate_hz") == 50.0).all() and (col("rt_solver_iters") == 10).all()
    assert np.array_equal(col("scene_seed"), scene.seed.numpy())
    assert list(col("task")) == ["lap"] * XI + ["goto"] * XI
    for i, n in enumerate(THETA_NAMES):
        assert np.array_equal(col(n), theta.values[:, i].numpy())
    for i, n in enumerate(PHI_NAMES):
        assert np.array_equal(col(f"phi_{n}"), phi[:, i].numpy())
    for n in METRIC_NAMES:
        assert np.array_equal(col(n), metrics[n].numpy())
    assert np.array_equal(col("xi_seed"), cond.seed.numpy())
    assert np.array_equal(col("xi_init_speed_mps"), cond.init_offset[:, 2].numpy())
    assert np.array_equal(col("solver_fail_count"), ((~traj.solver_ok) & traj.valid).sum(1).numpy())
    assert np.array_equal(col("mean_cost"), ids["solver_stats"].cost_mean.numpy(), equal_nan=True)
    assert np.array_equal(col("mean_grad_norm"), ids["solver_stats"].grad_norm_mean.numpy())
    # helpers used by the scoring pass
    c = cond_from_table(table)
    assert torch.equal(c["friction_scale"], cond.friction_scale) and torch.equal(c["latency_s"], cond.latency_s)
    m = metrics_from_table(table)
    assert list(m) == list(METRIC_NAMES) and torch.equal(m["energy_Wh"], metrics["energy_Wh"])
    # trajectories at store_hz (every 5th control step)
    assert np.array_equal(z["x"][:], traj.x[:, ::5].numpy())
    assert np.array_equal(z["u"][:], traj.u[:, ::5].numpy())
    assert np.array_equal(z["valid"][:], traj.valid[:, ::5].numpy())
    assert np.array_equal(z["path_tokens"][:], tok.numpy())
    assert z.attrs["dt_s"] == pytest.approx(0.1)
    meta = json.loads((tmp_path / "shard_000000" / "meta.json").read_text())
    assert meta["seed"] == 3 and meta["device"] == "cpu" and meta["n_episodes"] == B
    assert meta["torch_version"] == torch.__version__ and "git_hash" in meta


def test_existing_shard_is_not_overwritten(tmp_path, shard_inputs):
    robot, scene, theta, cond, traj, metrics, phi, tok, ids = shard_inputs
    args = (tmp_path, "s", robot, scene, theta, cond, traj, metrics, phi, tok)
    write_shard(*args, store_hz=10.0, meta={}, **ids)
    with pytest.raises(FileExistsError):
        write_shard(*args, store_hz=10.0, meta={}, **ids)


def test_stale_temp_dir_is_replaced(tmp_path, shard_inputs):
    # a killed job leaves a partial temp dir and no final shard
    robot, scene, theta, cond, traj, metrics, phi, tok, ids = shard_inputs
    stale = tmp_path / ".tmp_s"
    stale.mkdir()
    (stale / "episodes.parquet").write_text("partial")
    write_shard(tmp_path, "s", robot, scene, theta, cond, traj, metrics, phi, tok, store_hz=10.0, meta={}, **ids)
    assert not stale.exists()
    assert read_shard(tmp_path / "s")[0].num_rows == B
