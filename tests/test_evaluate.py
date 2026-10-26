import torch
from omegaconf import OmegaConf

from spec2mpc.features.metrics import METRIC_NAMES, episode_metrics
from spec2mpc.features.phi import PHI_NAMES, phi
from spec2mpc.mpc.theta import sobol_theta
from spec2mpc.robots.kart import RuntimeConfig
from spec2mpc.sim.conditions import sample_conditions
from spec2mpc.sim.evaluate import rollout_metrics
from spec2mpc.sim.problems import sample_problems
from spec2mpc.sim.rollout import episode_timeout_s, run_episodes_with_stats

B = 4


def tiny_cfg():
    # 0.3 s episodes at 20 Hz, 2 solver iterations, 0.5 m cells (tests/test_gen_smoke.py TINY)
    cfg = OmegaConf.load("configs/gen.yaml")
    for k, name in (("sim", "default"), ("mpc", "kart"), ("robots", "kart"), ("tokenize", "path")):
        cfg[k] = OmegaConf.load(f"configs/{k}/{name}.yaml")
    cfg.sim.episode.timeout_cap_lap_s = 0.3
    cfg.sim.episode.timeout_cap_goto_s = 0.3
    cfg.sim.grid.res_m = 0.5
    return cfg


def test_rollout_metrics_matches_gen_pipeline():  # tiny episode, CPU
    cfg = tiny_cfg()
    rt = RuntimeConfig(control_rate_hz=20.0, solver_iters=2, perception_rate_hz=10.0)
    pr = sample_problems({"lap": 2, "goto": 2}, cfg, 0, rt, "e")
    robot, scene = pr.robot, pr.scene
    theta = sobol_theta(B, cfg.mpc, 0)
    cond = sample_conditions(B, cfg.sim, torch.Generator().manual_seed(1))
    out = rollout_metrics(robot, scene, theta, cond, cfg, torch.Generator().manual_seed(0), torch.device("cpu"))
    assert out.phi.shape == (B, len(PHI_NAMES)) and set(out.metrics) == set(METRIC_NAMES)
    assert out.phi.device.type == "cpu" and all(v.device.type == "cpu" for v in out.metrics.values())
    # same as the inline gen_rollouts pipeline it replaced
    traj, stats = run_episodes_with_stats(robot, scene, theta, cond, cfg.sim, cfg.mpc, torch.Generator().manual_seed(0),
                                          tire_model=cfg.robots.tire_model)
    m = episode_metrics(traj, robot.params, scene, float(cfg.mpc.floors.m_floor_m))
    ph = phi(m, robot.params, scene, episode_timeout_s(scene, robot.params, cfg.sim))
    assert out.phi.numpy().tobytes() == ph.numpy().tobytes()
    assert torch.equal(out.traj.x, traj.x) and out.stats.cost_mean.numpy().tobytes() == stats.cost_mean.numpy().tobytes()
