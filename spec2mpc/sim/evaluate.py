from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.features.metrics import episode_metrics
from spec2mpc.features.phi import phi
from spec2mpc.mpc.theta import ThetaBatch
from spec2mpc.robots.kart import Robot
from spec2mpc.sim.conditions import Conditions
from spec2mpc.sim.rollout import Controller, SolverStats, episode_timeout_s, run_episodes_with_stats
from spec2mpc.sim.scenes import Scene
from spec2mpc.sim.trajectory import Trajectory


@dataclass
class RolloutOut:
    traj: Trajectory  # [B, T, ...] on the rollout device
    stats: SolverStats  # [B] on the rollout device
    metrics: Dict[str, Tensor]  # {name: [B]} on CPU
    phi: Tensor  # [B, D_PHI] on CPU


def rollout_metrics(robot: Robot, scene: Scene, theta: ThetaBatch, cond: Conditions, cfg: DictConfig,
                    gen: torch.Generator, device: torch.device,
                    make_controller: Optional[Callable[[Robot, Scene, Conditions], Controller]] = None) -> RolloutOut:
    # robot, scene, theta, cond [B]; cfg: root config with sim, mpc, robots -> closed-loop rollout, metrics, phi;
    # make_controller(robot, scene, cond) on the device builds a controller that replaces the MPC
    r, sc, c = robot.to(device), scene.to(device), cond.to(device)
    traj, stats = run_episodes_with_stats(r, sc, theta.to(device), c, cfg.sim, cfg.mpc, gen,
                                          tire_model=cfg.robots.tire_model,
                                          controller=None if make_controller is None else make_controller(r, sc, c))
    metrics = episode_metrics(traj, r.params, sc, float(cfg.mpc.floors.m_floor_m))
    ph = phi(metrics, r.params, sc, episode_timeout_s(sc, r.params, cfg.sim))
    return RolloutOut(traj=traj, stats=stats, metrics={k: v.cpu() for k, v in metrics.items()}, phi=ph.cpu())
