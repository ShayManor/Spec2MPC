"""Closed-loop throughput of one batch: steps/s, episodes/hour, and the perception/solver/plant time split.

Usage: uv run python scripts/bench_mpcc.py bench.B=4096 bench.steps=100 device=cuda mpc.solver.compile=true
Uses the same configs as gen_rollouts.py (configs/gen.yaml); the runtime config comes from cfg.bench.
"""
import math
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Callable, Dict, Iterator

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import spec2mpc.sim.rollout as rollout
from spec2mpc.mpc.mpcc import MPCC
from spec2mpc.robots.kart import RuntimeConfig
from spec2mpc.sim.rollout import episode_timeout_s, run_episodes

from gen_rollouts import sample_batch


@contextmanager
def timed(device: torch.device, acc: Dict[str, float], counts: Dict[str, int]) -> Iterator[None]:
    # wraps perceive, MPCC.solve and kart_step as called by run_episodes; syncs CUDA around each call
    def wrap(fn: Callable, key: str) -> Callable:
        def inner(*a, **k):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = fn(*a, **k)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            acc[key] += time.perf_counter() - t0
            counts[key] += 1
            return out
        return inner

    orig = (rollout.perceive, rollout.kart_step, MPCC.solve)
    rollout.perceive = wrap(orig[0], "perception_edt")
    rollout.kart_step = wrap(orig[1], "plant")
    MPCC.solve = wrap(orig[2], "solver")
    try:
        yield
    finally:
        rollout.perceive, rollout.kart_step, MPCC.solve = orig


def fixed_steps_cfg(sim_cfg: DictConfig, steps: int, control_rate_hz: float) -> DictConfig:
    # every episode times out after exactly `steps` control steps
    c = OmegaConf.create(OmegaConf.to_container(sim_cfg, resolve=True))
    t = (steps - 1) / control_rate_hz
    c.episode.timeout_cap_lap_s = t
    c.episode.timeout_cap_goto_s = t
    c.episode.timeout_v_frac = 1e-9  # uncapped timeout huge, so the cap binds
    return c


def run(cfg: DictConfig, steps: int, device: torch.device) -> Dict[str, float]:
    bc = cfg.bench
    rt = RuntimeConfig(control_rate_hz=float(bc.control_rate_hz), solver_iters=int(bc.solver_iters),
                       perception_rate_hz=float(bc.perception_rate_hz))
    robot, scene, theta, cond, _, _, gen = sample_batch(cfg, 0, int(bc.B))
    robot = replace(robot, runtime=rt).to(device)
    scene, theta, cond = scene.to(device), theta.to(device), cond.to(device)
    acc = {"perception_edt": 0.0, "solver": 0.0, "plant": 0.0}
    counts = {k: 0 for k in acc}
    sim = fixed_steps_cfg(cfg.sim, steps, rt.control_rate_hz)
    with timed(device, acc, counts):
        t0 = time.perf_counter()
        run_episodes(robot, scene, theta, cond, sim, cfg.mpc, gen, tire_model=cfg.robots.tire_model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total = time.perf_counter() - t0
    timeout = episode_timeout_s(scene, robot.params, cfg.sim)  # [B] at the configured episode length
    ep_steps = int(math.ceil(float(timeout.max()) * rt.control_rate_hz)) + 1
    return dict(total=total, steps=counts["solver"], ep_steps=ep_steps, **acc)


@hydra.main(version_base="1.3", config_path="../configs", config_name="gen")
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    bc = cfg.bench
    w = run(cfg, int(bc.warmup_steps), device)
    r = run(cfg, int(bc.steps), device)
    B, n = int(bc.B), r["steps"]
    sps = n / r["total"]
    eph = B * 3600.0 * sps / r["ep_steps"]
    other = r["total"] - r["perception_edt"] - r["solver"] - r["plant"]
    pct = lambda x: f"{100.0 * x / r['total']:.1f}%"
    print(f"envs={B} steps={n} device={device} compile={bool(cfg.mpc.solver.compile)} "
          f"rt=({float(bc.control_rate_hz):g}Hz,{int(bc.solver_iters)}it,{float(bc.perception_rate_hz):g}Hz)")
    print(f"warmup_s={w['total']:.1f} wall_s={r['total']:.2f} steps_per_s={sps:.2f} env_steps_per_s={B * sps:.0f}")
    print(f"episode_steps={r['ep_steps']} (longest configured timeout in the batch) episodes_per_hour={eph:.0f}")
    print(f"split perception_edt={pct(r['perception_edt'])} solver={pct(r['solver'])} plant={pct(r['plant'])} "
          f"other={pct(other)}")
    if device.type == "cuda":
        print(f"max_mem_GiB={torch.cuda.max_memory_allocated(device) / 2**30:.2f}")


if __name__ == "__main__":
    main()
