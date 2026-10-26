"""S0 rollout generation: one shard per batch of envs_per_batch episodes.

Usage: uv run python scripts/gen_rollouts.py n_blocks=8 device=cuda mpc.solver.compile=true out_dir=...
RESUME=1 skips batches whose shard already has meta.json.
"""
import os
import socket
import time
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from spec2mpc.mpc.theta import ThetaBatch, sobol_theta
from spec2mpc.robots.kart import Robot, sample_karts, sample_runtime
from spec2mpc.sim.conditions import Conditions, sample_conditions
from spec2mpc.sim.evaluate import rollout_metrics
from spec2mpc.sim.scenes import TASK_LAP, Scene, generate_scenes
from spec2mpc.sim.storage import META_FILE, read_meta, write_shard
from spec2mpc.tokenize.path_tokens import path_tokens

TASKS = ("lap", "goto")


def batch_seed(seed: int, block: int) -> int:
    # independent 63-bit stream per (seed, batch index), so any batch can be regenerated alone
    return int(np.random.SeedSequence([seed, block]).generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def task_counts(n: int, task_mix: DictConfig) -> dict:
    # n blocks split across tasks by task_mix weights; the last listed task takes the remainder
    unknown = set(task_mix.keys()) - set(TASKS)
    assert not unknown, f"unknown task in task_mix: {unknown}"
    w = {t: float(task_mix.get(t, 0.0)) for t in TASKS}
    total = sum(w.values())
    assert total > 0, "task_mix weights sum to 0"
    n_lap = int(round(n * w["lap"] / total))
    return {"lap": n_lap, "goto": n - n_lap}


def sample_batch(cfg: DictConfig, block: int, n_envs: int) -> Tuple[Robot, Scene, ThetaBatch, Conditions, Tensor, Tensor, torch.Generator]:
    # -> robot, scene, theta, cond [n_envs] on CPU, episode_id [n_envs], block_id [n_envs], generator for the rollout
    xi = int(cfg.xi_per_block)
    assert n_envs % xi == 0, f"envs_per_batch {n_envs} must be a multiple of xi_per_block {xi}"
    n_tr = n_envs // xi
    gen = torch.Generator().manual_seed(batch_seed(int(cfg.seed), block))
    rt = sample_runtime(cfg.robots, gen)
    robots = sample_karts(n_tr, cfg.robots, rt, gen)
    scenes = [generate_scenes(n, cfg.sim, gen, t) for t, n in task_counts(n_tr, cfg.task_mix).items() if n > 0]
    scene = Scene.cat(scenes) if len(scenes) > 1 else scenes[0]
    # one scrambled Sobol sequence per seed; batch `block` takes points [block * n_tr, (block + 1) * n_tr)
    theta = sobol_theta((block + 1) * n_tr, cfg.mpc, int(cfg.seed)).index(slice(block * n_tr, None))
    assert torch.isfinite(theta.values).all(), "non-finite theta"
    rep = torch.arange(n_tr).repeat_interleave(xi)  # [n_envs] block index within the batch
    cond = sample_conditions(n_envs, cfg.sim, gen)
    episode_id = block * n_envs + torch.arange(n_envs, dtype=torch.int64)
    block_id = block * n_tr + rep
    return robots.index(rep), scene.index(rep), theta.index(rep), cond, episode_id, block_id, gen


def run_batch(cfg: DictConfig, block: int, out_dir: Path, device: torch.device) -> None:
    sid = f"shard_{block:06d}"
    t0 = time.perf_counter()
    robot, scene, theta, cond, episode_id, block_id, gen = sample_batch(cfg, block, int(cfg.envs_per_batch))
    r, sc = robot.to(device), scene.to(device)
    out = rollout_metrics(robot, scene, theta, cond, cfg, gen, device)
    traj, stats, metrics, ph = out.traj, out.stats, out.metrics, out.phi
    tok = path_tokens(sc.path, sc, r.params, cfg.tokenize)
    rt = robot.runtime
    meta = {
        "family": cfg.family, "block": block, "seed": int(cfg.seed), "batch_seed": batch_seed(int(cfg.seed), block),
        "device": str(device), "host": socket.gethostname(), "runtime": vars(rt),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    write_shard(out_dir, sid, robot, scene, theta, cond, traj, metrics, ph, tok, episode_id=episode_id,
                block_id=block_id, solver_stats=stats, store_hz=float(cfg.sim.store_hz), meta=meta)
    wall = time.perf_counter() - t0
    comp = metrics["completion"].cpu()
    lap = scene.task == TASK_LAP
    frac = lambda m: f"{float(comp[m].mean()):.3f}" if bool(m.any()) else "nan"
    fails = int(((~traj.solver_ok) & traj.valid).sum())
    steps = int(traj.valid.sum())
    print(f"{sid} envs={theta.batch_size} wall_s={wall:.1f} eps_per_s={theta.batch_size / wall:.2f} "
          f"complete_lap={frac(lap)} complete_goto={frac(~lap)} solver_fail_rate={fails / max(steps, 1):.4f} "
          f"rt=({rt.control_rate_hz:g}Hz,{rt.solver_iters}it,{rt.perception_rate_hz:g}Hz)", flush=True)


RESUME_KEYS = ("seed", "envs_per_batch", "xi_per_block", "task_mix")


def check_resumable(shard: Path, cfg: DictConfig) -> None:
    # a complete shard is skipped only if it came from the same seed, batch layout and task mix as this run
    saved = read_meta(shard)["config"]
    now = OmegaConf.to_container(cfg, resolve=True)
    diff = {k: (saved.get(k), now[k]) for k in RESUME_KEYS if saved.get(k) != now[k]}
    if diff:
        raise ValueError(f"{shard} was generated with different settings (saved, current): {diff}; "
                         f"use the original settings or another out_dir")


@hydra.main(version_base="1.3", config_path="../configs", config_name="gen")
def main(cfg: DictConfig) -> None:
    if bool(cfg.deterministic):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only: grid_sample's CUDA backward (sdf_lookup) has no deterministic kernel and would raise;
        # byte-identity of the outputs is checked by rerunning, not guaranteed by this flag
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = os.environ.get("RESUME", "0") == "1"
    for block in range(int(cfg.first_block), int(cfg.first_block) + int(cfg.n_blocks)):
        shard = out_dir / f"shard_{block:06d}"
        if (shard / META_FILE).exists():
            if resume:
                check_resumable(shard, cfg)
                print(f"{shard.name} skip (RESUME=1, complete)", flush=True)
                continue
            raise FileExistsError(f"{shard} exists; set RESUME=1 to skip complete shards or pick another out_dir")
        run_batch(cfg, block, out_dir, device)


if __name__ == "__main__":
    main()
