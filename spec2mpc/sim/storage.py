import json
import shutil
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import zarr
from torch import Tensor

from spec2mpc.features.metrics import METRIC_NAMES
from spec2mpc.features.phi import PHI_NAMES
from spec2mpc.mpc.theta import THETA_NAMES, ThetaBatch
from spec2mpc.robots.kart import Robot
from spec2mpc.sim.conditions import Conditions
from spec2mpc.sim.rollout import SolverStats
from spec2mpc.sim.scenes import TASK_LAP, Scene
from spec2mpc.sim.trajectory import Trajectory, downsample

EPISODES_FILE = "episodes.parquet"
TRAJ_FILE = "traj.zarr"
META_FILE = "meta.json"
INIT_OFFSET_NAMES = ("init_lateral_m", "init_heading_rad", "init_speed_mps")
COND_SCALARS = ("friction_scale", "mass_scale", "latency_s")
TRAJ_CHUNK_EPISODES = 256


def git_hash() -> str:
    # HEAD of the repo holding this file, "+dirty" when the tree has changes; "unknown" outside git
    root = Path(__file__).resolve().parents[2]
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout
        return h + ("+dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _np(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def episode_columns(robot: Robot, scene: Scene, theta: ThetaBatch, cond: Conditions, traj: Trajectory,
                    metrics: Mapping[str, Tensor], phi: Tensor, episode_id: Tensor, block_id: Tensor,
                    solver_stats: SolverStats) -> Dict[str, Any]:
    # every input batched over B episodes; phi [B, len(PHI_NAMES)] -> {column: [B] array}
    B = theta.batch_size
    cols: Dict[str, Any] = {"episode_id": _np(episode_id).astype(np.int64), "block_id": _np(block_id).astype(np.int64),
                            "robot_id": list(robot.robot_id)}
    for f in fields(robot.params):
        cols[f"kart_{f.name}"] = _np(getattr(robot.params, f.name))
    for f in fields(robot.perception):
        cols[f"perc_{f.name}"] = _np(getattr(robot.perception, f.name))
    rt = robot.runtime
    cols["rt_control_rate_hz"] = np.full(B, rt.control_rate_hz, dtype=np.float32)
    cols["rt_solver_iters"] = np.full(B, rt.solver_iters, dtype=np.int64)
    cols["rt_perception_rate_hz"] = np.full(B, rt.perception_rate_hz, dtype=np.float32)
    cols["scene_seed"] = _np(scene.seed).astype(np.int64)
    cols["task"] = ["lap" if int(t) == TASK_LAP else "goto" for t in scene.task]
    for i, n in enumerate(THETA_NAMES):
        cols[n] = _np(theta.values[:, i])
    for n in COND_SCALARS:
        cols[f"xi_{n}"] = _np(getattr(cond, n))
    for i, n in enumerate(INIT_OFFSET_NAMES):
        cols[f"xi_{n}"] = _np(cond.init_offset[:, i])
    cols["xi_seed"] = _np(cond.seed).astype(np.int64)
    for i, n in enumerate(PHI_NAMES):
        cols[f"phi_{n}"] = _np(phi[:, i])
    for n in METRIC_NAMES:
        cols[n] = _np(metrics[n])
    cols["solver_fail_count"] = _np(((~traj.solver_ok) & traj.valid).sum(dim=1)).astype(np.int64)
    cols["mean_cost"] = _np(solver_stats.cost_mean)
    cols["mean_grad_norm"] = _np(solver_stats.grad_norm_mean)
    for k, v in cols.items():
        assert len(v) == B, f"column {k} has {len(v)} rows, expected {B}"
    return cols


def write_shard(out_dir: Path, shard_id: str, robot: Robot, scene: Scene, theta: ThetaBatch, cond: Conditions,
                traj: Trajectory, metrics: Mapping[str, Tensor], phi: Tensor, path_tok: Tensor, *,
                episode_id: Tensor, block_id: Tensor, solver_stats: SolverStats, store_hz: float,
                meta: Mapping[str, Any]) -> None:
    # traj [B, T, ...] at the control rate, path_tok [B, K, 6] -> out_dir/shard_id/{episodes.parquet, traj.zarr, meta.json}
    # Written into out_dir/.tmp_<shard_id> and renamed, so a shard directory always holds a complete shard.
    out_dir = Path(out_dir)
    final, tmp = out_dir / shard_id, out_dir / f".tmp_{shard_id}"
    if final.exists():
        raise FileExistsError(f"shard {final} exists")
    if tmp.exists():
        shutil.rmtree(tmp)  # partial output of a killed run
    tmp.mkdir(parents=True)

    cols = episode_columns(robot, scene, theta, cond, traj, metrics, phi, episode_id, block_id, solver_stats)
    pq.write_table(pa.table(cols), tmp / EPISODES_FILE)

    ds = downsample(traj, store_hz)
    z = zarr.open_group(str(tmp / TRAJ_FILE), mode="w")
    for name, arr in (("x", ds.x), ("u", ds.u), ("valid", ds.valid), ("path_tokens", path_tok)):
        a = _np(arr)
        z.create_array(name, data=a, chunks=(min(a.shape[0], TRAJ_CHUNK_EPISODES),) + a.shape[1:])
    z.attrs["dt_s"] = ds.dt_s
    z.attrs["store_hz"] = float(store_hz)

    full_meta = dict(meta, shard_id=shard_id, n_episodes=theta.batch_size, git_hash=git_hash(),
                     torch_version=torch.__version__)
    (tmp / META_FILE).write_text(json.dumps(full_meta, indent=1, default=str))  # meta last: marks the shard complete
    tmp.rename(final)


def read_shard(shard_dir: Path) -> Tuple[pa.Table, zarr.Group]:
    shard_dir = Path(shard_dir)
    return pq.read_table(shard_dir / EPISODES_FILE), zarr.open_group(str(shard_dir / TRAJ_FILE), mode="r")


def read_meta(shard_dir: Path) -> Dict[str, Any]:
    return json.loads((Path(shard_dir) / META_FILE).read_text())


def cond_from_table(table: pa.Table) -> Dict[str, Tensor]:
    # -> {Conditions scalar field name: [E]}
    return {n: torch.tensor(table.column(f"xi_{n}").to_numpy()) for n in COND_SCALARS}


def metrics_from_table(table: pa.Table) -> Dict[str, Tensor]:
    # -> {metric name: [E]} in METRIC_NAMES order
    return {n: torch.tensor(table.column(n).to_numpy()) for n in METRIC_NAMES}
