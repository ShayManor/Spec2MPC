from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.features.phi import PHI_NAMES
from spec2mpc.mpc.theta import THETA_NAMES, normalize_theta
from spec2mpc.sim.conditions import Conditions
from spec2mpc.sim.storage import COND_SCALARS, INIT_OFFSET_NAMES, META_FILE, read_shard
from spec2mpc.tokenize.robot_tokens import robot_tokens_from_columns

XI_NAMES = COND_SCALARS + INIT_OFFSET_NAMES
ROBOT_COLUMN_PREFIXES = ("kart_", "perc_", "rt_")
TASK_IDS = {"lap": 0, "goto": 1}


def xi_from_conditions(cond: Conditions) -> Tensor:
    # cond [B] -> xi [B, len(XI_NAMES)] raw, in XI_NAMES order
    return torch.cat([torch.stack([getattr(cond, n) for n in COND_SCALARS], dim=-1), cond.init_offset], dim=-1)


def normalize_xi(xi: Tensor, sim_cfg: DictConfig) -> Tensor:
    # xi [..., len(XI_NAMES)] -> [..., len(XI_NAMES)] in [-1, 1] over the global prior ranges
    lo = torch.tensor([float(sim_cfg.conditions[n][0]) for n in XI_NAMES], dtype=xi.dtype, device=xi.device)
    hi = torch.tensor([float(sim_cfg.conditions[n][1]) for n in XI_NAMES], dtype=xi.dtype, device=xi.device)
    return (xi - (lo + hi) / 2) / ((hi - lo) / 2)


@dataclass
class RolloutTensors:
    robot_tok: Tensor  # [E, 6, 24]
    path_tok: Tensor  # [E, 64, 6]
    theta_n: Tensor  # [E, 14]
    xi_n: Tensor  # [E, 6]
    phi: Tensor  # [E, 20] raw
    task: Tensor  # [E] long, 0 lap / 1 goto
    block_id: Tensor  # [E] long
    episode_id: Tensor  # [E] long
    shard: Tensor  # [E] long, index into shard_dirs
    shard_dirs: List[Path]

    def index(self, idx: Tensor) -> "RolloutTensors":
        return RolloutTensors(self.robot_tok[idx], self.path_tok[idx], self.theta_n[idx], self.xi_n[idx], self.phi[idx],
                              self.task[idx], self.block_id[idx], self.episode_id[idx], self.shard[idx], self.shard_dirs)


def _complete_shards(root: Path) -> List[Path]:
    root = Path(root)
    if root.is_file():  # manifest: one shard directory per line
        return [Path(line) for line in root.read_text().splitlines() if line.strip()]
    if (root / META_FILE).exists():
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".") and (d / META_FILE).exists())


def load_rollouts(roots: Sequence[Path], tok_cfg: DictConfig, mpc_cfg: DictConfig, sim_cfg: DictConfig) -> RolloutTensors:
    # roots: shard directories, directories of shards or manifest files -> all complete (or listed) shards, in order
    dirs = [s for r in roots for s in _complete_shards(r)]
    parts = {k: [] for k in ("robot_tok", "path_tok", "theta_n", "xi_n", "phi", "task", "block_id", "episode_id", "shard")}
    for i, d in enumerate(dirs):
        table, z = read_shard(d)
        col = lambda n: torch.tensor(table.column(n).to_numpy())  # noqa: E731
        robot_cols = {n: col(n) for n in table.column_names if n.startswith(ROBOT_COLUMN_PREFIXES)}
        parts["robot_tok"].append(robot_tokens_from_columns(robot_cols, tok_cfg))
        parts["path_tok"].append(torch.from_numpy(np.asarray(z["path_tokens"][:])).float())
        parts["theta_n"].append(normalize_theta(torch.stack([col(n) for n in THETA_NAMES], dim=-1).float(), mpc_cfg))
        parts["xi_n"].append(normalize_xi(torch.stack([col(f"xi_{n}") for n in XI_NAMES], dim=-1).float(), sim_cfg))
        parts["phi"].append(torch.stack([col(f"phi_{n}") for n in PHI_NAMES], dim=-1).float())
        parts["task"].append(torch.tensor([TASK_IDS[t] for t in table.column("task").to_pylist()], dtype=torch.long))
        parts["block_id"].append(col("block_id").long())
        parts["episode_id"].append(col("episode_id").long())
        parts["shard"].append(torch.full((table.num_rows,), i, dtype=torch.long))
    return RolloutTensors(**{k: torch.cat(v) for k, v in parts.items()}, shard_dirs=dirs)


def block_split(block_id: Tensor, val_frac: float, seed: int) -> Tensor:
    # block_id [E] -> val [E] bool; one hash per block, so the 16 xi of a block share a side and adding shards keeps it stable
    uniq, inv = torch.unique(block_id, return_inverse=True)
    u = [np.random.SeedSequence([int(seed), int(b)]).generate_state(1, dtype=np.uint32)[0] / 2.0 ** 32 for b in uniq]
    return (torch.tensor(u) < val_frac)[inv]


def load_theta_star(oracle_parquet: Path) -> Tuple[List[str], List[str], List[str], Tensor]:
    # -> problem_id, card_id, task (each [R]) and theta* [R, 14] in THETA_NAMES order (raw values)
    t = pq.read_table(oracle_parquet, columns=["problem_id", "card_id", "task", *THETA_NAMES])
    theta = torch.stack([torch.tensor(t.column(n).to_numpy()) for n in THETA_NAMES], dim=-1).float()
    return t.column("problem_id").to_pylist(), t.column("card_id").to_pylist(), t.column("task").to_pylist(), theta
