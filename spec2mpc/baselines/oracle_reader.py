"""Reads oracle theta (theta* or the CMA mean at k evaluations) from the S1 outputs; theta columns and ids only."""
from pathlib import Path
from typing import Dict, Optional, Tuple

import pyarrow.parquet as pq
import torch
from torch import Tensor

from spec2mpc.mpc.theta import THETA_NAMES


def load_oracle_thetas(split_dir: Path, k: Optional[int]) -> Dict[Tuple[str, str], Tensor]:
    # -> {(problem_id, card_id): theta [14]} in THETA_NAMES order (raw values); theta* if k is None, else CMA mean at k
    t = pq.read_table(Path(split_dir) / ("oracle.parquet" if k is None else "at_k.parquet"))
    if k is not None:
        t = t.filter(t.column("k").to_numpy() == k)
        if t.num_rows == 0:
            raise ValueError(f"no rows for k={k} in {Path(split_dir) / 'at_k.parquet'}")
    theta = torch.stack([torch.tensor(t.column(n).to_numpy()) for n in THETA_NAMES], dim=-1).float()
    return {(p, c): theta[i] for i, (p, c) in enumerate(zip(t.column("problem_id").to_pylist(), t.column("card_id").to_pylist()))}
