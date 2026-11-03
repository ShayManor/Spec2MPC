"""Pooled success rate and mean score with task-level bootstrap CIs; loading a round's result parquets."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from spec2mpc.eval.stats import bootstrap_ci

N_BOOT = 2000
ALPHA = 0.05


def pooled(df: pa.Table | list[dict], n_boot: int = N_BOOT, alpha: float = ALPHA, seed: int = 0) -> dict:
    # rows with task_id, success, score -> means and percentile CIs, resampling whole tasks (seeds stay together)
    rows = df.to_pylist() if isinstance(df, pa.Table) else list(df)
    ids = {t: i for i, t in enumerate(sorted({r["task_id"] for r in rows}))}
    groups = torch.tensor([ids[r["task_id"]] for r in rows], dtype=torch.long)  # [N]
    out: dict = {"n_tasks": len(ids), "n_rows": len(rows)}
    for name, key in (("success_rate", "success"), ("score", "score")):
        vals = torch.tensor([float(r[key]) for r in rows], dtype=torch.float64)  # [N]
        gen = torch.Generator().manual_seed(seed)
        m, lo, hi = bootstrap_ci(vals, groups, n_boot, alpha, gen)
        out[name], out[f"{name}_lo"], out[f"{name}_hi"] = m, lo, hi
    return out


def load_round(round_dir: str | Path) -> pa.Table:
    # every <executor>_<split>.parquet in the round; refuses executors evaluated on different task sets of a split
    tables = [pq.read_table(p) for p in sorted(Path(round_dir).glob("*.parquet"))]
    if not tables:
        raise FileNotFoundError(f"no result parquets in {round_dir}")
    t = pa.concat_tables(tables, promote_options="default")
    sets: dict[str, dict[str, set]] = {}
    for r in t.select(["split", "executor", "task_id"]).to_pylist():
        sets.setdefault(r["split"], {}).setdefault(r["executor"], set()).add(r["task_id"])
    for split, by_ex in sets.items():
        ref_ex, ref = next(iter(by_ex.items()))
        for ex, s in by_ex.items():
            if s != ref:
                raise ValueError(f"split {split}: {ex} and {ref_ex} ran different task sets "
                                 f"({len(s ^ ref)} task ids differ); refusing to compare")
    return t
