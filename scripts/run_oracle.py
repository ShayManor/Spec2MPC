"""S1 oracle: CMA-ES per (problem, card) on the true rubric score under true sim rollouts.

Usage: uv run python scripts/run_oracle.py split=train device=cuda mpc.solver.compile=true out_dir=...
RESUME=1 continues from {out_dir}/{split}/state.pt; without it an existing state.pt is an error.
"""
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from spec2mpc.baselines.oracle_bo import build_job, run_oracle


@hydra.main(version_base="1.3", config_path="../configs", config_name="oracle")
def main(cfg: DictConfig) -> None:
    job = build_job(cfg)
    print(f"oracle split={cfg.split} problems={job.problems.n} cards={len(job.cards)} pairs={len(job.pairs)} "
          f"device={cfg.device}", flush=True)
    run_oracle(job, cfg, Path(cfg.out_dir), torch.device(cfg.device), resume=os.environ.get("RESUME", "0") == "1")


if __name__ == "__main__":
    main()
