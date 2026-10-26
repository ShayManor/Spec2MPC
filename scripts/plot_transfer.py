"""Transfer matrix go/no-go: per-card diagonal margins, verdict per task and overall, and data/previews/transfer.png.

Usage: uv run python scripts/plot_transfer.py ORACLE_DIR   ({out_dir}/train of a train oracle run)
"""
import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from spec2mpc.eval.transfer import diag_margin, plot_transfer, transfer_matrix, transfer_verdict


def report(scope: str, margins: torch.Tensor, go_margin: float) -> None:
    verdict, n_valid, med = transfer_verdict(margins, go_margin)
    print(f"{scope} verdict={verdict} n_valid={n_valid}/{margins.numel()} median={med:.3f} go_margin={go_margin:g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/previews/transfer.png"))
    a = ap.parse_args()
    cfg = OmegaConf.load(a.oracle_dir / "config.yaml")
    table = pq.read_table(a.oracle_dir / "transfer.parquet")
    rows = table.to_pylist()
    by_task, margins = {}, []
    for task in ("lap", "goto"):
        cards = sorted({r["rubric_card"] for r in rows if r["task"] == task})
        if not cards:
            continue
        T, skipped = transfer_matrix(table, cards, float(cfg.transfer.min_gap), return_skipped=True)
        m = diag_margin(T)
        by_task[task] = (T, cards)
        margins.append(m)
        for c, v, n in zip(cards, m.tolist(), skipped.diagonal().tolist()):
            print(f"{task} {c} margin={v:.3f} skipped_problems={n}")
        report(task, m, float(cfg.transfer.go_margin))
    report("overall", torch.cat(margins), float(cfg.transfer.go_margin))
    plot_transfer(by_task, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
