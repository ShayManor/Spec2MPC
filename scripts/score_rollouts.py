"""Train-card scoring pass: writes <shard>/scores/<card_id>.parquet (episode_id, J, in_conditions).

Usage: uv run python scripts/score_rollouts.py data/rollouts/kart [--val] [--cards-root data/cards]
Only the train split (and val with --val) is read; test splits are never loaded.
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from spec2mpc.rubric.cards import Card, load_cards
from spec2mpc.rubric.scoring import in_card_conditions, score
from spec2mpc.sim.storage import EPISODES_FILE, META_FILE, cond_from_table, metrics_from_table, read_meta

SCORES_DIR = "scores"


def shard_dirs(root: Path) -> List[Path]:
    # root itself if it is a complete shard, else its complete shard subdirectories (temp dirs are skipped)
    if (root / META_FILE).exists():
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".") and (d / META_FILE).exists())


def score_shard(shard: Path, cards: List[Card]) -> int:
    # -> number of score files written
    family = read_meta(shard)["family"]
    table = pq.read_table(shard / EPISODES_FILE)
    out = shard / SCORES_DIR
    out.mkdir(exist_ok=True)
    n = 0
    for card in cards:
        if family not in card.family:
            continue
        rows = table.filter(pc.equal(table.column("task"), card.task))
        j = score(card, metrics_from_table(rows)).numpy() if rows.num_rows else np.zeros(0)
        ok = in_card_conditions(card, cond_from_table(rows)).numpy() if rows.num_rows else np.zeros(0, dtype=bool)
        res = pa.table({"episode_id": rows.column("episode_id"), "J": pa.array(j, pa.float64()),
                        "in_conditions": pa.array(ok, pa.bool_())})
        tmp = out / f".tmp_{card.id}.parquet"
        pq.write_table(res, tmp)
        tmp.replace(out / f"{card.id}.parquet")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rollouts", type=Path, help="a shard directory or a directory of shards")
    ap.add_argument("--cards-root", type=Path, default=Path("data/cards"))
    ap.add_argument("--val", action="store_true", help="also score val cards")
    a = ap.parse_args()
    cards = load_cards(a.cards_root, "train") + (load_cards(a.cards_root, "val") if a.val else [])
    for shard in shard_dirs(a.rollouts):
        print(f"{shard.name} score_files={score_shard(shard, cards)}", flush=True)


if __name__ == "__main__":
    main()
