from pathlib import Path
from typing import List, Sequence, Tuple

import pyarrow.parquet as pq
import torch
from torch import Tensor

from spec2mpc.model.data import TASK_IDS, RolloutTensors
from spec2mpc.rubric.cards import Card, load_cards
from spec2mpc.rubric.scoring import card_max_score
from spec2mpc.sim.storage import META_FILE

SCORES_DIR = "scores"  # same value as scripts/score_rollouts.py SCORES_DIR


def load_score_labels(data: RolloutTensors, cards: Sequence[Card]) -> Tuple[Tensor, Tensor]:
    # -> J / card_max_score [E, C] (NaN off-task) and in_conditions [E, C] bool (False off-task), joined on episode_id per shard
    E = data.episode_id.shape[0]
    J = torch.full((E, len(cards)), float("nan"))
    ok = torch.zeros(E, len(cards), dtype=torch.bool)
    for si, shard in enumerate(data.shard_dirs):
        rows = (data.shard == si).nonzero().squeeze(1)
        eid = data.episode_id[rows]
        for ci, card in enumerate(cards):
            f = Path(shard) / SCORES_DIR / f"{card.id}.parquet"
            if not f.exists():
                raise FileNotFoundError(f"{shard}: no score file for {card.id}; run scripts/score_rollouts.py")
            t = pq.read_table(f)
            sid = torch.tensor(t.column("episode_id").to_numpy()).long()
            j = torch.tensor(t.column("J").to_numpy()).float() / card_max_score(card)
            o = torch.tensor(t.column("in_conditions").to_numpy())
            pos = torch.searchsorted(sid.sort().values, eid)
            order = sid.argsort()
            pos = pos.clamp(max=max(len(sid) - 1, 0))
            on_task = data.task[rows] == TASK_IDS[card.task]
            hit = on_task & (len(sid) > 0) & (sid[order][pos] == eid) if len(sid) else on_task & False
            src = order[pos] if len(sid) else pos
            if (on_task & ~hit).any():
                raise ValueError(f"{shard}: {card.id} score file misses {int((on_task & ~hit).sum())} on-task episodes")
            J[rows[hit], ci] = j[src[hit]]
            ok[rows[hit], ci] = o[src[hit]]
    return J, ok


def scored_shards(root: Path, card_ids: Sequence[str]) -> List[Path]:
    # -> sorted shard dirs under root with meta.json and a score file for every card id (temp dirs skipped)
    root = Path(root)
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".") and (d / META_FILE).exists()
                  and all((d / SCORES_DIR / f"{c}.parquet").exists() for c in card_ids))


def write_manifest(root: Path, cards_root: Path, out: Path) -> List[Path]:
    # freezes the training shard set: the scored_shards of root for every train and val card, one path per line
    ids = [c.id for split in ("train", "val") for c in load_cards(Path(cards_root), split)]
    dirs = scored_shards(root, ids)
    if not dirs:
        raise ValueError(f"no shard under {root} has meta.json and scores for all {len(ids)} train and val cards")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{d}\n" for d in dirs))
    return dirs
