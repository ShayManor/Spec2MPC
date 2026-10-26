from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from spec2mpc.rubric.cards import load_cards


@dataclass(frozen=True)
class StatementRecord:
    card_id: str
    task: str
    text: str
    role: str  # "train" | "val_paraphrase" | "val_intent"; no rubric fields


def statement_records(split_cfg: DictConfig) -> list[StatementRecord]:
    root = Path(split_cfg.cards_root)
    val_cards = load_cards(root, "val")
    if sorted(c.id for c in val_cards) != sorted(split_cfg.val_intents):
        raise ValueError(f"val_intents {sorted(split_cfg.val_intents)} differ from cards in val: {sorted(c.id for c in val_cards)}")
    recs: list[StatementRecord] = []
    for c in load_cards(root, "train"):
        held = range(len(c.statements))[split_cfg.heldout_statement_index]
        recs += [StatementRecord(c.id, c.task, s, "val_paraphrase" if i == held else "train") for i, s in enumerate(c.statements)]
    for c in val_cards:
        recs += [StatementRecord(c.id, c.task, s, "val_intent") for s in c.statements]
    return recs
