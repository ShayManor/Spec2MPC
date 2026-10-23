from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from spec2mpc.rubric.cards import Card


def score(card: Card, metrics: Mapping[str, Tensor]) -> Tensor:
    """Per-episode J. metrics: name -> [E]. Returns [E]."""
    env = dict(metrics)
    ref = next(iter(env.values()))
    passed = torch.ones_like(ref, dtype=torch.bool)
    n_pass = torch.zeros(ref.shape, dtype=torch.float64)
    first_passed = torch.zeros_like(ref, dtype=torch.bool)
    first_fail = torch.full(ref.shape, len(card.rubric.tiers), dtype=torch.long)
    for i, tier in enumerate(card.rubric.tiers):
        passed = passed & tier.expr(env)  # NaN compares False
        first_fail = torch.where(~passed & (first_fail == len(card.rubric.tiers)), i, first_fail)
        n_pass = n_pass + passed.to(torch.float64)
        if i == 0:
            first_passed = passed
    j = n_pass
    if card.rubric.shaping is not None:
        thr = [t.threshold if t.metric in card.rubric.shaping.metrics else None for t in card.rubric.tiers]
        # T_next: threshold of the first tier on a shaping metric at or after the first failure; else the last one
        last = [x for x in thr if x is not None][-1] if any(x is not None for x in thr) else float("nan")
        t_next = torch.full(ref.shape, last, dtype=torch.float64)
        for i in reversed(range(len(thr))):
            if thr[i] is not None:
                t_next = torch.where(first_fail <= i, torch.tensor(thr[i], dtype=torch.float64), t_next)
        env["T_next"] = t_next
        shape = torch.nan_to_num(card.rubric.shaping.expr(env).to(torch.float64), nan=0.0)
        j = j + card.rubric.shaping.weight * shape * first_passed  # no shaping credit if tier 0 fails
    for p in card.rubric.penalties:
        j = j - p.weight * torch.nan_to_num(p.expr(env).to(torch.float64), nan=0.0)
    return j


def in_card_conditions(card: Card, cond: Mapping[str, Tensor]) -> Tensor:
    """cond: Conditions field name -> [E]. Returns [E] bool."""
    ref = next(iter(cond.values()))
    ok = torch.ones_like(ref, dtype=torch.bool)
    for k, v in card.rubric.conditions.items():
        if k != "episodes":
            ok = ok & (cond[k] >= v[0]) & (cond[k] <= v[1])
    return ok


def card_score(card: Card, metrics: Mapping[str, Tensor], cond: Mapping[str, Tensor]) -> tuple[float, int, bool]:
    ok = in_card_conditions(card, cond)  # [E]
    n_used = int(ok.sum())
    mean_j = float(score(card, metrics)[ok].mean()) if n_used else float("nan")
    return mean_j, n_used, n_used < card.rubric.conditions["episodes"]
