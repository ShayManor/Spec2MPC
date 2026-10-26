"""Rank correlation and group bootstrap for evaluation metrics."""
from typing import Tuple

import torch
from torch import Tensor


def _ranks(x: Tensor) -> Tensor:
    # x [N] -> [N] float64 ranks starting at 1, ties get their average rank
    x = x.double()
    uniq, inv, counts = torch.unique(x, return_inverse=True, return_counts=True)
    first = torch.cumsum(counts, 0) - counts  # [U] number of values below each unique value
    return (first + (counts + 1) / 2).double()[inv]


def spearman(a: Tensor, b: Tensor) -> float:
    # a, b [N] -> Pearson correlation of average ranks
    ra, rb = _ranks(a.flatten().cpu()), _ranks(b.flatten().cpu())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def bootstrap_ci(values: Tensor, groups: Tensor, n_boot: int, alpha: float, gen: torch.Generator) -> Tuple[float, float, float]:
    # values [N], groups [N] long -> (mean, lo, hi); resamples whole groups with replacement, CI is the percentile interval
    values, groups = values.double().cpu(), groups.cpu()
    uniq, inv = torch.unique(groups, return_inverse=True)
    G = uniq.numel()
    sums = torch.zeros(G, dtype=torch.float64).index_add_(0, inv, values)
    counts = torch.zeros(G, dtype=torch.float64).index_add_(0, inv, torch.ones_like(values))
    pick = torch.randint(0, G, (n_boot, G), generator=gen)  # [n_boot, G]
    means = sums[pick].sum(1) / counts[pick].sum(1)  # [n_boot]
    q = torch.tensor([alpha / 2, 1 - alpha / 2], dtype=torch.float64)
    lo, hi = means.quantile(q).tolist()
    return float(values.mean()), lo, hi


def disagreement_pairs(j_own: Tensor, j_other: Tensor, margin: float) -> Tensor:
    # j_own, j_other [N] -> pairs [N, N] bool: own J ranks i above j and the other card's J ranks j above i, both by > margin
    d_own = j_own[:, None] - j_own[None, :]
    d_other = j_other[:, None] - j_other[None, :]
    return (d_own > margin) & (d_other < -margin)


def pair_accuracy(score: Tensor, pairs: Tensor) -> Tuple[float, int]:
    # score [N], pairs [N, N] bool (i should outrank j) -> (fraction ordered correctly, ties count half; NaN if none), n pairs
    i, j = pairs.nonzero(as_tuple=True)
    if i.numel() == 0:
        return float("nan"), 0
    d = score[i] - score[j]
    return float(((d > 0).double() + 0.5 * (d == 0).double()).mean()), int(i.numel())
