import pytest
import scipy.stats
import torch

from spec2mpc.eval.stats import bootstrap_ci, disagreement_pairs, pair_accuracy, spearman


def test_spearman_matches_scipy():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(200, generator=g), torch.randn(200, generator=g)
    b[::7] = b[0]  # ties
    a[::5] = a[1]
    assert spearman(a, b) == pytest.approx(scipy.stats.spearmanr(a, b).statistic, abs=1e-9)


def test_spearman_monotone_is_one():
    a = torch.randn(50, generator=torch.Generator().manual_seed(1))
    assert spearman(a, a.exp()) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)


def test_bootstrap_ci_contains_mean_and_resamples_groups():
    values = torch.tensor([0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    mean, lo, hi = bootstrap_ci(values, groups, 2000, 0.05, torch.Generator().manual_seed(0))
    assert mean == pytest.approx(0.5)
    assert lo <= mean <= hi and lo < hi
    assert lo >= 0.1 - 1e-9 and hi <= 0.9 + 1e-9
    # constant within-group values: a resample of whole groups can only produce means on the 1/3 grid of group means
    one = torch.tensor([0.3, 0.3, 0.3, 0.3])
    m1, l1, h1 = bootstrap_ci(one, torch.tensor([0, 0, 1, 1]), 100, 0.05, torch.Generator().manual_seed(0))
    assert m1 == l1 == h1 == pytest.approx(0.3)


def test_disagreement_pairs_need_opposite_orders_beyond_margin():
    j_own = torch.tensor([1.0, 0.0, 0.5, 0.49])
    j_other = torch.tensor([0.0, 1.0, 0.5, 0.0])
    p = disagreement_pairs(j_own, j_other, 0.02)
    assert p.shape == (4, 4) and p.dtype == torch.bool
    # own: 0 > 1 and other: 1 > 0 -> pair (0, 1); own 2 vs 3 differ by 0.01 < margin -> no pair (2, 3)
    assert set(map(tuple, p.nonzero().tolist())) == {(0, 1), (0, 2), (2, 1), (3, 1)}


def test_pair_accuracy_counts_ties_half():
    pairs = torch.zeros(3, 3, dtype=torch.bool)
    pairs[0, 1] = pairs[0, 2] = pairs[1, 2] = True
    acc, n = pair_accuracy(torch.tensor([2.0, 1.0, 1.0]), pairs)
    assert n == 3 and acc == pytest.approx((1 + 1 + 0.5) / 3)
    acc0, n0 = pair_accuracy(torch.tensor([2.0, 1.0, 1.0]), torch.zeros(3, 3, dtype=torch.bool))
    assert n0 == 0 and acc0 != acc0  # NaN when there are no pairs
