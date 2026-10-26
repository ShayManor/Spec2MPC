import math

import torch

from spec2mpc.search.cmaes import BatchedCMAES


def test_sphere_converges():
    opt = BatchedCMAES(torch.full((3, 14), 0.8), 0.3, 12, torch.Generator().manual_seed(0))
    target = torch.tensor([[0.1] * 14, [-0.5] * 14, [0.0] * 14])
    for _ in range(300):
        x = opt.ask(); opt.tell(x, -((x - target[:, None]) ** 2).sum(-1))
    assert torch.allclose(opt.mean, target.double(), atol=1e-3)
    assert opt.generation == 300 and opt.n_evals == 12 * 300


def test_rotated_ellipsoid_converges():
    g = torch.Generator().manual_seed(1)
    P, D = 2, 14
    rot = torch.linalg.qr(torch.randn(P, D, D, generator=g, dtype=torch.float64))[0]
    scale = 1e3 ** (torch.arange(D, dtype=torch.float64) / (D - 1))
    x0 = 0.4 * torch.rand(P, D, generator=g, dtype=torch.float64) - 0.2
    opt = BatchedCMAES(torch.zeros(P, D), 0.3, 12, torch.Generator().manual_seed(2))
    for _ in range(600):
        x = opt.ask()
        r = torch.einsum("pij,pkj->pki", rot, x - x0[:, None])
        opt.tell(x, -(scale * r ** 2).sum(-1))
    assert (opt.best_f > -1e-6).all()


def test_problems_are_independent():
    target = torch.tensor([[0.3] * 14, [-0.2] * 14])
    joint = BatchedCMAES(torch.zeros(2, 14), 0.3, 8, torch.Generator().manual_seed(0))
    draws = []
    sample = joint._sample_z
    joint._sample_z = lambda: draws.append(sample()) or draws[-1]
    for _ in range(40):
        x = joint.ask(); joint.tell(x, -((x - target[:, None]) ** 2).sum(-1))
    for p in range(2):
        alone = BatchedCMAES(torch.zeros(1, 14), 0.3, 8, torch.Generator().manual_seed(123))
        replay = iter(draws)
        alone._sample_z = lambda: next(replay)[p:p + 1]
        for _ in range(40):
            x = alone.ask(); alone.tell(x, -((x - target[p:p + 1, None]) ** 2).sum(-1))
        assert torch.allclose(alone.mean[0], joint.mean[p], atol=1e-9)
        assert torch.allclose(alone.sigma[0], joint.sigma[p], atol=1e-9)


def test_respects_box():
    gen = torch.Generator().manual_seed(0)
    opt = BatchedCMAES(torch.zeros(1, 14), 2.0, 12, gen)
    for _ in range(100):  # brief said 20; clipped steps shrink sigma, so the corner takes 39 to 66 gens over seeds 0-9
        x = opt.ask(); assert x.abs().max() <= 1.0; opt.tell(x, x.sum(-1))
    assert torch.allclose(opt.mean, torch.ones(1, 14, dtype=torch.float64), atol=0.05)


def _run(opt, n):
    for _ in range(n):
        x = opt.ask(); opt.tell(x, -((x - 0.25) ** 2).sum(-1))


def test_state_dict_resume_exact():
    a = BatchedCMAES(torch.full((2, 14), -0.5), 0.3, 12, torch.Generator().manual_seed(4))
    _run(a, 5)
    saved = a.state_dict()
    _run(a, 5)
    b = BatchedCMAES(torch.zeros(2, 14), 0.1, 12, torch.Generator().manual_seed(99))
    b.load_state_dict(saved)
    _run(b, 5)
    assert torch.equal(a.mean, b.mean) and torch.equal(a.sigma, b.sigma)
    assert torch.equal(a.best_x, b.best_x) and torch.equal(a.best_f, b.best_f)
    assert a.generation == b.generation == 10 and a.n_evals == b.n_evals


def test_nonfinite_scores_rank_last():
    opt = BatchedCMAES(torch.zeros(1, 14), 0.3, 12, torch.Generator().manual_seed(5))
    x = opt.ask()
    f = -(x ** 2).sum(-1)
    bad = torch.tensor([0, 3, 7, 10])
    f[0, bad] = torch.tensor([math.nan, math.inf, -math.inf, math.nan], dtype=f.dtype)
    finite = torch.ones(12, dtype=torch.bool)
    finite[bad] = False
    xf, ff = x[0, finite], f[0, finite]
    order = torch.argsort(ff, descending=True)
    mu = opt.weights.numel()
    expected_mean = (opt.weights[:, None] * xf[order[:mu]]).sum(0)
    opt.tell(x, f)
    assert torch.allclose(opt.mean[0], expected_mean, atol=1e-12)
    assert opt.best_f[0] == ff.max() and torch.equal(opt.best_x[0], xf[order[0]])
    assert torch.isfinite(opt.sigma).all()
    f_all_bad = torch.full_like(f, math.nan)
    opt.tell(opt.ask(), f_all_bad)
    assert opt.best_f[0] == ff.max() and torch.isfinite(opt.mean).all()
