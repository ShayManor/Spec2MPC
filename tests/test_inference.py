import torch

from spec2mpc.search.inference import MODES, amortized, bon_pred, bon_true, refine
from tests.test_policy import B, ToyV, ToyW, cfg_of, ctx_batch
from spec2mpc.model.policy import build_policy


def setup():
    torch.manual_seed(0)
    policy = build_policy(cfg_of()).eval()
    W = ToyW(); V = ToyV(W.A)
    ctx = ctx_batch()
    xi = torch.rand(B, 3, 6, generator=torch.Generator().manual_seed(3)) * 2 - 1
    return policy, W, V, ctx, xi


def test_modes():
    assert MODES == ("amortized", "bon_pred", "refine", "bon_true")


def test_bon_pred_not_worse_than_amortized():
    policy, W, V, ctx, xi = setup()
    # n=16 random draws are not guaranteed to contain the prior mean, so compare against the best of the draws
    th, s = bon_pred(policy, W, V, ctx, 16, xi, 1.0, torch.Generator().manual_seed(0))
    cand = policy.sample(ctx, 16, torch.Generator().manual_seed(0))
    best = torch.stack([V.score(ctx.text_emb, ctx.slots, W.expected(ctx.robot_tok, ctx.path_tok, cand[:, i], xi), 1.0)
                        for i in range(16)]).max(0).values
    assert torch.allclose(s, best) and th.shape == (B, 14) and (th.abs() < 1).all()


def test_bon_pred_picks_argmax(monkeypatch):
    policy, W, V, ctx, xi = setup()
    cand = torch.rand(B, 5, 14, generator=torch.Generator().manual_seed(9)) * 1.8 - 0.9
    monkeypatch.setattr(policy, "sample", lambda c, n, g: cand)
    th, s = bon_pred(policy, W, V, ctx, 5, xi, 1.0, None)
    sc = torch.stack([V.score(ctx.text_emb, ctx.slots, W.expected(None, None, cand[:, i], xi), 1.0) for i in range(5)], 1)
    assert torch.equal(th, cand[torch.arange(B), sc.argmax(1)]) and torch.allclose(s, sc.max(1).values)


def test_refine_improves_and_stays_in_trust_region():
    policy, W, V, ctx, xi = setup()
    th0, s0 = amortized(policy, W, V, ctx, xi, 1.0)
    th, s = refine(policy, W, V, ctx, 50, 0.05, xi, 1.0, 0.3)
    assert (s >= s0 - 1e-6).all() and ((th - th0).abs() <= 0.3 + 1e-6).all() and (th.abs() < 1).all()
    assert (s > s0 + 1e-4).any()


def test_refine_nan_row_falls_back():
    policy, W, V, ctx, xi = setup()

    class NanW(ToyW):
        def forward(self, robot_tok, path_tok, theta_n, xi_n):
            out = super().forward(robot_tok, path_tok, theta_n, xi_n)
            bad = torch.arange(out.shape[0]) // xi.shape[1] == 1  # every xi draw of context 1
            out = out.clone()
            out[bad] = float("nan")
            return out

    Wn = NanW()
    th, _ = refine(policy, Wn, V, ctx, 10, 0.05, xi, 1.0, 0.3)
    assert torch.isfinite(th).all() and torch.equal(th[1], amortized(policy, Wn, V, ctx, xi, 1.0)[0][1])
    th0 = amortized(policy, W, V, ctx, xi, 1.0)[0]
    assert not torch.equal(th[0], th0[0])  # healthy rows still move


def test_bon_true_calls_rollout_once():
    policy, W, V, ctx, xi = setup()
    calls = []
    f = lambda th: calls.append(th.shape) or W.expected(None, None, th, xi.repeat_interleave(4, 0))[:, None]  # noqa: E731
    th, s = bon_true(policy, V, ctx, 4, f, 1.0, torch.Generator().manual_seed(0))
    assert calls == [(B * 4, 14)] and th.shape == (B, 14) and s.shape == (B,)


def test_deterministic_given_generator():
    policy, W, V, ctx, xi = setup()
    a = bon_pred(policy, W, V, ctx, 8, xi, 1.0, torch.Generator().manual_seed(5))
    b = bon_pred(policy, W, V, ctx, 8, xi, 1.0, torch.Generator().manual_seed(5))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


class DropW(ToyW):
    def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, robot_tok, path_tok, theta_n, xi_n):
        return self.drop(super().forward(robot_tok, path_tok, theta_n, xi_n))


def test_refine_no_param_grads_eval_mode_and_restores_flags():
    policy, _, V, ctx, xi = setup()
    W = DropW(); V = ToyV(W.A); W.train(); V.train()
    runs = [refine(policy, W, V, ctx, 10, 0.05, xi, 1.0, 0.3) for _ in range(2)]
    assert torch.equal(runs[0][0], runs[1][0]) and torch.equal(runs[0][1], runs[1][1])
    assert all(p.grad is None for p in list(W.parameters()) + list(V.parameters()))
    assert W.training and V.training and W.drop.training
    W.eval(); amortized(policy, W, V, ctx, xi, 1.0)
    assert not W.training


def test_pred_score_matches_returned_theta():
    policy, W, V, ctx, xi = setup()
    th, s = refine(policy, W, V, ctx, 20, 0.05, xi, 1.0, 0.3)
    assert torch.allclose(s, V.score(ctx.text_emb, ctx.slots, W.expected(None, None, th, xi), 1.0))

    class NanW(ToyW):
        def forward(self, robot_tok, path_tok, theta_n, xi_n):
            out = super().forward(robot_tok, path_tok, theta_n, xi_n)
            bad = torch.arange(out.shape[0]) // xi.shape[1] == 1  # every xi draw of context 1
            out = out.clone()
            out[bad] = float("nan")
            return out

    th, s = refine(policy, NanW(), V, ctx, 5, 0.05, xi, 1.0, 0.3)
    assert not torch.isinf(s).any() and torch.isnan(s[1]) and torch.isfinite(s[[0, 2]]).all()


def test_bon_true_picks_best_rollout(monkeypatch):
    policy, W, V, ctx, xi = setup()
    cand = torch.rand(B, 4, 14, generator=torch.Generator().manual_seed(9)) * 1.8 - 0.9
    monkeypatch.setattr(policy, "sample", lambda c, n, g: cand)
    pick = torch.tensor([2, 0, 3])
    target = ctx.text_emb @ V.M  # [B,20], phi equal to the target scores best
    phi = (target[:, None] + 5.0).repeat(1, 4, 1)
    phi[torch.arange(B), pick] = target
    th, s = bon_true(policy, V, ctx, 4, lambda t: phi.reshape(B * 4, 1, 20), 1.0, None)
    assert torch.equal(th, cand[torch.arange(B), pick]) and torch.allclose(s, torch.zeros(B), atol=1e-4)


def test_bon_true_scores_each_draw():
    # candidate 0: draws at target -/+ 5 (their mean hits the target); candidate 1: both draws at target + 1.
    # V per draw prefers candidate 1, V of the mean would prefer candidate 0
    policy, W, V, ctx, xi = setup()
    target = ctx.text_emb @ V.M  # [B,20]
    phi = torch.stack([torch.stack([target - 5, target + 5], 1), torch.stack([target + 1, target + 1], 1)], 1)
    th, s = bon_true(policy, V, ctx, 2, lambda t: phi.reshape(B * 2, 2, 20), 1.0, torch.Generator().manual_seed(0))
    cand = policy.sample(ctx, 2, torch.Generator().manual_seed(0))
    assert torch.allclose(th, cand[:, 1], atol=1e-6) and torch.allclose(s, torch.full((B,), -20.0))
