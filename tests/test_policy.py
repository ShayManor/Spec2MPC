import inspect
from dataclasses import replace
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

import spec2mpc.model.train_policy as tp
from spec2mpc.model.policy import ContextEncoder, ContextTable, PolicyContext, Spec2MPCPolicy, build_policy, vw_objective
from spec2mpc.model.train_policy import train_policy

ROOT = Path(__file__).parent.parent
B, D_TEXT, D_SLOT = 3, 8, 10


def ctx_batch(n: int = B, seed: int = 0) -> PolicyContext:
    g = torch.Generator().manual_seed(seed)
    return PolicyContext(torch.randn(n, 6, 24, generator=g), torch.randn(n, D_TEXT, generator=g),
                         torch.randn(n, D_SLOT, generator=g), torch.randn(n, 64, 6, generator=g))


def table_of(ctx: PolicyContext) -> ContextTable:
    n = ctx.batch_size
    return ContextTable(ctx.robot_tok, ctx.path_tok, ctx.text_emb, ctx.slots, torch.arange(n).repeat(2, 1).T.contiguous())


def cfg_of(**over):
    cfg = OmegaConf.create({
        "model": {"d_model": 32, "n_layers": 2, "n_heads": 4, "dropout": 0.0, "d_latent": 4, "hidden": 64},
        "text_mode": "full", "beta": 0.1, "epochs_distill": 0, "steps_vw": 0, "alpha_distill": 1.0,
        "n_samples": 4, "n_xi": 3, "lam": 1.0, "opt": {"lr": 3.0e-3, "batch": 16}, "seed": 0, "device": "cpu",
        "d_text": D_TEXT, "d_slot": D_SLOT})
    cfg.sim = OmegaConf.load(ROOT / "configs/sim/default.yaml")
    return OmegaConf.merge(cfg, over)


class ToyW(nn.Module):
    # fixed random linear map theta -> phi (xi ignored)
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.randn(14, 20, generator=torch.Generator().manual_seed(1)))

    def expected(self, robot_tok, path_tok, theta_n, xi_n):
        # theta_n [B,14], xi_n [B,X,6] -> [B,20]
        assert xi_n.dim() == 3 and xi_n.shape[0] == theta_n.shape[0]
        return theta_n @ self.A

    def forward(self, robot_tok, path_tok, theta_n, xi_n):
        # theta_n [N,14], xi_n [N,6] one draw per row -> [N,20]
        assert xi_n.dim() == 2 and xi_n.shape[0] == theta_n.shape[0]
        return theta_n @ self.A


class ToyV(nn.Module):
    # score = -||phi - target(text)||^2, target reachable inside the box
    def __init__(self, A: torch.Tensor):
        super().__init__()
        P = 0.3 * torch.randn(D_TEXT, 14, generator=torch.Generator().manual_seed(2))
        self.M = nn.Parameter(P @ A.detach())

    def score(self, text_emb, slots, phi, lam):
        return -((phi - text_emb @ self.M) ** 2).sum(-1)


def clone_params(m):
    return [p.detach().clone() for p in m.parameters()]


def params_equal(m, ps):
    return all(torch.equal(p, q) for p, q in zip(m.parameters(), ps))


def test_sample_shapes_and_box():
    torch.manual_seed(0)
    policy = build_policy(cfg_of())
    ctx, gen = ctx_batch(), torch.Generator().manual_seed(0)
    th = policy.sample(ctx, 5, gen)
    assert th.shape == (B, 5, 14) and (th.abs() < 1).all()
    assert policy.amortized(ctx).shape == (B, 14)
    assert torch.isfinite(policy.elbo_loss(ctx, torch.rand(B, 14) * 2 - 1, 0.1))
    assert not torch.allclose(th[:, 0], th[:, 1])  # distinct draws


def test_distill_fits_two_contexts():
    torch.manual_seed(0)
    cfg = cfg_of(epochs_distill=500)
    ctx = ctx_batch(2)
    target = torch.tensor([[0.5] * 14, [-0.4] * 14]) * torch.linspace(-1, 1, 14)
    W = ToyW(); V = ToyV(W.A)
    policy = train_policy(cfg, (table_of(ctx), target), table_of(ctx), W, V)
    policy.eval()
    assert (policy.amortized(ctx) - target).abs().max() < 0.05


def test_vw_improves_and_freezes():
    torch.manual_seed(0)
    cfg = cfg_of(steps_vw=200, alpha_distill=0.1)
    ctx = ctx_batch(8)
    W = ToyW(); V = ToyV(W.A)
    w0, v0 = clone_params(W), clone_params(V)
    xi = torch.rand(8, 3, 6, generator=torch.Generator().manual_seed(3)) * 2 - 1
    torch.manual_seed(0)
    policy = build_policy(cfg)
    s0 = vw_objective(policy, W, V, ctx, xi, 4, 1.0, torch.Generator().manual_seed(4)).mean()
    torch.manual_seed(0)
    distill = (table_of(ctx.index(slice(0, 2))), torch.zeros(2, 14))
    log = []
    policy = train_policy(cfg, distill, table_of(ctx), W, V, log=log)
    policy.eval()
    s1 = vw_objective(policy, W, V, ctx, xi, 4, 1.0, torch.Generator().manual_seed(4)).mean()
    assert s1 > s0 + 0.1, (s0, s1)
    assert params_equal(W, w0) and params_equal(V, v0)
    assert sum(r["stage"] == "vw" for r in log) == 200


def test_text_zero_ignores_text():
    torch.manual_seed(0)
    enc = ContextEncoder(D_TEXT, D_SLOT, 32, 2, 4, 0.0, text_mode="zero")
    ctx = ctx_batch()
    assert torch.equal(enc(ctx), enc(replace(ctx, text_emb=torch.randn_like(ctx.text_emb))))
    full = ContextEncoder(D_TEXT, D_SLOT, 32, 2, 4, 0.0, text_mode="full")
    assert not torch.equal(full(ctx), full(replace(ctx, text_emb=torch.randn_like(ctx.text_emb))))


def test_vw_uses_global_prior_only(monkeypatch):
    assert list(inspect.signature(vw_objective).parameters) == ["policy", "W", "V", "ctx", "xi_n", "n_samples", "lam", "gen"]
    cfg = cfg_of(steps_vw=3)
    seen = []
    real = tp.sample_conditions

    def spy(n, sim_cfg, gen):
        seen.append(OmegaConf.to_container(sim_cfg.conditions))
        return real(n, sim_cfg, gen)

    monkeypatch.setattr(tp, "sample_conditions", spy)
    ctx = ctx_batch(4)
    W = ToyW(); V = ToyV(W.A)
    train_policy(cfg, (table_of(ctx), torch.zeros(4, 14)), table_of(ctx), W, V)
    assert len(seen) == 3 and all(s == OmegaConf.to_container(cfg.sim.conditions) for s in seen)


def test_context_table_index():
    ctx = ctx_batch(3)
    pair = torch.tensor([[0, 2], [2, 1]])
    t = ContextTable(ctx.robot_tok, ctx.path_tok, ctx.text_emb, ctx.slots, pair)
    sub = t.index(torch.tensor([1]))
    assert t.batch_size == 2 and torch.equal(sub.robot_tok[0], ctx.robot_tok[2]) and torch.equal(sub.text_emb[0], ctx.text_emb[1])
    assert torch.equal(sub.path_tok[0], ctx.path_tok[2]) and torch.equal(sub.slots[0], ctx.slots[1])


def test_policy_checkpoint_roundtrip(tmp_path):
    from spec2mpc.model.io import load_checkpoint, save_checkpoint
    torch.manual_seed(0)
    cfg = cfg_of()
    p = build_policy(cfg).eval()
    save_checkpoint(tmp_path / "model.pt", p, cfg, {})
    q, _, _ = load_checkpoint(tmp_path / "model.pt", build_policy)
    ctx = ctx_batch()
    assert torch.equal(p.amortized(ctx), q.eval().amortized(ctx))


def test_distill_table_expands_statements():
    from spec2mpc.model.train_policy import distill_table
    ctx = ctx_batch(2)
    theta = torch.tensor([[0.1] * 14, [0.2] * 14, [0.3] * 14])
    table, th = distill_table(["p0", "p1"], ctx.robot_tok, ctx.path_tok, torch.randn(3, D_TEXT), torch.zeros(3, D_SLOT),
                              ["a", "b", "a"], ["p1", "p0", "p1"], ["a", "b", "b"], theta)
    assert table.pair.tolist() == [[1, 0], [1, 2], [0, 1], [1, 1]]
    assert torch.equal(th[:, 0], theta[[0, 0, 1, 2], 0])


def test_vw_table_first_episode_per_train_block_by_task():
    from spec2mpc.model.data import RolloutTensors
    from spec2mpc.model.train_policy import vw_table
    E = 6
    block = torch.tensor([5, 5, 2, 2, 9, 9])
    task = torch.tensor([0, 0, 1, 1, 0, 0])
    robot = torch.arange(E).float()[:, None, None].expand(E, 6, 24)
    z = torch.zeros(E, dtype=torch.long)
    d = RolloutTensors(robot, torch.zeros(E, 64, 6), torch.zeros(E, 14), torch.zeros(E, 6), torch.zeros(E, 20),
                       task, block, torch.arange(E), z, [])
    train = torch.tensor([True, True, True, True, False, False])
    t = vw_table(d, train, torch.randn(3, D_TEXT), torch.zeros(3, D_SLOT), torch.tensor([0, 1, 0]))
    # blocks sorted by id: 2 (episode 2, goto) then 5 (episode 0, lap); block 9 is val
    assert t.robot_tok[:, 0, 0].tolist() == [2.0, 0.0]
    assert t.pair.tolist() == [[0, 1], [1, 0], [1, 2]]


def test_logstd_clamped():
    torch.manual_seed(0)
    policy = build_policy(cfg_of())
    with torch.no_grad():
        policy.prior_net[-1].bias.fill_(50.0)
        policy.posterior_net[-1].bias.fill_(-50.0)
    z = policy.encoder(ctx_batch())
    assert (policy.prior(z)[1] == 2.0).all() and (policy.posterior(z, torch.zeros(B, 14))[1] == -5.0).all()


def test_check_split_guards_w_and_v():
    import pytest
    from spec2mpc.model.train_policy import check_split
    pc = OmegaConf.create({"data": {"val_frac": 0.1, "split_seed": 0}})
    w = OmegaConf.create({"data": {"val_frac": 0.1, "split_seed": 0}})
    v = OmegaConf.create({"val_frac": 0.1, "split_seed": 0})
    check_split(pc, w, v)
    with pytest.raises(ValueError, match="world model"):
        check_split(pc, OmegaConf.create({"data": {"val_frac": 0.2, "split_seed": 0}}), v)
    with pytest.raises(ValueError, match="critic"):
        check_split(pc, w, OmegaConf.create({"val_frac": 0.1, "split_seed": 3}))


def test_expected_score_is_mean_of_per_draw_scores():
    from spec2mpc.model.policy import expected_score
    from spec2mpc.model.world_model import WorldModel
    torch.manual_seed(0)
    W = WorldModel(16, 1, 2, 0.0).eval()

    class SquareV(nn.Module):  # nonlinear in phi, so the mean of V differs from V of the mean
        def score(self, text_emb, slots, phi, lam):
            return (phi ** 2).sum(-1) + text_emb.sum(-1) - lam * phi.abs().sum(-1)

    V, ctx = SquareV(), ctx_batch()
    theta = torch.rand(B, 14) * 2 - 1
    xi = torch.rand(B, 4, 6, generator=torch.Generator().manual_seed(3)) * 2 - 1
    with torch.no_grad():
        s = expected_score(W, V, ctx.robot_tok, ctx.path_tok, theta, xi, ctx.text_emb, ctx.slots, 0.5)
        per_draw = torch.stack([V.score(ctx.text_emb, ctx.slots, W(ctx.robot_tok, ctx.path_tok, theta, xi[:, x]), 0.5)
                                for x in range(4)]).mean(0)
        of_mean = V.score(ctx.text_emb, ctx.slots, W.expected(ctx.robot_tok, ctx.path_tok, theta, xi), 0.5)
    assert s.shape == (B,) and torch.allclose(s, per_draw, atol=1e-5)
    assert not torch.allclose(s, of_mean, atol=1e-3)
