import torch

from spec2mpc.critic.critic import CriticEnsemble, critic_loss, flip_pair_weight
from spec2mpc.eval.stats import spearman


def test_ensemble_shapes_and_score():
    torch.manual_seed(0)
    v = CriticEnsemble(384, 10, 20, 5, 64, 2)
    t, s, p = torch.randn(8, 384), torch.zeros(8, 10), torch.rand(8, 20)
    out = v(t, s, p)
    assert out.shape == (5, 8)
    assert torch.allclose(v.score(t, s, p, 1.0), out.mean(0) - out.std(0))
    assert torch.allclose(v.member(2, t, s, p), out[2])
    assert not torch.allclose(out[0], out[1])  # members initialized independently


def test_phi_stats_robust_and_finite_on_heavy_tails():
    v = CriticEnsemble(4, 2, 3, 2, 16, 2)
    phi = torch.randn(1000, 3, generator=torch.Generator().manual_seed(0))
    phi[0, 0] = -1310.0
    phi[:, 2] = 5.0  # constant feature
    v.set_phi_stats(phi)
    assert abs(v.phi_center[0].item()) < 0.2 and abs(v.phi_scale[0].item() - 1.0) < 0.15
    assert v.phi_scale[2].item() > 0 and v.phi_center[2].item() == 5.0
    z = v.phi_to_z(phi)
    assert torch.isfinite(z).all() and z.abs().max() < 10
    assert {"phi_center", "phi_scale"} <= set(v.state_dict())


def test_rank_loss_only_within_group():
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    group = torch.tensor([0, 0, 1, 1])
    pred = torch.tensor([[0.2, 0.8, 0.9, 0.1]])  # group 0 ordered right, group 1 reversed
    base = critic_loss(pred, target, group, 1.0, 0.02)
    # flip the cross-group order: shift group 1 targets and preds by the same constant (MSE and within-group pairs unchanged)
    c = torch.tensor([0.0, 0.0, 5.0, 5.0])
    assert torch.allclose(critic_loss(pred + c, target + c, group, 1.0, 0.02), base)
    # the within-group order matters at equal MSE
    fixed = torch.tensor([[0.2, 0.8, -0.9, 1.9]])  # group 1 ordered right, same squared errors
    mse = ((pred - target) ** 2).mean()
    assert torch.allclose(((fixed - target) ** 2).mean(), mse)
    assert critic_loss(fixed, target, group, 1.0, 0.02) < base
    # pairs within the margin are ignored, so the loss reduces to MSE
    assert torch.allclose(critic_loss(pred, target, group, 1.0, 2.0), mse)


def test_rank_loss_is_per_member_sum():
    target, group = torch.tensor([0.0, 1.0, 2.0]), torch.zeros(3, dtype=torch.long)
    pred = torch.tensor([[0.0, 2.0, 1.0], [1.0, 0.0, 2.0]])
    both = critic_loss(pred, target, group, 1.0, 0.02)
    assert torch.allclose(both, critic_loss(pred[:1], target, group, 1.0, 0.02) + critic_loss(pred[1:], target, group, 1.0, 0.02))


def test_learns_synthetic_card_dependent_target():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    n = 2048
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # card A, card B
    card = torch.randint(0, 2, (n,), generator=g)
    phi = torch.randn(n, 4, generator=g) * 3
    target = torch.where(card == 0, phi[:, 0], -phi[:, 0]).tanh()
    v = CriticEnsemble(2, 1, 4, 3, 32, 2)
    v.set_phi_stats(phi)
    opt = torch.optim.Adam(v.parameters(), 3e-3)
    for _ in range(300):
        idx = torch.randint(0, n, (256,), generator=g)
        loss = critic_loss(v(text[card[idx]], torch.zeros(256, 1), phi[idx]), target[idx], card[idx], 1.0, 0.02)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s = v.score(text[card], torch.zeros(n, 1), phi, 1.0)
    for c in (0, 1):
        assert spearman(s[card == c], target[card == c]) > 0.9


def test_checkpoint_roundtrip(tmp_path):
    from omegaconf import OmegaConf

    from spec2mpc.critic.critic import build_critic
    from spec2mpc.model.io import load_checkpoint, save_checkpoint
    cfg = OmegaConf.create({"d_text": 8, "d_slot": 4, "members": 2, "hidden": 16, "n_layers": 2})
    v = build_critic(cfg)
    v.set_phi_stats(torch.randn(100, 20))
    save_checkpoint(tmp_path / "model.pt", v, cfg, {})
    w, _, _ = load_checkpoint(tmp_path / "model.pt", build_critic)
    t, s, p = torch.randn(3, 8), torch.randn(3, 4), torch.randn(3, 20)
    assert torch.equal(v(t, s, p), w(t, s, p))


def test_pair_weight_and_mse_target_are_optional_and_backward_compatible():
    target, group = torch.tensor([0.0, 1.0, 2.0]), torch.zeros(3, dtype=torch.long)
    pred = torch.tensor([[0.0, 2.0, 1.0]])
    base = critic_loss(pred, target, group, 1.0, 0.02)
    assert torch.allclose(critic_loss(pred, target, group, 1.0, 0.02, pair_weight=torch.ones(3, 3)), base)
    # weighting only the misordered pair (2, 1) raises the weighted mean rank loss
    w = torch.ones(3, 3); w[2, 1] = 4.0
    assert critic_loss(pred, target, group, 1.0, 0.02, pair_weight=w) > base
    # mse_target replaces the target in the MSE term only
    mt = torch.asinh(target)
    expect = ((pred - mt) ** 2).mean() + (critic_loss(pred, target, group, 1.0, 0.02) - ((pred - target) ** 2).mean())
    assert torch.allclose(critic_loss(pred, target, group, 1.0, 0.02, mse_target=mt), expect)


def test_flip_pair_weight_marks_pairs_reordered_by_another_card():
    # rows: 3 episodes for card 0; J of cards 0 and 1 per row, card 1 in conditions only for rows 0 and 1
    j_all = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    ok_all = torch.tensor([[True, True], [True, True], [True, False]])
    w = flip_pair_weight(j_all, ok_all, 0.02, 4.0)
    assert w.shape == (3, 3)
    assert w[0, 1] == 4.0 and w[1, 0] == 4.0  # card 1 orders (0, 1) the other way
    assert w[0, 2] == 1.0 and w[2, 1] == 1.0  # card 1 not in conditions for row 2
