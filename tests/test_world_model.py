import inspect

import torch
from omegaconf import OmegaConf

from spec2mpc.model.data import RolloutTensors
from spec2mpc.model.io import load_checkpoint, save_checkpoint
from spec2mpc.model.world_model import WorldModel, build_world_model, wm_loss
from spec2mpc.sim.storage import git_hash


def random_batch(n: int, seed: int = 0) -> RolloutTensors:
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(n, dtype=torch.long)
    return RolloutTensors(torch.randn(n, 6, 24, generator=g), torch.randn(n, 64, 6, generator=g),
                          torch.rand(n, 14, generator=g) * 2 - 1, torch.rand(n, 6, generator=g) * 2 - 1,
                          torch.randn(n, 20, generator=g) * 3 + 1, z, z, z, z, [])


def small_model(seed: int = 0) -> WorldModel:
    torch.manual_seed(seed)
    m = WorldModel(32, 2, 4, 0.0)
    m.set_phi_stats(random_batch(256, seed=1).phi)
    return m


def test_overfits_tiny_batch():
    m = small_model(); data = random_batch(64)
    l0 = wm_loss(m, data).item()
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(300):
        opt.zero_grad(); wm_loss(m, data).backward(); opt.step()
    assert wm_loss(m, data).item() < 0.1 * l0


def test_theta_gradient_finite_nonzero():
    m = small_model(); b = random_batch(4)
    th = torch.zeros(4, 14, requires_grad=True)
    m(b.robot_tok, b.path_tok, th, b.xi_n).sum().backward()
    assert torch.isfinite(th.grad).all() and th.grad.abs().sum() > 0


def test_output_shape_and_no_text_arguments():
    m = small_model(); b = random_batch(5)
    assert m(b.robot_tok, b.path_tok, b.theta_n, b.xi_n).shape == (5, 20)
    for fn in (m.forward, m.expected):
        assert list(inspect.signature(fn).parameters) == ["robot_tok", "path_tok", "theta_n", "xi_n"]


def test_expected_equals_mean_over_xi():
    m = small_model().eval(); b = random_batch(3)
    xi = torch.rand(3, 5, 6) * 2 - 1
    e = m.expected(b.robot_tok, b.path_tok, b.theta_n, xi)
    ref = torch.stack([m(b.robot_tok, b.path_tok, b.theta_n, xi[:, k]) for k in range(5)], dim=1).mean(1)
    assert e.shape == (3, 20) and torch.allclose(e, ref, atol=1e-5)


def test_batch_order_equivariant():
    m = small_model().eval(); b = random_batch(8)
    perm = torch.randperm(8, generator=torch.Generator().manual_seed(3))
    y = m(b.robot_tok, b.path_tok, b.theta_n, b.xi_n)
    yp = m(b.robot_tok[perm], b.path_tok[perm], b.theta_n[perm], b.xi_n[perm])
    assert torch.allclose(yp, y[perm], atol=1e-5)


def test_phi_stats_are_robust():
    m = small_model()
    phi = random_batch(256, seed=1).phi
    q = phi.quantile(torch.tensor([0.25, 0.5, 0.75]), dim=0)
    assert torch.allclose(m.phi_center, q[1]) and torch.allclose(m.phi_scale, ((q[2] - q[0]) / 1.349).clamp_min(1e-3))
    # a zero head predicts z = 0, i.e. the train median in raw units
    m.head.weight.data.zero_(); m.head.bias.data.zero_()
    b = random_batch(16)
    assert torch.allclose(m(b.robot_tok, b.path_tok, b.theta_n, b.xi_n), m.phi_center.expand(16, 20))


def test_transform_roundtrip_with_outliers():
    m = small_model()
    phi = random_batch(64, seed=4).phi.double()
    phi[0, 3], phi[1, 7], phi[2, 13] = 1e4, -1e4, 7333.0
    back = m.z_to_phi(m.phi_to_z(phi))
    assert torch.allclose(back, phi, rtol=1e-9, atol=1e-9)


def test_constant_feature_scale_floor():
    m = WorldModel(32, 2, 4, 0.0)
    phi = random_batch(64).phi; phi[:, 5] = 2.0
    m.set_phi_stats(phi)
    assert m.phi_scale[5] == 1e-3 and torch.isfinite(m.phi_to_z(phi)).all()


def test_outlier_target_loss_bounded():
    m = small_model(); b = random_batch(16)
    base = wm_loss(m, b).item()
    b.phi[0, 3] = 1e4
    out = wm_loss(m, b).item()
    # asinh compresses 1e4 to z of about 10, so one outlier adds at most ~10 / (16 * 20) per Huber element
    assert out < base + 1.0


def test_theta_gradient_finite_nonzero_after_training():
    m = small_model(); data = random_batch(64)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(50):
        opt.zero_grad(); wm_loss(m, data).backward(); opt.step()
    th = torch.zeros(4, 14, requires_grad=True)
    m(data.robot_tok[:4], data.path_tok[:4], th, data.xi_n[:4]).sum().backward()
    assert torch.isfinite(th.grad).all() and th.grad.abs().sum() > 0


def test_checkpoint_roundtrip(tmp_path):
    cfg = OmegaConf.create({"model": {"d_model": 32, "n_layers": 2, "n_heads": 4, "dropout": 0.0}, "seed": 0})
    m = build_world_model(cfg); m.set_phi_stats(random_batch(64, seed=2).phi); m.eval()
    b = random_batch(6)
    save_checkpoint(tmp_path / "model.pt", m, cfg, {"epochs": 1})
    m2, cfg2, extra = load_checkpoint(tmp_path / "model.pt", build_world_model)
    m2.eval()
    assert torch.equal(m(b.robot_tok, b.path_tok, b.theta_n, b.xi_n), m2(b.robot_tok, b.path_tok, b.theta_n, b.xi_n))
    assert cfg2.model.d_model == 32 and extra["epochs"] == 1 and extra["git_hash"] == git_hash()
