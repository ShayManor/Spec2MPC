import pytest
import torch

from spec2mpc.baselines.applv_lang import APPLVRegressor, bc_loss, reinforce_step
from spec2mpc.model.policy import ContextEncoder, PolicyContext

D_TEXT, D_SLOT = 8, 10


def ctx_batch(n: int, seed: int = 0) -> PolicyContext:
    g = torch.Generator().manual_seed(seed)
    return PolicyContext(torch.randn(n, 6, 24, generator=g), torch.randn(n, D_TEXT, generator=g),
                         torch.randn(n, D_SLOT, generator=g), torch.randn(n, 64, 6, generator=g))


def make_model(seed: int = 0) -> APPLVRegressor:
    torch.manual_seed(seed)
    return APPLVRegressor(ContextEncoder(D_TEXT, D_SLOT, 32, 2, 4, 0.0, "full"), 64)


def test_forward_shape_and_box():
    out = make_model()(ctx_batch(3))
    assert out.shape == (3, 14) and out.abs().max() < 1


def test_bc_fits():
    model, ctx = make_model(), ctx_batch(2)
    theta = torch.tensor([[0.5] * 14, [-0.3] * 14])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss = bc_loss(model, ctx, theta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert bc_loss(model, ctx, theta).item() < 1e-3


def test_reinforce_moves_toward_reward():
    model, ctx = make_model(), ctx_batch(2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    target = torch.full((14,), 0.5)
    rf = lambda th: -((th - target) ** 2).sum(-1)  # noqa: E731
    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        reinforce_step(model, ctx, rf, 0.2, 16, opt, gen)
    assert (model(ctx) - target).abs().max() < 0.15


def test_reward_fn_called_once_per_step():
    model, ctx = make_model(), ctx_batch(3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    shapes = []

    def rf(th):
        shapes.append(tuple(th.shape))
        return -th.pow(2).sum(-1)

    out = reinforce_step(model, ctx, rf, 0.15, 5, opt)
    assert shapes == [(3, 5, 14)]
    assert set(out) >= {"reward", "loss"} and all(isinstance(v, float) for v in out.values())


def test_zero_advantage_fraction():
    model, ctx = make_model(), ctx_batch(2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rf = lambda th: torch.stack([th[0].sum(-1), torch.zeros(th.shape[1])])  # noqa: E731  row 1 constant
    out = reinforce_step(model, ctx, rf, 0.2, 4, opt, torch.Generator().manual_seed(0))
    assert out["zero_adv_frac"] == 0.5


class FixedMu(torch.nn.Module):
    # model(ctx) = the same learnable mu [14] for every row
    def __init__(self, value: float):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.full((14,), value))

    def forward(self, ctx):
        return self.mu.expand(ctx.batch_size, -1)


def test_logprob_on_unclamped_sample_at_box_edge():
    # mu at 0.98, reward prefers 1.0: the score term must be eps / sigma (unclamped sample), not (clamp - mu) / sigma^2
    B, n, sigma = 2, 16, 0.2
    model, ctx = FixedMu(0.98), ctx_batch(B)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    seen = []

    def rf(th):
        seen.append(th.clone())
        return -((th - 1.0) ** 2).sum(-1)

    reinforce_step(model, ctx, rf, sigma, n, opt, torch.Generator().manual_seed(4))
    eps = torch.randn((B, n, 14), generator=torch.Generator().manual_seed(4))
    assert torch.equal(seen[0], (0.98 + sigma * eps).clamp(-1, 1)) and (seen[0] == 1).any()
    r = rf(seen[0])
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-8)
    want = -(adv[..., None] * eps / sigma).sum((0, 1)) / (B * n)
    assert torch.allclose(model.mu.grad, want, atol=1e-5)
    opt, gen = torch.optim.Adam(model.parameters(), lr=1e-2), torch.Generator().manual_seed(5)
    for _ in range(100):
        reinforce_step(model, ctx, rf, sigma, n, opt, gen)
    assert model.mu.mean() > 0.99  # the mean moves toward the preferred edge


def test_samples_stay_in_box():
    model, ctx = make_model(), ctx_batch(4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    seen = []

    def rf(th):
        seen.append(th.detach().clone())
        return th.sum(-1)

    reinforce_step(model, ctx, rf, 5.0, 32, opt, torch.Generator().manual_seed(1))
    th = seen[0]
    assert th.abs().max() <= 1 and (th.abs() == 1).any() and not th.requires_grad


def test_reinforce_deterministic_with_generator():
    outs = []
    for _ in range(2):
        model, ctx = make_model(), ctx_batch(2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        outs.append(reinforce_step(model, ctx, lambda th: -th.pow(2).sum(-1), 0.2, 4, opt,
                                   torch.Generator().manual_seed(3)))
    assert outs[0] == outs[1]


TINY_SIM = ["sim.episode.timeout_cap_lap_s=0.3", "sim.episode.timeout_cap_goto_s=0.3", "sim.grid.res_m=0.5"]


def write_synthetic_oracle(root, out):
    # tiny oracle run dir: config.yaml (1 lap + 1 goto problem) and oracle.parquet with random theta in the box
    import pyarrow as pa
    import pyarrow.parquet as pq
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from spec2mpc.baselines.oracle_bo import build_job
    from spec2mpc.mpc.theta import THETA_NAMES, denormalize_theta

    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="oracle", overrides=TINY_SIM + ["runtime.solver_iters=2", "problems.n_per_task.lap=1",
                                                                  "problems.n_per_task.goto=1"])
    out.mkdir(parents=True)
    OmegaConf.save(cfg, out / "config.yaml")
    job = build_job(cfg)
    th = denormalize_theta(torch.rand(len(job.pairs), 14, generator=torch.Generator().manual_seed(0)) * 2 - 1, cfg.mpc)
    cols = {"problem_id": [job.problems.problem_id[p] for p, _ in job.pairs],
            "card_id": [job.cards[c].id for _, c in job.pairs], "task": [job.problems.task[p] for p, _ in job.pairs]}
    cols.update({n: th[:, j].tolist() for j, n in enumerate(THETA_NAMES)})
    pq.write_table(pa.table(cols), out / "oracle.parquet")
    return cfg


@pytest.mark.slow
def test_script_bc_rl_resume(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    write_synthetic_oracle(root, tmp_path / "oracle")
    base = [sys.executable, "scripts/train_applv.py", "device=cpu", f"oracle={tmp_path}/oracle/oracle.parquet",
            f"out_dir={tmp_path}/applv", "text.encoder.backend=hashed", "model.d_model=32", "model.n_layers=1",
            "hidden=32", "bc.epochs=2", "rl.rows_per_iter=2", "rl.n_samples=2", "rl.xi_per_sample=1"]
    env = dict(os.environ, RESUME="0")
    run = lambda *a, **kw: subprocess.run(base + list(a), cwd=root, capture_output=True, text=True, **kw)  # noqa: E731
    r = run("stage=bc", env=env)
    assert r.returncode == 0, r.stderr
    r = run("stage=rl", "rl.iterations=1", env=env)
    assert r.returncode == 0, r.stderr
    r = run("stage=rl", "rl.iterations=2", env=env)
    assert r.returncode != 0 and "RESUME=1" in r.stderr
    r = run("stage=rl", "rl.iterations=2", env=dict(os.environ, RESUME="1"))
    assert r.returncode == 0, r.stderr
    hist = json.loads((tmp_path / "applv/seed0/metrics.json").read_text())["history"]
    assert [h["iteration"] for h in hist] == [1, 2] and all(h["episodes"] == 4 for h in hist)
    assert (tmp_path / "applv/seed0/model.pt").exists() and (tmp_path / "applv/seed0/bc.pt").exists()
    other = tmp_path / "other_bc.pt"
    other.write_bytes((tmp_path / "applv/seed0/bc.pt").read_bytes())
    r = run("stage=rl", "rl.iterations=3", f"bc_checkpoint={other}", env=dict(os.environ, RESUME="1"))
    assert r.returncode != 0 and "was made with rl settings" in r.stderr  # a resume keeps its bc start


def load_script():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("train_applv", Path(__file__).resolve().parents[1] / "scripts/train_applv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rl_rollouts_use_oracle_run_config(tmp_path, monkeypatch):
    # synthetic oracle dir with 0.3 s episode caps (default 300 / 120 s): the caps and its mpc reach rollout_metrics
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from spec2mpc.rubric.cards import load_cards
    from spec2mpc.sim.problems import load_oracle_problems

    root = Path(__file__).resolve().parents[1]
    write_synthetic_oracle(root, tmp_path / "oracle")
    problems, ocfg = load_oracle_problems(tmp_path / "oracle")
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train/applv", overrides=["device=cpu", "rl.xi_per_sample=1", "mpc.solver.compile=false"])
    mod = load_script()
    seen = []
    real = mod.rollout_metrics

    def spy(robot, scene, theta, cond, rcfg, gen, device):
        seen.append((float(rcfg.sim.episode.timeout_cap_lap_s), float(rcfg.sim.episode.timeout_cap_goto_s),
                     bool(rcfg.mpc.solver.compile), int(robot.runtime.solver_iters)))
        return real(robot, scene, theta, cond, rcfg, gen, device)

    monkeypatch.setattr(mod, "rollout_metrics", spy)
    card = next(c for c in load_cards(root / "data/cards", "train") if c.task == problems.task[0])
    rc = mod.rollout_cfg(cfg, ocfg)
    fn = mod.card_reward_fn(problems, [0], [card], rc, int(cfg.rl.xi_per_sample), torch.Generator().manual_seed(0),
                            torch.Generator().manual_seed(1))
    r = fn(torch.zeros(1, 2, 14))
    assert r.shape == (1, 2) and torch.isfinite(r).all()
    assert seen == [(0.3, 0.3, False, 2)]
