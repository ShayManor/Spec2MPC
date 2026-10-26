import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import spec2mpc.eval.metrics as metrics_mod
from spec2mpc.eval.metrics import (attach_jhat, critic_gap, critic_target_asinh, drop_unanchored, floor_violation_rate,
                                   normalized_score, summarize)
from spec2mpc.eval.plots import plot_critic_gap, plot_methods, plot_score_vs_budget
import spec2mpc.eval.runner as runner_mod
from spec2mpc.eval.runner import (EvalContext, EvalUnit, evaluate_setting, make_layout, result_files, result_table,
                                    run_settings, run_thetas, settings, val_units)
from spec2mpc.mpc.theta import D_THETA, THETA_NAMES, theta_bounds
from spec2mpc.robots.kart import RuntimeConfig, sample_karts
from spec2mpc.rubric.cards import load_cards
from spec2mpc.rubric.scoring import card_max_score
from spec2mpc.sim.problems import sample_problems

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
TINY = ["sim.episode.timeout_cap_lap_s=0.3", "sim.episode.timeout_cap_goto_s=0.3", "sim.grid.res_m=0.5",
        "runtime.solver_iters=2"]


@pytest.fixture(scope="module")
def tiny():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name="oracle", overrides=TINY)


@pytest.fixture(scope="module")
def units(tiny):
    return val_units(tiny.splits)


@pytest.fixture(scope="module")
def cards(tiny):
    root = Path(tiny.splits.cards_root)
    return {c.id: c for c in load_cards(root, "train") + load_cards(root, "val")}


@pytest.fixture(scope="module")
def problems(tiny):
    rt = RuntimeConfig(control_rate_hz=20.0, solver_iters=2, perception_rate_hz=10.0)
    return sample_problems({"lap": 2, "goto": 1}, tiny, 3, rt, "val")


def test_val_units_14(units):
    assert len(units) == 14 and all(isinstance(u, EvalUnit) for u in units)
    assert sum(u.kind == "paraphrase" for u in units) == 8 and sum(u.kind == "intent" for u in units) == 6


def test_normalized_score_excludes_small_gap():
    jh, ok = normalized_score(torch.tensor([1., 1.]), torch.tensor([0., 1.]), torch.tensor([2., 1.01]), 0.05)
    assert ok.tolist() == [True, False] and jh[0] == 0.5 and torch.isnan(jh[1])


def test_layout_identical_across_calls_and_task_matched(units, problems, cards, tiny):
    a, b = make_layout(units, problems, cards, 16, 99, tiny.sim), make_layout(units, problems, cards, 16, 99, tiny.sim)
    assert torch.equal(a.cond.friction_scale, b.cond.friction_scale) and torch.equal(a.cond.seed, b.cond.seed)
    assert a.gen_seed == b.gen_seed
    assert all(units[u].task == problems.task[p] for u, p in zip(a.unit_idx.tolist(), a.problem_idx.tolist()))
    n_match = sum(u.task == t for u in units for t in problems.task)
    assert a.unit_idx.numel() == 16 * n_match
    # unit-major, then problem, then xi
    key = (a.unit_idx * problems.n + a.problem_idx).tolist()
    assert key == sorted(key)


def test_paraphrases_of_one_card_share_xi(units, problems, cards, tiny):
    lay = make_layout(units, problems, cards, 4, 99, tiny.sim)
    same = [i for i, u in enumerate(units) if u.card_id == "kart_camera_smooth_001"]
    assert len(same) == 3
    rows = lambda u, p: ((lay.unit_idx == u) & (lay.problem_idx == p)).nonzero().squeeze(1)  # noqa: E731
    for p in range(problems.n):
        if problems.task[p] != "lap":
            continue
        ref = rows(same[0], p)
        for u in same[1:]:
            assert torch.equal(lay.cond.friction_scale[rows(u, p)], lay.cond.friction_scale[ref])
            assert torch.equal(lay.cond.seed[rows(u, p)], lay.cond.seed[ref])
    other = next(i for i, u in enumerate(units) if u.task == "lap" and u.card_id != "kart_camera_smooth_001")
    assert not torch.equal(lay.cond.seed[rows(other, 0)], lay.cond.seed[rows(same[0], 0)])
    # card conditions are honored
    r = rows(same[0], 0)
    assert (lay.cond.friction_scale[r] >= 0.8).all()


def test_same_theta_same_J(units, problems, cards, tiny):
    sub = [next(u for u in units if u.task == "lap"), next(u for u in units if u.task == "goto")]
    lay = make_layout(sub, problems, cards, 2, 99, tiny.sim)
    th = torch.rand(len(sub), problems.n, D_THETA, generator=torch.Generator().manual_seed(0)) * 2 - 1
    a = run_thetas(th, lay, problems, cards, tiny, CPU)
    b = run_thetas(th, lay, problems, cards, tiny, CPU)
    n_pairs = 3
    assert a["J"].shape == (n_pairs,) and torch.equal(a["J"], b["J"])
    assert a["tier_rate"].shape[0] == n_pairs
    assert a["unit_idx"].tolist() == [0, 0, 1] and a["problem_idx"].tolist() == [0, 1, 2]
    n_t = [len(cards[sub[u].card_id].rubric.tiers) for u in a["unit_idx"].tolist()]
    for i, n in enumerate(n_t):  # NaN-padded past the card's tiers
        assert torch.isfinite(a["tier_rate"][i, :n]).all() and torch.isnan(a["tier_rate"][i, n:]).all()
    assert ((a["boundary_violation_rate"] >= 0) & (a["boundary_violation_rate"] <= 1)).all()
    tab = result_table("amortized", 0, 1, sub, problems, cards, a, torch.zeros(n_pairs), torch.zeros(n_pairs),
                       torch.ones(n_pairs), torch.ones(n_pairs), torch.zeros(n_pairs, dtype=torch.bool))
    d = tab.to_pydict()
    assert d["problem_id"] == [problems.problem_id[p] for p in (0, 1, 2)] and d["kind"][0] == sub[0].kind
    assert d["J_norm"][0] == pytest.approx(d["J"][0] / card_max_score(cards[sub[0].card_id]))


def test_floor_violation_rate_zero_for_extremes(tiny):
    params8 = sample_karts(8, tiny.robots, RuntimeConfig(20.0, 2, 10.0), torch.Generator().manual_seed(0)).params
    th = torch.cat([torch.full((4, 14), -1.0), torch.full((4, 14), 1.0)])
    assert floor_violation_rate(th, params8, tiny.mpc) == 0.0


def test_floor_violation_rate_detects_unclipped_margin(tiny, monkeypatch):
    params4 = sample_karts(4, tiny.robots, RuntimeConfig(20.0, 2, 10.0), torch.Generator().manual_seed(0)).params
    real = metrics_mod.adapt

    def leaky(theta, p, mpc_cfg):  # an adapter that lost its margin clip on the first row
        w = real(theta, p, mpc_cfg)
        w.margin_m[0] = 0.0
        return w
    monkeypatch.setattr(metrics_mod, "adapt", leaky)
    assert floor_violation_rate(torch.zeros(4, 14), params4, tiny.mpc) == 0.25


def test_critic_gap_space():
    J_norm = torch.tensor([0.0, 2.0])
    assert torch.allclose(critic_gap(torch.tensor([0.0, 1.0]), J_norm, True), torch.tensor([0.0, 1.0 - math.asinh(2.0)]))
    assert torch.allclose(critic_gap(torch.tensor([0.0, 1.0]), J_norm, False), torch.tensor([0.0, -1.0]))
    assert critic_target_asinh(OmegaConf.create({"target_asinh": True})) is True
    with pytest.raises(KeyError):
        critic_target_asinh(OmegaConf.create({"lam": 1.0}))


def rows(method, K, seed, card, kind, J, problem="p0", unit=0, pred=float("nan"), gap=float("nan")):
    return {"method": method, "K": K, "seed": seed, "unit": unit, "card_id": card, "kind": kind, "problem_id": problem,
            "J": J, "J_norm": J, "Jhat": float("nan"), "valid": False, "pred_score": pred, "critic_gap": gap,
            "pred_mean": pred + 1, "critic_gap_mean": gap + 1,
            "boundary_violation_rate": 0.0, "completion_rate": 1.0, "floor_violation": False,
            "tier_rate_0": 1.0, "tier_rate_1": float("nan")}


def table(rs):
    return pa.Table.from_pylist(rs)


def test_attach_jhat_joins_anchors():
    rs = [rows("default", 0, 0, "a", "intent", 0.0), rows("oracle", 0, 0, "a", "intent", 2.0),
          rows("amortized", 0, 1, "a", "intent", 1.0), rows("amortized", 0, 2, "a", "intent", 3.0)]
    d = attach_jhat(table(rs), 0.05).to_pydict()
    assert d["Jhat"] == [0.0, 1.0, 0.5, 1.5] and all(d["valid"])
    with pytest.raises(ValueError):
        attach_jhat(table(rs[2:]), 0.05)


def test_jhat_pairs_anchors_within_group_only():
    g = lambda r, grp: {**r, "anchor_group": grp}  # noqa: E731
    rs = [g(rows("default", 0, 0, "a", "intent", 0.0), "a"), g(rows("oracle", 0, 0, "a", "intent", 2.0), "a"),
          g(rows("default", 0, 0, "a", "intent", 1.0), "b"), g(rows("oracle", 0, 0, "a", "intent", 3.0), "b"),
          g(rows("amortized", 0, 0, "a", "intent", 2.0), "a"), g(rows("bon_pred", 4, 0, "a", "intent", 2.0), "b"),
          g(rows("refine", 10, 0, "a", "intent", 2.0), "c")]
    kept, dropped = drop_unanchored(table(rs))
    assert dropped == {("refine", 10, "intent", "c"): 1} and kept.num_rows == 6
    d = attach_jhat(kept, 0.05).to_pydict()
    assert d["Jhat"] == [0.0, 1.0, 0.0, 1.0, 1.0, 0.5]  # amortized on a's anchors, bon_pred on b's
    s = summarize(attach_jhat(kept, 0.05), 10, 0, dropped).to_pylist()
    assert all(r["n_unanchored"] == 0 for r in s)


def test_result_files_match_eval_config(tmp_path):
    import json
    for name, n_xi, seed in (("default_0_0_a", 16, 99), ("oracle_0_0_a", 8, 99), ("amortized_0_0_a", 16, 7)):
        pq.write_table(table([rows("default", 0, 0, "a", "intent", 0.0)]), tmp_path / f"{name}.parquet")
        (tmp_path / f"{name}.json").write_text(json.dumps({"config": {"n_xi": n_xi, "xi_seed": seed}}))
    pq.write_table(table([rows("default", 0, 0, "a", "intent", 0.0)]), tmp_path / "nojson_0_0_a.parquet")
    pq.write_table(table([rows("default", 0, 0, "a", "intent", 0.0)]), tmp_path / ".tmp_x_0_0_a.parquet")
    pq.write_table(table([rows("default", 0, 0, "a", "intent", 0.0)]), tmp_path / "summary.parquet")
    files, skipped = result_files(tmp_path, 16, 99)
    assert [p.name for p in files] == ["default_0_0_a.parquet"]
    assert sorted(p.name for p in skipped) == ["amortized_0_0_a.parquet", "nojson_0_0_a.parquet", "oracle_0_0_a.parquet"]


def test_summary_ci_over_cards():
    # card a: 20 rows at Jhat 0 (two seeds pooled), card b: 2 rows at Jhat 1; cards weigh equally (mean 0.5, not
    # 2/22) and the CI resamples cards, so it spans [0, 1]
    rs = []
    for i in range(10):
        for card, j_o in (("a", 2.0), ("b", 2.0)):
            p = f"p{i}"
            rs += [rows("default", 0, 0, card, "intent", 0.0, p), rows("oracle", 0, 0, card, "intent", j_o, p)]
    for seed in (0, 1):
        rs += [rows("amortized", 0, seed, "a", "intent", 0.0, f"p{i}", pred=0.1, gap=0.1) for i in range(10)]
    rs += [rows("amortized", 0, 0, "b", "intent", 2.0, "p0", pred=0.3, gap=0.3)]
    rs += [rows("amortized", 0, 1, "b", "intent", 2.0, "p1", pred=0.3, gap=0.3)]
    s = summarize(attach_jhat(table(rs), 0.05), 1000, 0).to_pylist()
    am = next(r for r in s if r["method"] == "amortized" and r["kind"] == "intent")
    assert am["Jhat_mean"] == pytest.approx(0.5)
    assert am["Jhat_lo"] == pytest.approx(0.0) and am["Jhat_hi"] == pytest.approx(1.0)
    assert am["n_excluded"] == 0 and am["n_cards"] == 2 and am["n_seeds"] == 2
    assert am["gap_mean"] == pytest.approx(0.2) and am["gap_ens_mean"] == pytest.approx(1.2)
    assert am["gap_ens_lo"] == pytest.approx(1.1) and am["gap_ens_hi"] == pytest.approx(1.3)
    assert am["tier_0"] == 1.0 and math.isnan(am["tier_1"]) and am["floor_violation_rate"] == 0.0
    de = next(r for r in s if r["method"] == "default")
    assert de["Jhat_mean"] == 0.0 and math.isnan(de["gap_mean"])


def test_plots_render(tmp_path):
    rs = []
    for kind in ("paraphrase", "intent"):
        for m, ks in (("default", [0]), ("oracle", [0]), ("amortized", [0]), ("bon_pred", [4, 16]), ("refine", [10, 50]),
                      ("bon_true", [4, 16]), ("oracle_bo", [12, 48]), ("nn_retrieval", [0])):
            for k in ks:
                g = float("nan") if m in ("default", "oracle", "oracle_bo", "nn_retrieval") else 0.1
                rs.append({"method": m, "K": k, "kind": kind, "Jhat_mean": 0.5, "Jhat_lo": 0.3, "Jhat_hi": 0.7,
                           "gap_mean": g, "gap_lo": g - 0.05, "gap_hi": g + 0.05,
                           "gap_ens_mean": g + 0.1, "gap_ens_lo": g + 0.05, "gap_ens_hi": g + 0.15})
    s = pa.Table.from_pylist(rs)
    for f in (plot_score_vs_budget, plot_critic_gap, plot_methods):
        out = tmp_path / f"{f.__name__}.png"
        f(s, out)
        assert out.stat().st_size > 0


def _oracle_files(d: Path, problem_ids, tasks, cards, ks, gen) -> None:
    # synthetic oracle.parquet / at_k.parquet over every (problem, same-task card), theta inside the box
    lo, hi = theta_bounds(OmegaConf.load(ROOT / "configs/mpc/kart.yaml"))
    pairs = [(p, c.id, t) for p, t in zip(problem_ids, tasks) for c in cards if c.task == t]
    th = lo + (hi - lo) * torch.rand(len(pairs) * (1 + len(ks)), D_THETA, generator=gen)
    base = {"problem_id": [p for p, _, _ in pairs], "card_id": [c for _, c, _ in pairs]}
    o = {**base, "task": [t for _, _, t in pairs], **{n: th[:len(pairs), j].tolist() for j, n in enumerate(THETA_NAMES)},
         "J_oracle": [0.0] * len(pairs), "J_default": [0.0] * len(pairs), "J_mean": [0.0] * len(pairs),
         "J_best": [0.0] * len(pairs), "n_evals": [1] * len(pairs)}
    pq.write_table(pa.table(o), d / "oracle.parquet")
    a = {"problem_id": base["problem_id"] * len(ks), "card_id": base["card_id"] * len(ks),
         "k": [k for k in ks for _ in pairs], **{n: th[len(pairs):, j].tolist() for j, n in enumerate(THETA_NAMES)}}
    pq.write_table(pa.table(a), d / "at_k.parquet")


@pytest.mark.slow
def test_evaluate_setting_every_method(tmp_path, tiny, cards):
    # wiring of every method on synthetic oracle outputs and small untrained models (1 problem per task, 1 xi)
    from spec2mpc.critic.critic import build_critic
    from spec2mpc.model.policy import build_policy
    from spec2mpc.model.world_model import build_world_model
    from spec2mpc.rubric.splits import statement_records
    from spec2mpc.sim.problems import load_oracle_problems, problem_digest
    from spec2mpc.text.encoder import make_text_encoder, text_features
    from spec2mpc.tokenize.robot_tokens import robot_tokens
    from spec2mpc.text.slots import D_SLOT
    g = torch.Generator().manual_seed(0)
    root = tmp_path / "oracle"
    for split in ("train", "val"):
        (root / split).mkdir(parents=True)
        rc = OmegaConf.merge(tiny, {"split": split, "problems": {"n_per_task": {"lap": 1, "goto": 1}}})
        OmegaConf.save(rc, root / split / "config.yaml")
        pr, _ = load_oracle_problems(root / split)
        torch.save({"problem_id": pr.problem_id, "digest": problem_digest(pr)}, root / split / "problems.pt")
        split_cards = [c for c in cards.values() if split == "val" or not c.id.startswith(("kart_camera", "kart_warehouse"))]
        _oracle_files(root / split, pr.problem_id, pr.task, split_cards, [2], g)
    problems, rcfg = load_oracle_problems(root / "val")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="eval", overrides=[
            f"oracle_root={root}", "n_xi=1", "bon_true_xi=1", "n_xi_model=2", "seeds=[0]", "device=cpu",
            "budgets={bon_pred: [2], refine: [2], bon_true: [2], oracle_bo: [2]}", "text.encoder.backend=hashed",
            "text.encoder.hashed.dim=16"])
    units = val_units(cfg.splits)
    layout = make_layout(units, problems, cards, int(cfg.n_xi), int(cfg.xi_seed), rcfg.sim)
    train = [r for r in statement_records(cfg.splits) if r.role == "train"]
    enc = make_text_encoder(cfg.text)
    emb, slots = text_features(enc, [u.statement for u in units] + [r.text for r in train], cfg.text.slots)
    small = {"d_model": 16, "n_layers": 1, "n_heads": 2, "dropout": 0.0}
    pc = OmegaConf.create({"model": {**small, "d_latent": 2, "hidden": 16}, "text_mode": "full", "d_text": 16,
                           "d_slot": D_SLOT, "mpc": rcfg.mpc})
    vc = OmegaConf.create({"d_text": 16, "d_slot": D_SLOT, "members": 2, "hidden": 8, "n_layers": 1, "target_asinh": True})
    torch.manual_seed(0)
    models = {"policy_full": (build_policy(pc), pc), "policy_zero": (build_policy(OmegaConf.merge(pc, {"text_mode": "zero"})), pc),
              "world_model": (build_world_model(OmegaConf.create({"model": small})), OmegaConf.create({})),
              "critic": (build_critic(vc), vc)}
    from spec2mpc.baselines.applv_lang import build_applv
    models["applv"] = (build_applv(OmegaConf.create({"model": small, "hidden": 8, "d_text": 16, "d_slot": D_SLOT})), pc)
    loaders = {n: (lambda s, n=n: (models[n][0], models[n][1], {"encoder": enc.name})) for n in models}
    ctx = EvalContext(units=units, problems=problems, layout=layout, cards=cards, rcfg=rcfg, cfg=cfg, device=CPU,
                      unit_emb=emb[:14], unit_slots=slots[:14], train_emb=emb[14:], train_card=[r.card_id for r in train],
                      encoder_name=enc.name, robot_tok=robot_tokens(problems.robot, cfg.robot_tokens), loaders=loaders)
    tabs = [evaluate_setting(ctx, m, k, s) for m, k, s in settings(cfg)]
    assert [(t.column("method")[0].as_py(), t.column("K")[0].as_py()) for t in tabs][:2] == [("default", 0), ("oracle", 0)]
    assert len(tabs) == 11 and all(t.num_rows == 14 for t in tabs)
    by = {t.column("method")[0].as_py(): t.to_pydict() for t in tabs}
    assert all(math.isnan(x) for x in by["nn_retrieval"]["critic_gap"])
    for m in ("amortized", "bon_pred", "refine", "bon_true"):
        d = by[m]
        assert all(math.isfinite(x) for x in d["critic_gap"] + d["critic_gap_mean"] + d["pred_mean"])
        # ensemble mean >= mean - lam * std of the same prediction (lam = 1)
        assert all(e >= p - 1e-5 for e, p in zip(d["pred_mean"], d["pred_score"]))
    assert all(math.isnan(x) for x in by["slot_only"]["pred_mean"])
    s = summarize(attach_jhat(pa.concat_tables(tabs), 0.05), 50, 0)
    assert set(s.column("floor_violation_rate").to_pylist()) == {0.0}
    # with ctx.problems J under the oracle theta equals the oracle anchor: Jhat 1 where valid
    o = attach_jhat(pa.concat_tables(tabs[:2]), 0.05).to_pydict()
    assert all(j == 1.0 for j, v, m in zip(o["Jhat"], o["valid"], o["method"]) if v and m == "oracle")


def test_run_settings_isolates_a_failing_setting(tmp_path, monkeypatch, capsys):
    def fake(ctx, method, K, seed):
        if method == "bon_pred":
            raise RuntimeError("boom")
        return table([rows(method, K, seed, "a", "intent", 1.0)])
    monkeypatch.setattr(runner_mod, "evaluate_setting", fake)
    todo = [("default", 0, 0), ("bon_pred", 4, 0), ("refine", 10, 0)]
    tags = {"device_name": "gpu0", "code_digest": "d0"}
    failed = run_settings(None, todo, tmp_path, 0.05, {"git_hash": "x"}, "b", tags)
    assert failed == ["bon_pred_4_0_b"]
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == ["default_0_0_b.parquet", "refine_10_0_b.parquet"]
    t = pq.read_table(tmp_path / "refine_10_0_b.parquet").to_pydict()
    assert t["anchor_group"] == ["b"] and t["device_name"] == ["gpu0"] and t["code_digest"] == ["d0"]
    import json
    js = json.loads((tmp_path / "refine_10_0_b.json").read_text())
    assert js["anchor_group"] == "b" and js["device_name"] == "gpu0" and js["code_digest"] == "d0" and js["git_hash"] == "x"
    out = capsys.readouterr().out
    assert "FAILED bon_pred_4_0_b" in out and "RuntimeError: boom" in out
    assert run_settings(None, todo, tmp_path, 0.05, {}, "b", tags) == ["bon_pred_4_0_b"]  # finished settings are skipped
