import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
from hydra import compose, initialize_config_dir

import spec2mpc.baselines.oracle_bo as oracle_bo
from spec2mpc.baselines.oracle_bo import (OracleJob, build_job, final_conditions, final_gen, final_layout, final_scores,
                                          run_oracle)
from spec2mpc.mpc.theta import ThetaBatch
from spec2mpc.sim.evaluate import rollout_metrics
from spec2mpc.mpc.theta import THETA_NAMES, default_theta, theta_bounds
from spec2mpc.rubric.cards import load_cards
from spec2mpc.sim.problems import load_oracle_problems, problem_digest

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
TINY = [  # tests/test_gen_smoke.py TINY episode settings, 1 lap + 1 goto problem, tiny CMA budget
    "sim.episode.timeout_cap_lap_s=0.3", "sim.episode.timeout_cap_goto_s=0.3", "sim.grid.res_m=0.5",
    "runtime.solver_iters=2", "problems.n_per_task.lap=1", "problems.n_per_task.goto=1",
    "cmaes.lam=4", "cmaes.generations=2", "cmaes.xi_per_candidate=2", "final_xi=2", "at_k=[4,8]",
]


def tiny_cfg(*extra: str):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name="oracle", overrides=TINY + list(extra))


def tiny_job(cfg) -> OracleJob:
    # 2 lap + 2 goto train cards
    job = build_job(cfg)
    cards = load_cards(ROOT / "data/cards", "train")
    keep = [c for t in ("lap", "goto") for c in [c for c in cards if c.task == t][:2]]
    pairs = [(p, c) for p in range(job.problems.n) for c in range(len(keep)) if job.problems.task[p] == keep[c].task]
    return OracleJob(problems=job.problems, cards=keep, pairs=pairs)


@pytest.fixture(scope="module")
def tiny_oracle_cfg():
    return tiny_cfg()


@pytest.fixture(scope="module")
def job(tiny_oracle_cfg):
    return tiny_job(tiny_oracle_cfg)


@pytest.fixture(scope="module")
def run_a(tmp_path_factory, job, tiny_oracle_cfg):
    out = tmp_path_factory.mktemp("a")
    run_oracle(job, tiny_oracle_cfg, out, CPU)
    return out


def test_oracle_outputs(run_a, job, tiny_oracle_cfg):
    d = run_a / "train"
    t = pq.read_table(d / "oracle.parquet").to_pydict()
    assert len(t["card_id"]) == len(job.pairs) == 4
    assert all(o == max(m, b, d) for o, m, b, d in zip(t["J_oracle"], t["J_mean"], t["J_best"], t["J_default"]))
    assert all(o >= d for o, d in zip(t["J_oracle"], t["J_default"]))
    assert set(t["theta_star_source"]) <= {"mean", "best", "default"}
    for o, src, m, b, jd in zip(t["J_oracle"], t["theta_star_source"], t["J_mean"], t["J_best"], t["J_default"]):
        assert o == {"mean": m, "best": b, "default": jd}[src]
    lo, hi = theta_bounds(tiny_oracle_cfg.mpc)
    th = torch.tensor([t[n] for n in THETA_NAMES]).T
    assert ((th >= lo - 1e-6) & (th <= hi + 1e-6)).all()
    assert set(t["n_evals"]) == {8}
    k = pq.read_table(d / "at_k.parquet").to_pydict()
    assert sorted(set(k["k"])) == [4, 8] and len(k["k"]) == 2 * len(job.pairs)
    for f in ("state.pt", "config.yaml", "meta.json"):
        assert (d / f).exists(), f


def test_problems_file_matches_rebuild(run_a, job):
    d = run_a / "train"
    st = torch.load(d / "problems.pt", weights_only=True)
    assert st["problem_id"] == job.problems.problem_id
    assert len(st["digest"]) == job.problems.n and st["digest"] == problem_digest(job.problems)
    problems, cfg = load_oracle_problems(d)  # rebuilt from the saved, resolved config; digests checked inside
    assert problems.problem_id == job.problems.problem_id
    assert cfg.robots and cfg.sim and cfg.tokenize and cfg.mpc and cfg.problems and cfg.runtime


def test_card_task_matching(job):
    assert all(job.problems.task[p] == job.cards[c].task for p, c in job.pairs)
    assert len(job.pairs) == 4


def test_build_job_pairs_every_matching_card(tiny_oracle_cfg):
    job = build_job(tiny_oracle_cfg)
    assert len(job.cards) == 8 and job.problems.n == 2
    assert len(job.pairs) == 8 and all(job.problems.task[p] == job.cards[c].task for p, c in job.pairs)
    val = build_job(tiny_cfg("split=val"))
    assert len(val.cards) == 10 and val.problems.problem_id[0].startswith("val_")


def test_refuses_overwrite_without_resume(run_a, job, tiny_oracle_cfg):
    with pytest.raises(FileExistsError):
        run_oracle(job, tiny_oracle_cfg, run_a, CPU)


def test_resume_matches_uninterrupted(tmp_path, run_a, job, tiny_oracle_cfg):
    run_oracle(job, tiny_cfg("cmaes.generations=1"), tmp_path, CPU)
    run_oracle(job, tiny_oracle_cfg, tmp_path, CPU, resume=True)
    a = pq.read_table(run_a / "train" / "oracle.parquet").to_pydict()
    b = pq.read_table(tmp_path / "train" / "oracle.parquet").to_pydict()
    for n in THETA_NAMES + ("J_oracle", "J_default"):
        assert a[n] == b[n], n
    ka = pq.read_table(run_a / "train" / "at_k.parquet").to_pydict()
    kb = pq.read_table(tmp_path / "train" / "at_k.parquet").to_pydict()
    assert ka == kb
    with pytest.raises(ValueError, match="different settings"):
        run_oracle(job, tiny_cfg("cmaes.sigma0=0.5"), tmp_path, CPU, resume=True)
    with pytest.raises(ValueError, match="different settings"):
        run_oracle(job, tiny_cfg("final_xi=3"), tmp_path, CPU, resume=True)


def test_one_rollout_call_per_generation(monkeypatch, tmp_path, job):
    calls = []
    real = oracle_bo.rollout_metrics

    def counting(robot, scene, theta, *a, **k):
        calls.append(theta.batch_size)
        return real(robot, scene, theta, *a, **k)

    monkeypatch.setattr(oracle_bo, "rollout_metrics", counting)
    run_oracle(job, tiny_cfg("cmaes.generations=1"), tmp_path, CPU)
    # pairs * lam * xi, then one pairs * final_xi call per final theta, then one transfer call (2 cards per task)
    assert calls == [4 * 4 * 2] + [4 * 2] * 3 + [4 * 2]


def test_final_conditions_shared_seed(job, tiny_oracle_cfg):
    sim = tiny_oracle_cfg.sim
    a = final_conditions(job, 0, 1, 16, sim, 7)
    b = final_conditions(job, 0, 1, 16, sim, 7)
    c = final_conditions(job, 0, 0, 16, sim, 7)
    assert torch.equal(a.friction_scale, b.friction_scale) and torch.equal(a.seed, b.seed)
    assert not torch.equal(a.seed, c.seed)


@pytest.mark.slow  # Hydra subprocesses, about 25 s
def test_script_prints_generations_and_resumes(tmp_path):
    args = [sys.executable, "scripts/run_oracle.py", *TINY, "cmaes.generations=1", f"out_dir={tmp_path}"]
    r = subprocess.run(args, cwd=ROOT, env=dict(os.environ, RESUME="0"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "gen=1 eps=64 " in r.stdout and "eps_per_s=" in r.stdout and "mean_best_J=" in r.stdout
    r = subprocess.run(args, cwd=ROOT, env=dict(os.environ, RESUME="0"), capture_output=True, text=True)
    assert r.returncode != 0 and "RESUME=1" in r.stderr
    r = subprocess.run(args[:-2] + ["cmaes.generations=2", f"out_dir={tmp_path}"], cwd=ROOT,
                       env=dict(os.environ, RESUME="1"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "gen=2 " in r.stdout and "gen=1 " not in r.stdout


def test_default_final_reproduces_standalone(run_a, job, tiny_oracle_cfg):
    t = pq.read_table(run_a / "train" / "oracle.parquet").to_pydict()
    J = final_scores(job, tiny_oracle_cfg, default_theta(len(job.pairs), tiny_oracle_cfg.mpc).values, CPU)
    assert J.shape == (len(job.pairs), 2)
    assert J.mean(1).tolist() == t["J_default"]
    # tiny episodes score J = 0, so also check the paired layout reproduces phi byte for byte
    rep, cond = final_layout(job, tiny_oracle_cfg)
    robot, scene = job.problems.expand(rep)
    th = ThetaBatch(values=default_theta(rep.shape[0], tiny_oracle_cfg.mpc).values)
    a = rollout_metrics(robot, scene, th, cond, tiny_oracle_cfg, final_gen(tiny_oracle_cfg), CPU).phi
    b = rollout_metrics(robot, scene, th, cond, tiny_oracle_cfg, final_gen(tiny_oracle_cfg), CPU).phi
    assert a.numpy().tobytes() == b.numpy().tobytes()


def test_at_k_must_be_multiple_of_lam(tmp_path, job):
    with pytest.raises(AssertionError, match="at_k"):
        run_oracle(job, tiny_cfg("at_k=[6]"), tmp_path, CPU)


def test_resume_verifies_problems(tmp_path, job, tiny_oracle_cfg):
    cfg1 = tiny_cfg("cmaes.generations=1")
    run_oracle(job, cfg1, tmp_path, CPU)
    p = tmp_path / "train" / "problems.pt"
    st = torch.load(p, weights_only=True)
    torch.save({"problem_id": st["problem_id"], "digest": ["0" * 32] * len(st["digest"])}, p)
    with pytest.raises(ValueError, match="problems.pt"):
        run_oracle(job, tiny_oracle_cfg, tmp_path, CPU, resume=True)


def test_resume_after_finals_reuses_them(monkeypatch, tmp_path, run_a, job, tiny_oracle_cfg):
    # a job killed between the finals and the transfer rollouts: the resume runs only the transfer call
    run_oracle(job, tiny_oracle_cfg, tmp_path, CPU)
    for f in ("transfer.parquet", "meta.json"):
        (tmp_path / "train" / f).unlink()
    calls = []
    real = oracle_bo.rollout_metrics

    def counting(robot, scene, theta, *a, **k):
        calls.append(theta.batch_size)
        return real(robot, scene, theta, *a, **k)

    monkeypatch.setattr(oracle_bo, "rollout_metrics", counting)
    run_oracle(job, tiny_oracle_cfg, tmp_path, CPU, resume=True)
    assert calls == [4 * 2]
    for f in ("oracle.parquet", "transfer.parquet"):
        assert pq.read_table(run_a / "train" / f).to_pydict() == pq.read_table(tmp_path / "train" / f).to_pydict(), f


def test_resume_recomputes_stale_finals(tmp_path, run_a, job, tiny_oracle_cfg):
    # finals of an earlier, shorter run are not reused once more generations have run
    run_oracle(job, tiny_cfg("cmaes.generations=1"), tmp_path, CPU)
    stale = pq.read_table(tmp_path / "train" / "oracle.parquet").to_pydict()
    run_oracle(job, tiny_oracle_cfg, tmp_path, CPU, resume=True)
    fresh = pq.read_table(tmp_path / "train" / "oracle.parquet").to_pydict()
    assert fresh["n_evals"] != stale["n_evals"]
    assert fresh == pq.read_table(run_a / "train" / "oracle.parquet").to_pydict()
