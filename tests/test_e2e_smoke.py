"""End-to-end smoke: every stage at tiny settings, through the Gautschi payload scripts where one exists.

S0 gen_rollouts (no E2E payload) and plot_transfer (run by hand after the train oracle) are called directly with the
arguments their usage lines give. Every other stage runs through its payload under the ideas8-rehearsal env
overrides, so the payloads' argument lines and $MODELS / $APPLV / $ORACLE wiring are what is tested.
stage_common.sh asserts CUDA is available, so the test needs a CUDA host; the stages themselves run on CPU.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="stage_common.sh asserts a CUDA device")]

TEXT = "text.encoder.backend=hashed"
GEN = [  # 2 shards x 64 envs (4 blocks each), 0.3 s episodes, 2 solver iterations, 0.5 m cells
    "envs_per_batch=64", "n_blocks=2", "device=cpu", "seed=5",
    "sim.episode.timeout_cap_lap_s=0.3", "sim.episode.timeout_cap_goto_s=0.3",
    "robots.runtime.control_rate_hz=[20]", "robots.runtime.solver_iters=[2]",
    "robots.runtime.perception_rate_hz=[10]", "sim.grid.res_m=0.5",
]
# the oracle run's sim and runtime are what applv rl and run_eval roll out with
ORACLE_EXTRA = ("device=cpu mpc.solver.compile=false problems.n_per_task.lap=1 problems.n_per_task.goto=1 "
                "cmaes.lam=4 cmaes.generations=1 cmaes.xi_per_candidate=2 final_xi=2 at_k=[4] "
                "sim.episode.timeout_cap_lap_s=0.3 sim.episode.timeout_cap_goto_s=0.3 runtime.solver_iters=2 "
                "sim.grid.res_m=0.5")
TRAIN_ENV = {
    "SEEDS": "0", "DEVICE": "cpu",
    "WM_EXTRA": "opt.epochs=1 opt.batch=64 data.val_frac=0.5",
    "CRITIC_EXTRA": f"{TEXT} opt.epochs=1 opt.batch=64 members=2 log_every=1 ci.n_boot=10 val_frac=0.5",
    "POLICY_EXTRA": f"{TEXT} epochs_distill=1 steps_vw=1 opt.batch=8 n_samples=2 n_xi=2 data.val_frac=0.5",
    "APPLV_EXTRA": f"{TEXT} bc.epochs=1",
}
APPLV_EXTRA = f"{TEXT} mpc.solver.compile=false rl.iterations=1 rl.rows_per_iter=2 rl.n_samples=2 rl.xi_per_sample=1"
EVAL_EXTRA = (f"{TEXT} seeds=[0] n_xi=2 n_xi_model=2 bon_true_xi=1 n_boot=10 budgets.bon_pred=[2] budgets.refine=[2] "
              "budgets.bon_true=[2] budgets.oracle_bo=[4]")
METHODS = ["default", "oracle", "oracle_bo", "mean_theta", "nn_retrieval", "slot_only", "applv_lang", "amortized",
           "bon_pred", "refine", "bon_true"]


def _env(**extra: str) -> dict:
    tmp = extra.pop("TMP")
    return dict(os.environ, REPO=str(ROOT), PY=sys.executable, TMPDIR=tmp, RESUME="0", **extra)


def _check(r: subprocess.CompletedProcess, what: str) -> str:
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"{what} exited {r.returncode}\n{out[-4000:]}"
    # payloads log a failed stage and continue; a smoke run must not hide one
    assert " exited " not in r.stdout, f"{what} logged a failed stage\n{out[-4000:]}"
    return r.stdout


def _script(name: str, *args: str, env: dict) -> str:
    r = subprocess.run([sys.executable, f"scripts/{name}", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    return _check(r, name)


def _payload(name: str, *args: str, env: dict) -> str:
    r = subprocess.run(["bash", f"scripts/gautschi/{name}", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    out = _check(r, name)
    assert "payload done" in out, out[-2000:]
    return out


def _card_ids() -> set:
    return {yaml.safe_load(p.read_text())["id"] for s in ("train", "val") for p in (ROOT / "data/cards" / s).glob("*.yaml")}


def test_pipeline_end_to_end(tmp_path):
    roll, oracle, models, ev = (tmp_path / n for n in ("rollouts", "oracle", "models", "eval"))
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    base = {"TMP": str(tmp), "MODELS": str(models)}
    train_parquet = str(oracle / "train" / "oracle.parquet")

    # S0 and its score pass (run by the S0 jobs, not the train payload)
    _script("gen_rollouts.py", *GEN, f"out_dir={roll}", env=_env(TMP=str(tmp)))
    assert sorted(p.name for p in roll.iterdir()) == ["shard_000000", "shard_000001"]
    shutil.copytree(roll / "shard_000001", roll / "shard_000002")  # left unscored: the manifest must exclude it
    for s in ("shard_000000", "shard_000001"):
        _script("score_rollouts.py", str(roll / s), "--val", env=_env(TMP=str(tmp)))

    # S1 oracle, both splits, then the transfer go/no-go
    for split in ("train", "val"):
        _payload("oracle_payload.sh", split, env=_env(TMP=str(tmp), OUT=str(oracle), ORACLE_EXTRA=ORACLE_EXTRA))
        for f in ("oracle.parquet", "at_k.parquet", "state.pt", "config.yaml", "problems.pt", "meta.json"):
            assert (oracle / split / f).exists(), (split, f)
    assert (oracle / "train" / "transfer.parquet").exists()
    png = tmp_path / "transfer.png"
    out = _script("plot_transfer.py", str(oracle / "train"), "--out", str(png), env=_env(TMP=str(tmp)))
    assert png.exists() and "overall verdict=" in out

    # S2 to S4 and the APPLV bc baseline on the frozen manifest (only the scored shard); a rerun skips every stage
    _payload("train_payload.sh", env=_env(**base, ROLL=str(roll), ORACLE=train_parquet, **TRAIN_ENV))
    assert (models / "manifest.txt").read_text().split() == [str(roll / s) for s in ("shard_000000", "shard_000001")]
    rerun = _payload("train_payload.sh", env=_env(**base, ROLL=str(roll), ORACLE=train_parquet, **TRAIN_ENV))
    assert rerun.count("skip ") == 5 and "reused" in rerun, rerun[-2000:]
    for d, files in {"world_model": ("model.pt", "metrics.json"), "critic": ("model.pt", "metrics.json"),
                     "policy_full": ("model.pt", "metrics.json"), "policy_zero": ("model.pt", "metrics.json"),
                     "applv": ("bc.pt", "bc_metrics.json")}.items():
        for f in files:
            assert (models / d / "seed0" / f).exists(), (d, f)

    # APPLV rl from the bc checkpoint
    _payload("applv_payload.sh", env=_env(**base, ORACLE=train_parquet, SEEDS="0", DEVICE="cpu",
                                          APPLV_EXTRA=APPLV_EXTRA))
    for f in ("model.pt", "rl_state.pt", "metrics.json"):
        assert (models / "applv" / "seed0" / f).exists(), f

    # evaluation: both groups cover every method at one budget and seed 0
    for group in ("a", "b"):
        _payload("eval_payload.sh", group, env=_env(**base, ORACLE_ROOT=str(oracle), OUT=str(ev), DEVICE="cpu",
                                                    EVAL_EXTRA=EVAL_EXTRA))
    ran = {p.stem.rsplit("_", 3)[0] for p in ev.glob("*.parquet") if p.name != "summary.parquet"}
    assert ran == set(METHODS), ran
    for g in ("a", "b"):
        assert (ev / f"default_0_0_{g}.parquet").exists() and (ev / f"oracle_0_0_{g}.parquet").exists(), g
    summary = pq.read_table(ev / "summary.parquet")
    assert all(n == 0 for n in summary.column("n_unanchored").to_pylist())
    assert set(summary.column("method").to_pylist()) == set(METHODS)
    assert all(r == 0.0 for r in summary.column("floor_violation_rate").to_pylist())
    for f in ("eval_score_vs_budget.png", "eval_critic_gap.png", "eval_methods.png"):
        assert (ev / "plots" / f).exists(), f

    # single-seed group s covers every method in one job, with its own anchors
    ev_s = tmp_path / "eval_s"
    _payload("eval_payload.sh", "s", env=_env(**base, ORACLE_ROOT=str(oracle), OUT=str(ev_s), DEVICE="cpu",
                                              EVAL_EXTRA=EVAL_EXTRA))
    assert {p.stem.rsplit("_", 3)[0] for p in ev_s.glob("*.parquet") if p.name != "summary.parquet"} == set(METHODS)
    summary_s = pq.read_table(ev_s / "summary.parquet")
    assert set(summary_s.column("method").to_pylist()) == set(METHODS)
    assert all(n == 0 for n in summary_s.column("n_unanchored").to_pylist())

    # no card outside train and val reaches any oracle or eval output
    allowed = _card_ids()
    for p in [oracle / s / "oracle.parquet" for s in ("train", "val")] + sorted(ev.glob("*.parquet")):
        if "card_id" in pq.read_schema(p).names:
            assert set(pq.read_table(p, columns=["card_id"]).column(0).to_pylist()) <= allowed, p
    assert set(pq.read_table(oracle / "train" / "transfer.parquet").column("rubric_card").to_pylist()) <= allowed
