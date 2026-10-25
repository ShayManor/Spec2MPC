import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from spec2mpc.features.phi import PHI_NAMES
from spec2mpc.mpc.theta import THETA_NAMES
from spec2mpc.sim.storage import read_shard

ROOT = Path(__file__).resolve().parents[1]
TINY = [  # 32 envs, 0.3 s episodes, 2 solver iterations, 0.5 m cells: a few seconds on CPU
    "envs_per_batch=32", "n_blocks=1", "device=cpu", "seed=5",
    "sim.episode.timeout_cap_lap_s=0.3", "sim.episode.timeout_cap_goto_s=0.3",
    "robots.runtime.control_rate_hz=[20]", "robots.runtime.solver_iters=[2]",
    "robots.runtime.perception_rate_hz=[10]", "sim.grid.res_m=0.5",
]


def _run(script: str, *args: str, resume: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, RESUME="1" if resume else "0")
    r = subprocess.run([sys.executable, f"scripts/{script}", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    if check:
        assert r.returncode == 0, r.stderr[-3000:]
    return r


@pytest.fixture(scope="module")
def gen_dirs(tmp_path_factory):
    a, b = tmp_path_factory.mktemp("a"), tmp_path_factory.mktemp("b")
    log = _run("gen_rollouts.py", *TINY, f"out_dir={a}").stdout
    _run("gen_rollouts.py", *TINY, f"out_dir={b}")
    return a, b, log


def test_shard_contents(gen_dirs):
    a, _, log = gen_dirs
    assert sorted(p.name for p in a.iterdir()) == ["shard_000000"]
    table, z = read_shard(a / "shard_000000")
    assert table.num_rows == 32
    bid = table.column("block_id").to_numpy()
    assert sorted(np.unique(bid, return_counts=True)[1]) == [16, 16]
    for b in np.unique(bid):
        rows = bid == b
        for n in THETA_NAMES + ("scene_seed", "kart_mass_kg"):
            v = table.column(n).to_numpy()[rows]
            assert (v == v[0]).all(), n
        assert len(set(np.asarray(table.column("robot_id"))[rows])) == 1
        assert len(np.unique(table.column("xi_seed").to_numpy()[rows])) == 16  # 16 distinct condition draws
    assert set(table.column("task").to_pylist()) == {"lap", "goto"}
    cost, gn = table.column("mean_cost").to_numpy(), table.column("mean_grad_norm").to_numpy()
    ok = table.column("solver_fail_count").to_numpy() < 6  # 0.3 s at 20 Hz: 6 solves, at least one ok
    assert np.isfinite(cost[ok]).all() and np.isfinite(gn[ok]).all() and (gn[ok] >= 0).all()
    assert z["x"].shape[0] == 32 and z["path_tokens"].shape[0] == 32
    assert "shard_000000 envs=32" in log and "complete_lap=" in log and "solver_fail_rate=" in log


def test_same_seed_byte_identical_phi(gen_dirs):
    a, b, _ = gen_dirs
    ta, tb = read_shard(a / "shard_000000")[0], read_shard(b / "shard_000000")[0]
    for n in PHI_NAMES:
        assert ta.column(f"phi_{n}").to_numpy().tobytes() == tb.column(f"phi_{n}").to_numpy().tobytes(), n


def test_resume_skips_and_refuses_overwrite(gen_dirs):
    a, _, _ = gen_dirs
    before = (a / "shard_000000" / "meta.json").read_text()
    r = _run("gen_rollouts.py", *TINY, f"out_dir={a}", resume=True)
    assert "skip" in r.stdout
    assert (a / "shard_000000" / "meta.json").read_text() == before
    r = _run("gen_rollouts.py", *TINY, f"out_dir={a}", check=False)
    assert r.returncode != 0 and "RESUME=1" in r.stderr
    for change in ("seed=6", "envs_per_batch=48", "xi_per_block=8", "task_mix.lap=0.25"):
        r = _run("gen_rollouts.py", *TINY, f"out_dir={a}", change, resume=True, check=False)
        assert r.returncode != 0 and "different settings" in r.stderr, change
    assert (a / "shard_000000" / "meta.json").read_text() == before


def _card_dir(tmp_path: Path) -> tuple[Path, list[str]]:
    # train: the real train cards; val and test_intent: one renamed copy each (synthetic, not the real test split)
    root = tmp_path / "cards"
    for split in ("train", "val", "test_intent"):
        (root / split).mkdir(parents=True)
    train = sorted((ROOT / "data/cards/train").glob("*.yaml"))
    for p in train:
        (root / "train" / p.name).write_text(p.read_text())
    for split in ("val", "test_intent"):
        raw = yaml.safe_load(train[0].read_text())
        raw["id"] = f"{split}_copy"
        (root / split / f"{split}_copy.yaml").write_text(yaml.safe_dump(raw))
    return root, [yaml.safe_load(p.read_text())["id"] for p in train]


def test_score_train_cards_only(gen_dirs, tmp_path):
    a, _, _ = gen_dirs
    cards, train_ids = _card_dir(tmp_path)
    scores = a / "shard_000000" / "scores"
    _run("score_rollouts.py", str(a), "--cards-root", str(cards))
    assert sorted(p.stem for p in scores.glob("*.parquet")) == sorted(train_ids)
    table = read_shard(a / "shard_000000")[0]
    tasks = dict(zip(table.column("episode_id").to_pylist(), table.column("task").to_pylist()))
    for cid in train_ids:
        s = pq.read_table(scores / f"{cid}.parquet")
        assert s.column_names == ["episode_id", "J", "in_conditions"]
        card_task = yaml.safe_load((cards / "train" / f"{cid}.yaml").read_text())["task"]
        assert s.num_rows == 16 and all(tasks[e] == card_task for e in s.column("episode_id").to_pylist())
        assert np.isfinite(s.column("J").to_numpy()).all()
    _run("score_rollouts.py", str(a), "--cards-root", str(cards), "--val")
    names = sorted(p.stem for p in scores.glob("*.parquet"))
    assert names == sorted(train_ids + ["val_copy"])
    assert not any(p.name.startswith(".tmp") for p in scores.iterdir())
