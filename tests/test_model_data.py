import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.model.data import XI_NAMES, block_split, load_rollouts, normalize_xi
from tests.test_gen_smoke import ROOT, TINY

tok_cfg = OmegaConf.load(ROOT / "configs/robot_tokens/kart.yaml")
mpc_cfg = OmegaConf.load(ROOT / "configs/mpc/kart.yaml")
sim_cfg = OmegaConf.load(ROOT / "configs/sim/default.yaml")


@pytest.fixture(scope="module")
def two_shards(tmp_path_factory):
    out = tmp_path_factory.mktemp("rolls")
    args = [a for a in TINY if not a.startswith("n_blocks")]
    r = subprocess.run([sys.executable, "scripts/gen_rollouts.py", *args, "n_blocks=2", f"out_dir={out}"],
                       cwd=ROOT, env=dict(os.environ, RESUME="0"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return out


def test_load_rollouts_shapes_and_alignment(two_shards):
    d = load_rollouts([two_shards], tok_cfg, mpc_cfg, sim_cfg)
    assert d.robot_tok.shape == (64, 6, 24) and d.path_tok.shape == (64, 64, 6) and d.phi.shape == (64, 20)
    assert d.theta_n.shape == (64, 14) and d.xi_n.shape == (64, len(XI_NAMES)) == (64, 6)
    t = pq.read_table(two_shards / "shard_000000" / "episodes.parquet")
    assert torch.allclose(d.phi[:32, 0], torch.tensor(t.column("phi_completion").to_numpy()))
    assert (d.theta_n.abs() <= 1 + 1e-6).all() and (d.xi_n.abs() <= 1 + 1e-6).all()
    assert d.shard.tolist() == [0] * 32 + [1] * 32 and len(d.shard_dirs) == 2
    assert set(d.task.tolist()) == {0, 1}
    assert d.index(torch.arange(5)).phi.shape == (5, 20)


def test_incomplete_shards_skipped(two_shards, tmp_path):
    (two_shards / ".tmp_shard_000009").mkdir()
    assert load_rollouts([two_shards], tok_cfg, mpc_cfg, sim_cfg).phi.shape[0] == 64


def test_normalize_xi_range():
    lo = torch.tensor([r[0] for r in sim_cfg.conditions.values()])
    hi = torch.tensor([r[1] for r in sim_cfg.conditions.values()])
    assert torch.allclose(normalize_xi(lo, sim_cfg), -torch.ones(6)) and torch.allclose(normalize_xi(hi, sim_cfg), torch.ones(6))


def test_block_split_keeps_blocks_together():
    b = torch.arange(1000).repeat_interleave(16)
    v = block_split(b, 0.1, 0)
    assert (v.view(1000, 16).all(1) | (~v.view(1000, 16)).all(1)).all() and 0.05 < v.float().mean() < 0.15
    assert torch.equal(v, block_split(b, 0.1, 0)) and not torch.equal(v, block_split(b, 0.1, 1))


def test_manifest_loads_only_listed_shards(two_shards, tmp_path):
    m = tmp_path / "manifest.txt"
    m.write_text(f"{two_shards / 'shard_000001'}\n")
    d = load_rollouts([m], tok_cfg, mpc_cfg, sim_cfg)
    assert d.phi.shape[0] == 32 and d.shard_dirs == [two_shards / "shard_000001"]
