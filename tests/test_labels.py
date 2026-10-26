import pytest
import torch

from spec2mpc.model.data import load_rollouts
from spec2mpc.rubric.cards import load_cards
from spec2mpc.rubric.labels import load_score_labels
from spec2mpc.rubric.scoring import card_max_score
from tests.test_gen_smoke import ROOT, _run
from tests.test_model_data import mpc_cfg, sim_cfg, tok_cfg, two_shards  # noqa: F401


@pytest.fixture(scope="module")
def cards():
    return load_cards(ROOT / "data/cards", "train")


@pytest.fixture(scope="module")
def scored(two_shards):  # noqa: F811
    _run("score_rollouts.py", str(two_shards), "--cards-root", str(ROOT / "data/cards"))
    return two_shards


def test_card_max_score(cards):
    for c in cards:
        assert card_max_score(c) == len(c.rubric.tiers) + (c.rubric.shaping.weight if c.rubric.shaping else 0)


def test_labels_join_and_task_mask(scored, cards):
    d = load_rollouts([scored], tok_cfg, mpc_cfg, sim_cfg)
    J, ok = load_score_labels(d, cards)
    assert J.shape == ok.shape == (64, len(cards))
    for i, c in enumerate(cards):
        other = d.task == (1 if c.task == "lap" else 0)
        assert torch.isnan(J[other, i]).all() and not ok[other, i].any()
        assert torch.isfinite(J[~other, i]).all()
        assert J[~other, i].abs().max() <= 1.5 + 5  # normalized by max score; penalties may push below 0


def test_missing_score_file_raises(scored, cards):
    d = load_rollouts([scored], tok_cfg, mpc_cfg, sim_cfg)
    p = scored / "shard_000000" / "scores" / f"{cards[0].id}.parquet"
    bak = p.read_bytes()
    p.unlink()
    try:
        with pytest.raises(FileNotFoundError, match=f"shard_000000.*{cards[0].id}"):
            load_score_labels(d, cards)
    finally:
        p.write_bytes(bak)


def test_manifest_excludes_unscored_and_incomplete_shards(tmp_path):
    from spec2mpc.rubric.labels import scored_shards, write_manifest
    ids = [c.id for s in ("train", "val") for c in load_cards(ROOT / "data/cards", s)]
    roll = tmp_path / "roll"
    for name, meta, scored in (("shard_a", True, ids), ("shard_b", True, ids[:-1]), ("shard_c", False, ids),
                               (".tmp_shard_d", True, ids)):
        d = roll / name
        (d / "scores").mkdir(parents=True)
        if meta:
            (d / "meta.json").write_text("{}")
        for cid in scored:
            (d / "scores" / f"{cid}.parquet").write_text("")
    assert scored_shards(roll, ids) == [roll / "shard_a"]
    out = tmp_path / "models" / "manifest.txt"
    assert write_manifest(roll, ROOT / "data/cards", out) == [roll / "shard_a"]
    assert out.read_text().split() == [str(roll / "shard_a")]
