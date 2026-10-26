from collections import Counter

import pytest
from omegaconf import OmegaConf

from spec2mpc.rubric import splits
from spec2mpc.rubric.splits import statement_records


@pytest.fixture
def split_cfg():
    return OmegaConf.load("configs/splits/kart.yaml")


def test_roles_partition_statements(split_cfg):
    recs = statement_records(split_cfg)
    by_role = Counter(r.role for r in recs)
    assert by_role["val_paraphrase"] == 8 and by_role["val_intent"] == 6     # 3 + 3 statements
    assert len({r.text for r in recs}) == len(recs)                            # no statement in two roles
    for cid in ("kart_fast_001",):
        assert [r.role for r in recs if r.card_id == cid].count("val_paraphrase") == 1


def test_val_intents_have_no_train_statements(split_cfg):
    assert not any(r.role == "train" and r.card_id in split_cfg.val_intents for r in statement_records(split_cfg))


def test_never_loads_test_splits(split_cfg, monkeypatch):
    seen = []
    real = splits.load_cards
    monkeypatch.setattr(splits, "load_cards", lambda root, s: seen.append(s) or real(root, s))
    statement_records(split_cfg)
    assert set(seen) == {"train", "val"}


def test_val_intents_mismatch_raises(split_cfg):
    split_cfg.val_intents = ["kart_camera_smooth_001"]
    with pytest.raises(ValueError):
        statement_records(split_cfg)
