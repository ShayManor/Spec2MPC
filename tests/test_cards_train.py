import re
from pathlib import Path

from spec2mpc.rubric.cards import load_cards

ROOT = Path(__file__).parent.parent / "data" / "cards"
GLOBAL_PRIOR = {"friction_scale": (0.6, 1.1), "mass_scale": (0.9, 1.15), "latency_s": (0.0, 0.06)}
NUM = re.compile(r"\d+(?:\.\d+)?")


def cards():
    return load_cards(ROOT, "train") + load_cards(ROOT, "val")


def test_ten_train_cards_load_and_compile():
    cs = cards()
    assert len(cs) == 10
    assert len({c.id for c in cs}) == 10
    assert len(load_cards(ROOT, "train")) == 8 and len(load_cards(ROOT, "val")) == 2
    for c in cs:
        assert c.family == ["kart"] and c.task in ("lap", "goto")
        assert 3 <= len(c.statements) <= 5
        assert c.rubric.conditions["episodes"] == 16
        assert c.provenance["author"] == "claude-draft" and c.provenance["source"] and c.provenance["blind"] is False


def test_both_tasks_covered():
    assert {c.task for c in cards()} == {"lap", "goto"}


def test_condition_ranges_inside_global_prior():
    for c in cards():
        for k, v in c.rubric.conditions.items():
            if k != "episodes":
                lo, hi = GLOBAL_PRIOR[k]
                assert lo <= v[0] <= v[1] <= hi, (c.id, k)


def test_statements_add_no_numbers():
    for c in cards():
        base = set(NUM.findall(c.statements[0]))
        for s in c.statements[1:]:
            assert set(NUM.findall(s)) <= base, (c.id, s)
