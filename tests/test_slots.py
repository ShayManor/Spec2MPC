import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.rubric.cards import load_cards
from spec2mpc.text.slots import Slot, parse_slots, slot_matrix, slot_vector, slots_dict

ROOT = Path(__file__).parent.parent / "data" / "cards"


@pytest.fixture
def slot_cfg():
    return OmegaConf.load("configs/text/default.yaml").slots


@pytest.mark.parametrize("text, expected", [
    ("Deliver the package in under 2 minutes", {"t_max_s": 120.0}),
    ("Courier kart with a 2 minute delivery window", {"t_max_s": 120.0}),
    ("a two-minute window", {"t_max_s": 120.0}),
    ("finish within 90 s and keep it below 0.3 g", {"t_max_s": 90.0, "a_lat_max_g": 0.3}),
    ("no faster than 10 km/h", {"v_max_mps": pytest.approx(2.7778, rel=1e-4)}),
    ("use at most 0.5 kWh", {"energy_max_Wh": 500.0}),
    ("Autonomous kart that goes fast", {}),
    ("half a minute to the dock", {"t_max_s": 30.0}),
])
def test_typed_slots(text, expected):
    assert slots_dict(parse_slots(text)) == expected


def test_generic_number():
    assert parse_slots("kart number 7") == [Slot("generic", 7.0, "")]


def test_never_raises():
    for s in ["", "...", "1e999 minutes", "minutes", "-3 s", "١٢ min", "2" * 500]:
        parse_slots(s)


def test_cards_slots_expected_match():
    for c in load_cards(ROOT, "train") + load_cards(ROOT, "val"):
        for s in c.statements:
            assert slots_dict(parse_slots(s)) == c.slots_expected, (c.id, s)


def test_slot_vector_layout(slot_cfg):
    v = slot_vector(parse_slots("under 2 minutes"), slot_cfg)
    assert v.dtype == torch.float32
    assert v.shape == (10,) and v[0] == 1 and v[1] == pytest.approx(math.log1p(2.0)) and v[2:].abs().sum() == 0


def test_slot_matrix(slot_cfg):
    m = slot_matrix(["under 2 minutes", "fast"], slot_cfg)
    assert m.shape == (2, 10) and m.dtype == torch.float32 and m[1].abs().sum() == 0


def test_non_finite_skipped(slot_cfg):
    assert parse_slots("2" * 400 + " s") == []
    m = slot_matrix(["2" * 400 + " s", "under 2 minutes", ""], slot_cfg)
    assert m.shape == (3, 10) and torch.isfinite(m).all()
