from pathlib import Path

import pytest
import torch
import yaml

from spec2mpc.features.metrics import METRIC_NAMES
from spec2mpc.rubric.cards import Card, CardError, compile_card, compile_expr, load_cards
from spec2mpc.rubric.scoring import card_score, in_card_conditions, score

FIXTURES = Path(__file__).parent / "fixtures" / "cards"


def t(*v: float) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float64)


def metrics(**kw: torch.Tensor) -> dict[str, torch.Tensor]:
    n = len(next(iter(kw.values())))
    base = {k: torch.zeros(n, dtype=torch.float64) for k in METRIC_NAMES}
    base.update(kw)
    return base


def raw_card() -> dict:
    return yaml.safe_load((FIXTURES / "train" / "kart_fast_001.yaml").read_text())


def yaml_with(expr: str, field: str = "tier") -> dict:
    raw = raw_card()
    raw["id"] = "kart_bad_001"
    if field == "tier":
        raw["rubric"]["tiers"][1]["require"] = expr
    else:
        raw["rubric"]["penalties"][0]["expr"] = expr
    return raw


@pytest.fixture
def card_fast() -> Card:
    return load_cards(FIXTURES, "train")[0]


def test_tiers_stop_at_first_failure(card_fast):
    m = metrics(completion=t(1., 1., 0.), lap_time_s=t(200., 700., 50.))
    shape_200 = (120 - 200) / 120
    shape_700 = (600 - 700) / 600
    out = score(card_fast, m)
    # lap_time 200 passes complete, 600, 300; fails 120 -> 3 tiers; shaping clipped to 0
    assert out.tolist() == pytest.approx([3 + 0.5 * max(shape_200, 0), 1 + 0.5 * max(shape_700, 0), 0.0])


def test_shaping_uses_next_unreached_threshold(card_fast):
    m = metrics(completion=t(1.), lap_time_s=t(140.))
    # passes 600, 300; next is 120; 140 > 120 so clip((120-140)/120) = 0
    assert score(card_fast, m).item() == pytest.approx(3.0)
    # not complete: order stops at tier 0 and shaping is withheld
    m = metrics(completion=t(0.), lap_time_s=t(540.))
    assert score(card_fast, m).item() == pytest.approx(0.0)  # no shaping credit when tier 0 fails


def test_all_tiers_met_uses_last_threshold(card_fast):
    m = metrics(completion=t(1.), lap_time_s=t(90.))
    assert score(card_fast, m).item() == pytest.approx(4 + 0.5 * (120 - 90) / 120)


def test_penalties_subtract(card_fast):
    m = metrics(completion=t(1., 1.), lap_time_s=t(500., 500.),
                boundary_violations=t(0., 3.), solver_failures=t(0., 1.))
    out = score(card_fast, m)
    assert out[0] - out[1] == pytest.approx(3 * 1.0 + 1 * 2.0)


def test_nan_metric_fails_comparison(card_fast):
    m = metrics(completion=t(1.), lap_time_s=t(float("nan")))
    assert score(card_fast, m).item() == pytest.approx(1.0)


@pytest.mark.parametrize("src", [
    "__import__('os').system('x')",
    "lap_time_s.real < 3",
    "open('f')",
    "lap_time_s ** 2 < 3",
    "[1, 2]",
    "'a' < 'b'",
    "1 < lap_time_s < 3",
    "lap_time_s if completion else 3",
    "lambda: 1",
    "sum(lap_time_s)",
])
def test_rejects_unsafe_expression(src):
    with pytest.raises(CardError, match="kart_bad_001"):
        compile_card(yaml_with(src))


def test_unknown_metric_named_in_error():
    with pytest.raises(CardError, match="not_a_metric") as e:
        compile_card(yaml_with("not_a_metric < 3"))
    assert "kart_bad_001" in str(e.value)


def test_penalty_expressions_validated():
    with pytest.raises(CardError, match="kart_bad_001"):
        compile_card(yaml_with("__import__('os')", field="penalty"))


def test_compile_expr_ops():
    f = compile_expr("max(a, 1) + abs(-b) * 2 - min(a, b) / 2 > 3 and a == a", {"a", "b"})
    out = f({"a": t(2., 0.), "b": t(1., 0.)})
    assert out.tolist() == [True, False]


def test_unparseable_t_next_is_error():
    raw = raw_card()
    raw["rubric"]["tiers"] = [{"name": "c", "require": "completion == 1"}]
    with pytest.raises(CardError, match="T_next"):
        compile_card(raw)


def test_unknown_condition_key_and_task():
    raw = raw_card()
    raw["rubric"]["conditions"]["wind"] = [0, 1]
    with pytest.raises(CardError, match="kart_fast_001"):
        compile_card(raw)
    raw = raw_card()
    raw["task"] = "swim"
    with pytest.raises(CardError, match="kart_fast_001"):
        compile_card(raw)


def test_condition_filter_and_rerun_flag(card_fast):
    cond = {"friction_scale": t(0.8, 0.5, 0.9), "mass_scale": t(1., 1., 1.),
            "latency_s": t(0.02, 0.02, 0.05)}
    assert in_card_conditions(card_fast, cond).tolist() == [True, False, False]
    m = metrics(completion=t(1., 1., 1.), lap_time_s=t(100., 100., 100.))
    mean_j, n_used, needs = card_score(card_fast, m, cond)
    assert n_used == 1 and needs and mean_j == pytest.approx(score(card_fast, m)[0].item())
    cond = {"friction_scale": t(0.8, 0.9), "mass_scale": t(1., 1.), "latency_s": t(0.02, 0.03)}
    _, n_used, needs = card_score(card_fast, metrics(completion=t(1., 1.), lap_time_s=t(1., 1.)), cond)
    assert n_used == 2 and not needs


def test_test_splits_refused(tmp_path):
    with pytest.raises(PermissionError):
        load_cards(tmp_path, "test_intent")


def test_load_cards_fields(card_fast):
    assert card_fast.id == "kart_fast_001" and card_fast.family == ["kart"] and card_fast.task == "lap"
    assert len(card_fast.rubric.tiers) == 4 and card_fast.rubric.conditions["episodes"] == 2


def test_t_next_ignores_tiers_on_other_metrics():
    raw = raw_card()
    raw["rubric"]["tiers"].insert(1, {"name": "slow", "require": "speed_max_mps < 3"})
    card = compile_card(raw)
    m = metrics(completion=t(1.), speed_max_mps=t(5.), lap_time_s=t(400.))
    # fails the speed tier (1 tier met); T_next is the lap-time threshold 600, not the speed threshold 3
    assert score(card, m).item() == pytest.approx(1 + 0.5 * (600 - 400) / 600)


def test_nan_penalty_contributes_zero_and_card_mean_finite(card_fast):
    m = metrics(completion=t(1., 1.), lap_time_s=t(500., 500.), boundary_violations=t(float("nan"), 0.))
    out = score(card_fast, m)
    assert torch.isfinite(out).all() and out[0] == pytest.approx(out[1])
    nan = float("nan")
    m = metrics(completion=t(1., nan), lap_time_s=t(500., nan), boundary_violations=t(0., nan),
                solver_failures=t(0., nan))
    cond = {"friction_scale": t(0.8, 0.8), "mass_scale": t(1., 1.), "latency_s": t(0.02, 0.02)}
    mean_j, n_used, _ = card_score(card_fast, m, cond)
    assert n_used == 2 and mean_j == mean_j and abs(mean_j) != float("inf")


def test_boolop_nan_operand_is_false():
    f = compile_expr("a < 1 or b", {"a", "b"})
    nan = float("nan")
    assert f({"a": t(5., 5.), "b": t(nan, 1.)}).tolist() == [False, True]
    g = compile_expr("b and a < 1", {"a", "b"})
    assert g({"a": t(0., 0.), "b": t(nan, 2.)}).tolist() == [False, True]


@pytest.mark.parametrize("split", ["./test_intent", "train/../test_intent", "../x", "..", "", "a/b", "test_x", "TEST_intent"])
def test_split_must_be_bare_and_not_test(split):
    with pytest.raises(PermissionError):
        load_cards(FIXTURES, split)


@pytest.mark.parametrize("lit", ["1" + "0" * 400, "1e999", "9" * 5000])
def test_huge_literal_is_card_error(lit):
    with pytest.raises(CardError, match="kart_bad_001"):
        compile_card(yaml_with(f"lap_time_s < {lit}"))


@pytest.mark.parametrize("where", ["penalty", "shaping"])
def test_non_numeric_weight_is_card_error(where):
    raw = raw_card()
    raw["id"] = "kart_bad_001"
    if where == "penalty":
        raw["rubric"]["penalties"][0]["weight"] = "abc"
    else:
        raw["rubric"]["shaping"]["weight"] = "abc"
    with pytest.raises(CardError, match="kart_bad_001"):
        compile_card(raw)
