from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from torch import Tensor

from spec2mpc.features.metrics import METRIC_NAMES

CONDITION_KEYS: tuple[str, ...] = ("friction_scale", "mass_scale", "latency_s")
TASKS: tuple[str, ...] = ("lap", "goto")
Expr = Callable[[dict[str, Tensor]], Tensor]


class CardError(ValueError):
    pass


@dataclass(frozen=True)
class Tier:
    name: str
    expr: Expr
    src: str
    metric: str | None  # metric of a `metric < X` / `metric <= X` tier, else None
    threshold: float | None  # constant X of such a tier


@dataclass(frozen=True)
class Shaping:
    expr: Expr
    weight: float
    metrics: frozenset[str]  # metric names in the margin term; T_next comes from tiers on these


@dataclass(frozen=True)
class Penalty:
    expr: Expr
    weight: float


@dataclass(frozen=True)
class Rubric:
    conditions: dict[str, Any]
    tiers: list[Tier]
    shaping: Shaping | None
    penalties: list[Penalty]


@dataclass(frozen=True)
class Card:
    id: str
    family: list[str]
    task: str
    statements: list[str]
    slots_expected: dict
    rubric: Rubric
    provenance: dict


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Eq: operator.eq}


def _num(x: Tensor) -> Tensor:
    return x.to(torch.float64) if x.dtype == torch.bool else x


def _clip(x: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    return torch.minimum(torch.maximum(x, lo), hi)


_CALLS: dict[str, tuple[int, Callable[..., Tensor]]] = {
    "clip": (3, _clip),
    "min": (2, torch.minimum),
    "max": (2, torch.maximum),
    "abs": (1, torch.abs),
}


def compile_expr(src: str, allowed: set[str], card_id: str = "?") -> Expr:
    """Compile a rubric expression by walking its AST; metrics are [E] tensors, result is [E]."""

    def fail(why: str) -> CardError:
        return CardError(f"{card_id}: {src}: {why}")

    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise fail(f"syntax error ({e.msg})") from None
    except ValueError as e:
        raise fail(f"unparseable ({e})") from None

    def build(n: ast.AST) -> Expr:
        if isinstance(n, ast.Name):
            if n.id not in allowed:
                raise fail(f"unknown name '{n.id}'")
            name = n.id
            return lambda env: env[name]
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
                raise fail("only numeric constants are allowed")
            try:
                val = torch.tensor(float(n.value), dtype=torch.float64)
            except OverflowError:
                raise fail("numeric constant out of range") from None
            if not torch.isfinite(val):
                raise fail("numeric constant out of range")
            return lambda env: val
        if isinstance(n, ast.BinOp):
            if type(n.op) not in _BIN:
                raise fail(f"operator {type(n.op).__name__} not allowed")
            op, l, r = _BIN[type(n.op)], build(n.left), build(n.right)
            return lambda env: op(_num(l(env)), _num(r(env)))
        if isinstance(n, ast.UnaryOp):
            if not isinstance(n.op, ast.USub):
                raise fail(f"operator {type(n.op).__name__} not allowed")
            o = build(n.operand)
            return lambda env: -_num(o(env))
        if isinstance(n, ast.Compare):
            if len(n.ops) != 1 or type(n.ops[0]) not in _CMP:
                raise fail("only a single comparison (< <= > >= ==) is allowed")
            op, l, r = _CMP[type(n.ops[0])], build(n.left), build(n.comparators[0])
            return lambda env: op(_num(l(env)), _num(r(env)))
        if isinstance(n, ast.BoolOp):
            parts = [build(v) for v in n.values]
            comb = torch.logical_and if isinstance(n.op, ast.And) else torch.logical_or

            def boolop(env: dict[str, Tensor]) -> Tensor:
                def truth(x: Tensor) -> Tensor:
                    return ~torch.isnan(x) & (x != 0)  # NaN operand is false
                out = truth(parts[0](env))
                for p in parts[1:]:
                    out = comb(out, truth(p(env)))
                return out
            return boolop
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in _CALLS or n.keywords:
                raise fail("only clip, min, max, abs calls are allowed")
            arity, fn = _CALLS[n.func.id]
            if len(n.args) != arity:
                raise fail(f"{n.func.id} takes {arity} arguments")
            args = [build(a) for a in n.args]
            return lambda env: fn(*[_num(a(env)) for a in args])
        raise fail(f"{type(n).__name__} is not allowed")

    f = build(tree.body)

    def run(env: dict[str, Tensor]) -> Tensor:
        out = f(env)
        ref = next(iter(env.values()))
        return out.expand(ref.shape) if out.dim() == 0 else out
    return run


def _threshold(src: str) -> tuple[str, float] | tuple[None, None]:
    try:
        n = ast.parse(src, mode="eval").body
    except (SyntaxError, ValueError):
        return None, None
    if (isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and len(n.ops) == 1
            and isinstance(n.ops[0], (ast.Lt, ast.LtE)) and isinstance(n.comparators[0], ast.Constant)
            and isinstance(n.comparators[0].value, (int, float))):
        try:
            return n.left.id, float(n.comparators[0].value)
        except OverflowError:
            return None, None
    return None, None


def compile_card(raw: dict) -> Card:
    cid = str(raw.get("id", "?"))
    metrics = set(METRIC_NAMES)
    try:
        family = raw["family"]
        family = [family] if isinstance(family, str) else list(family)
        task = raw["task"]
        if task not in TASKS:
            raise CardError(f"{cid}: task '{task}' not in {TASKS}")
        rub = raw["rubric"]
        conditions = dict(rub.get("conditions", {}))
        for k in conditions:
            if k != "episodes" and k not in CONDITION_KEYS:
                raise CardError(f"{cid}: unknown condition key '{k}'")
        if not isinstance(conditions.get("episodes"), int):
            raise CardError(f"{cid}: rubric.conditions.episodes must be an int")
        tiers = [Tier(t["name"], compile_expr(t["require"], metrics, cid), t["require"], *_threshold(t["require"]))
                 for t in rub["tiers"]]
        shaping = None
        if rub.get("shaping"):
            s = rub["shaping"]
            expr = compile_expr(s["margin_term"], metrics | {"T_next"}, cid)
            used = frozenset(n.id for n in ast.walk(ast.parse(s["margin_term"], mode="eval")) if isinstance(n, ast.Name)) & metrics
            if "T_next" in s["margin_term"] and not any(t.metric in used for t in tiers):
                raise CardError(f"{cid}: {s['margin_term']}: T_next needs a `metric < X` tier on a metric in the term")
            shaping = Shaping(expr, float(s["weight"]), used)
        penalties = [Penalty(compile_expr(p["expr"], metrics, cid), float(p["weight"]))
                     for p in rub.get("penalties", [])]
        return Card(cid, family, task, list(raw["statements"]), dict(raw.get("slots_expected") or {}),
                    Rubric(conditions, tiers, shaping, penalties), dict(raw.get("provenance") or {}))
    except CardError:
        raise
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        raise CardError(f"{cid}: malformed card ({type(e).__name__}: {e})") from None


def load_cards(root: Path, split: str) -> list[Card]:
    if split != Path(split).name or split in ("", ".", "..") or "\\" in split:
        raise PermissionError(f"split must be a bare directory name, got '{split}'")
    if split.lower().startswith("test"):
        raise PermissionError(f"refusing to load rubrics of split '{split}' (CLAUDE.md section 11)")
    return [compile_card(yaml.safe_load(p.read_text())) for p in sorted((Path(root) / split).glob("*.yaml"))]
