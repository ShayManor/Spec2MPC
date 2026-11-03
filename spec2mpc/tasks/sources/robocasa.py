"""RoboCasa365 kitchen tasks as training cards.

The task list (365 tasks: atomic and composite, with horizons) comes from robocasa/utils/dataset_registry.py. Each
task's `_check_success` is translated by static analysis of the RoboCasa source (no simulator import), so this
module runs in the core env as long as a RoboCasa source tree is on disk (envs/robocasa venv or a clone).

Translated checks are conjunctions of these RoboCasa helpers, with literal object names or fixture attributes:
  OU.obj_inside_of(env, o, self.f)            -> inside(o, f)
  OU.check_obj_in_receptacle(env, o, r)       -> inside(o, r; tol_m 0.02)
  OU.check_obj_fixture_contact(env, o, self.f) -> on(o, f)
  OU.check_obj_upright(env, o, th_deg)        -> upright(o; tol_rad)
  OU.check_obj_grasped(env, o)                -> held(o)
  self.f.is_open(env, th)                     -> open_frac(f) >= th
  OU.gripper_obj_far / gripper_fxtr_far       -> dropped (no "far from gripper" predicate; on/inside with support
                                                 contact already excludes held), counted per task
Anything else (or, not, loops, appliance state, "closed" checks, which need an upper bound open_frac lacks) skips
the task with a reason.
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import math
import os
import re
import warnings
from pathlib import Path

from spec2mpc.core import RobotSpec, TaskCard
from spec2mpc.tasks.sources import Translation, make_cards, pred, source_robot, write_coverage

SOURCE = "robocasa"
CONTROL_HZ = 20.0  # RoboCasa control frequency; registry horizons are in control steps
DEFAULT_HORIZON = 1000  # robosuite default, for registered tasks without a dataset entry
_KITCHEN = "robocasa.environments.kitchen."
_SRC_CANDIDATES = (
    "/home/shay/languagespec-assets/venvs/robocasa/lib/python3.11/site-packages/robocasa",
    "/home/shay/languagespec-assets/src/robocasa/robocasa",
)
# checks with a loop or branch the static translator cannot read, written by hand from the source
_OVERRIDES: dict[str, tuple[list[dict], list[str]]] = {
    "OpenDrawer": ([pred("open_frac", "drawer", min=0.95)], []),  # every drawer joint >= 0.95
}


def source_dir() -> Path | None:
    # RoboCasa package dir: $ROBOCASA_SRC, the importable package, then the known asset locations
    cands = [os.environ.get("ROBOCASA_SRC", "")]
    spec = importlib.util.find_spec("robocasa") if importlib.util.find_spec is not None else None
    if spec is not None and spec.submodule_search_locations:
        cands += list(spec.submodule_search_locations)
    cands += list(_SRC_CANDIDATES)
    for c in cands:
        if c and (Path(c) / "utils" / "dataset_registry.py").exists():
            return Path(c)
    return None


def robots() -> dict[str, RobotSpec]:
    return {"default": source_robot("mobile_manipulator", "robocasa_panda_omron", "robosuite:PandaOmron",
                                    ["base", "ee", "gripper"], CONTROL_HZ)}


# ---- task registry ----

def _parse(p: Path) -> ast.Module:
    with warnings.catch_warnings():  # RoboCasa sources carry invalid escape sequences in strings
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        return ast.parse(p.read_text())


def registry(src: Path) -> dict[str, int]:
    # task name -> horizon (control steps): every kitchen env robocasa/__init__.py registers, with the horizon
    # from ATOMIC_ and COMPOSITE_TASK_DATASETS when the task has a dataset there
    init = _parse(src / "__init__.py")
    out: dict[str, int] = {a.asname or a.name: DEFAULT_HORIZON for n in init.body
                           if isinstance(n, ast.ImportFrom) and (n.module or "").startswith(_KITCHEN)
                           for a in n.names if (a.asname or a.name) != "Kitchen"}
    tree = _parse(src / "utils" / "dataset_registry.py")
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("ATOMIC_TASK_DATASETS", "COMPOSITE_TASK_DATASETS")):
            continue
        for kw in node.value.keywords:
            h = next((k.value.value for k in kw.value.keywords if k.arg == "horizon"), 0)
            out[kw.arg] = int(h)
    return out


def classes(src: Path) -> dict[str, ast.ClassDef]:
    out = {}
    for p in sorted((src / "environments" / "kitchen").rglob("*.py")):
        for node in ast.walk(_parse(p)):
            if isinstance(node, ast.ClassDef):
                out.setdefault(node.name, node)
    return out


def _bases(c: ast.ClassDef) -> list[str]:
    return [b.id if isinstance(b, ast.Name) else b.attr for b in c.bases if isinstance(b, (ast.Name, ast.Attribute))]


def _method(name: str, cls: dict[str, ast.ClassDef], meth: str) -> ast.FunctionDef | None:
    # method resolution along the first-base chain
    seen = set()
    while name in cls and name not in seen:
        seen.add(name)
        c = cls[name]
        for f in c.body:
            if isinstance(f, ast.FunctionDef) and f.name == meth:
                return f
        bs = _bases(c)
        name = bs[0] if bs else ""
    return None


def _behavior(name: str, cls: dict[str, ast.ClassDef]) -> str | None:
    # behavior="open"/"close" passed up through super().__init__ calls
    seen = set()
    while name in cls and name not in seen:
        seen.add(name)
        for n in ast.walk(cls[name]):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "__init__":
                for kw in n.keywords:
                    if kw.arg == "behavior" and isinstance(kw.value, ast.Constant):
                        return str(kw.value.value)
            if isinstance(n, ast.FunctionDef) and n.name == "__init__":  # def __init__(self, behavior="open")
                a = n.args.args[len(n.args.args) - len(n.args.defaults):]
                for arg, d in zip(a, n.args.defaults):
                    if arg.arg == "behavior" and isinstance(d, ast.Constant):
                        return str(d.value)
        bs = _bases(cls[name])
        name = bs[0] if bs else ""
    return None


# ---- success-check translation ----

class Skip(Exception):
    pass


def _ref(node: ast.AST) -> str:
    # literal object name or self.<fixture> attribute -> entity id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    raise Skip(f"dynamic_arg:{ast.unparse(node)}")


def _arg(call: ast.Call, i: int, kw: str, default: ast.AST | None = None) -> ast.AST:
    for k in call.keywords:
        if k.arg == kw:
            return k.value
    if len(call.args) > i:
        return call.args[i]
    if default is None:
        raise Skip(f"missing_arg:{kw}")
    return default


def _num(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    raise Skip(f"dynamic_threshold:{ast.unparse(node)}")


def _call(c: ast.Call) -> tuple[list[dict], list[str]]:
    f = ast.unparse(c.func)
    leaf = f.split(".")[-1]
    if f.startswith("OU.") and leaf in ("gripper_obj_far", "gripper_fxtr_far"):
        return [], [leaf]
    if f == "OU.obj_inside_of":
        return [pred("inside", _ref(_arg(c, 1, "obj_name")), _ref(_arg(c, 2, "fixture_id")))], []
    if f == "OU.check_obj_in_receptacle":
        return [pred("inside", _ref(_arg(c, 1, "obj_name")), _ref(_arg(c, 2, "receptacle_name")), tol_m=0.02)], []
    if f == "OU.check_obj_fixture_contact":
        return [pred("on", _ref(_arg(c, 1, "obj_name")), _ref(_arg(c, 2, "fixture_name")))], []
    if f == "OU.check_obj_upright":
        th = _num(_arg(c, 2, "th", ast.Constant(15)))
        return [pred("upright", _ref(_arg(c, 1, "obj_name")), tol_rad=round(math.radians(th), 4))], []
    if f == "OU.check_obj_grasped":
        return [pred("held", _ref(_arg(c, 1, "obj_name")))], []
    if leaf == "is_open" and isinstance(c.func, ast.Attribute):
        if any(k.arg in ("compartment", "reg_type", "drawer_rack_index", "joint_names") for k in c.keywords):
            raise Skip("is_open_subpart")
        th = _num(_arg(c, 99, "th", ast.Constant(0.9)))
        return [pred("open_frac", _ref(c.func.value), min=th)], []
    raise Skip(f"call:{f}")


def _expr(e: ast.AST, env: dict[str, tuple[list[dict], list[str]] | str]) -> tuple[list[dict], list[str]]:
    if isinstance(e, ast.BoolOp) and isinstance(e.op, ast.And):
        ps, dr = [], []
        for v in e.values:
            p, d = _expr(v, env)
            ps += p
            dr += d
        return ps, dr
    if isinstance(e, ast.Name):
        v = env.get(e.id, f"unbound:{e.id}")
        if isinstance(v, str):
            raise Skip(v)
        return v
    if isinstance(e, ast.Call):
        return _call(e)
    if isinstance(e, ast.Constant) and e.value is True:
        return [], []
    raise Skip(f"expr:{type(e).__name__}")


def _body(stmts: list[ast.stmt], env: dict, behavior: str | None) -> tuple[list[dict], list[str]] | None:
    for s in stmts:
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            continue  # docstring
        if isinstance(s, ast.Return):
            if s.value is None:
                raise Skip("bare_return")
            return _expr(s.value, env)
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
            try:
                env[s.targets[0].id] = _expr(s.value, env)
            except Skip as e:
                env[s.targets[0].id] = str(e)
            continue
        if isinstance(s, ast.If):
            t = ast.unparse(s.test)
            m = re.fullmatch(r"self\.behavior == ['\"](\w+)['\"]", t)
            if m is None or behavior is None:
                raise Skip("branch")
            if behavior == m.group(1):
                r = _body(s.body, env, behavior)
            else:
                r = _body(s.orelse, env, behavior) if s.orelse else None
            if r is not None:
                return r
            continue
        # any other statement poisons the names it binds
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                env[n.id] = f"stmt:{type(s).__name__}"
    return None


def translate_check(name: str, cls: dict[str, ast.ClassDef]) -> tuple[list[dict], list[str]]:
    if name in _OVERRIDES:
        return _OVERRIDES[name]
    f = _method(name, cls, "_check_success")
    if f is None:
        raise Skip("no_check")
    r = _body(f.body, {}, _behavior(name, cls))
    if r is None:
        raise Skip("no_return")
    if not r[0]:
        raise Skip("only_dropped_conjuncts")
    return r


# ---- commands ----

def _noun(e: ast.AST, behavior: str | None) -> str:
    # placeholder in an f-string instruction -> a generic noun ("{obj_lang}" -> "object")
    if "behavior" in ast.unparse(e) and behavior:
        return behavior.capitalize() if "capitalize" in ast.unparse(e) else behavior
    ids = [n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(e)
           if isinstance(n, (ast.Name, ast.Attribute))]
    for w in ids:
        w = re.sub(r"(_?nat)?_?(lang|name|str)(_?\d+)?$", "", w).replace("get_", "").replace("_", " ").strip()
        if w and w not in ("self", "ep meta", "capitalize", "lower", "upper", "format", "title"):
            return {"obj": "object", "cab": "cabinet", "fxtr": "fixture", "obj name": "object"}.get(w, w)
    return "object"


def _lang(stmts: list[ast.stmt], behavior: str | None) -> str | None:
    # the ep_meta["lang"] assignment on the branch this task's behavior takes
    for s in stmts:
        if (isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Subscript)
                and ast.unparse(s.targets[0].slice).strip("'\"") == "lang"):
            v = s.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                return v.value
            if isinstance(v, ast.JoinedStr):
                return "".join(p.value if isinstance(p, ast.Constant) else _noun(p.value, behavior)
                               for p in v.values)
        if isinstance(s, ast.If):
            m = re.fullmatch(r"self\.behavior == ['\"](\w+)['\"]", ast.unparse(s.test))
            if m is not None and behavior is not None:
                r = _lang(s.body if behavior == m.group(1) else s.orelse, behavior)
            else:
                r = _lang(s.body, behavior) or _lang(s.orelse, behavior)
            if r is not None:
                return r
    return None


def command(name: str, cls: dict[str, ast.ClassDef]) -> str:
    # the instruction RoboCasa writes into ep_meta["lang"], with placeholders made generic
    f = _method(name, cls, "get_ep_meta")
    r = _lang(f.body, _behavior(name, cls)) if f is not None else None
    r = r or re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower().capitalize() + "."
    return re.sub(r"\s+", " ", r).strip()


def _goal(ps: list[dict]) -> str:
    kinds = {d["pred"] for d in ps}
    if kinds <= {"open_frac"}:
        return "open"
    if kinds & {"inside", "on"}:
        return "pick_place" if len([d for d in ps if d["pred"] in ("inside", "on")]) == 1 else "rearrange"
    return "pick" if "held" in kinds else "manipulate"


# ---- public ----

def _missing_ctor_args(name: str, cls: dict[str, ast.ClassDef]) -> set[str]:
    # required __init__ args along the first-base chain that no subclass __init__ gives a default; robots comes
    # from robosuite.make
    given, missing, seen = {"self", "robots"}, set(), set()
    while name in cls and name not in seen:
        seen.add(name)
        for f in cls[name].body:
            if isinstance(f, ast.FunctionDef) and f.name == "__init__":
                a = f.args.args
                missing |= {x.arg for x in a[:len(a) - len(f.args.defaults)]} - given
                given |= {x.arg for x in a[len(a) - len(f.args.defaults):]}
        bs = _bases(cls[name])
        name = bs[0] if bs else ""
    return missing


def translations(src: Path | None = None) -> tuple[list[Translation], dict[str, str], list[str]]:
    # (translated, skipped name -> reason, all task names)
    src = src or source_dir()
    if src is None:
        return [], {}, []
    tr, sk, names = _translations(str(src))
    return list(tr), dict(sk), list(names)


@functools.lru_cache(maxsize=4)
def _translations(src_s: str) -> tuple[list[Translation], dict[str, str], list[str]]:
    src = Path(src_s)
    reg, cls = registry(src), classes(src)
    out, skipped = [], {}
    for name, horizon in reg.items():
        if name not in cls:
            skipped[name] = "no_class"
            continue
        if _missing_ctor_args(name, cls):
            skipped[name] = "needs_ctor_arg"  # parametrized base (OpenDoor needs fixture_id): not buildable alone
            continue
        try:
            ps, dropped = translate_check(name, cls)
        except Skip as e:
            skipped[name] = str(e)
            continue
        ents = sorted({a for d in ps for a in d["args"]})
        out.append(Translation(name, ps, ents, command(name, cls), _goal(ps), horizon / CONTROL_HZ, dropped))
    return out, skipped, list(reg)


def coverage() -> dict:
    tr, sk, names = translations()
    src = source_dir()
    return write_coverage(SOURCE, names, tr, sk, {"source_dir": str(src) if src else None})


def cards(n: int, seed: int) -> list[TaskCard]:
    tr, sk, names = translations()
    write_coverage(SOURCE, names, tr, sk, {"source_dir": str(source_dir())})
    return make_cards(SOURCE, tr, robots(), n, seed)


if __name__ == "__main__":
    import json
    c = coverage()
    print(json.dumps({k: c[k] for k in ("total", "translated", "skipped", "skip_reasons")}, indent=1))
