import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent
GUARDED = ("model", "critic", "search", "text", "tokenize", "baselines")
FORBIDDEN = ("spec2mpc.rubric", "spec2mpc.eval")


def guarded_files():
    for d in GUARDED:
        for p in sorted((ROOT / "spec2mpc" / d).rglob("*.py")):
            if p.relative_to(ROOT / "spec2mpc" / "baselines").as_posix() != "oracle_bo.py" if d == "baselines" else True:
                yield p


def imported_modules(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            yield from (a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            yield n.module or ""
            yield from (f"{n.module}.{a.name}" for a in n.names)


def test_guarded_modules_do_not_import_rubric_or_eval():
    for p in guarded_files():
        for m in imported_modules(ast.parse(p.read_text())):
            assert not any(m == f or m.startswith(f + ".") for f in FORBIDDEN), (p, m)


def test_no_test_card_paths_in_code():
    files = [p for d in ("spec2mpc", "scripts") for p in sorted((ROOT / d).rglob("*.py"))]
    assert files
    for p in files:
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                assert "cards/test" not in n.value, (p, n.lineno)
