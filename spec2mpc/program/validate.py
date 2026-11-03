from __future__ import annotations

from spec2mpc.core import Program, RobotSpec, Scene, Term
from spec2mpc.program.vocab import SYMBOLS, UNTIL, VOCAB, TermSpec


def candidates(kind: str, robot: RobotSpec, scene: Scene) -> list[str]:
    # allowed values for one argument kind on this robot and scene
    ids = [e.id for e in scene.entities]
    if kind == "frame":
        return list(robot.frames)
    if kind == "pointer":
        return ids + [f for f in robot.frames if f not in ids]
    if kind == "path":
        return [e.id for e in scene.entities if e.kind == "path"]
    if kind == "body":
        return list(robot.frames) + ["robot"]
    if kind == "cloud":
        return ids + ["scene"]
    return list(SYMBOLS[kind])


def _check_term(t: Term, table: dict[str, TermSpec], robot: RobotSpec, scene: Scene) -> str | None:
    # first error code for one term, or None
    spec = table.get(t.type)
    if spec is None:
        return "unknown_term"
    if spec.families is not None and robot.family not in spec.families:
        return "wrong_family"
    if len(t.args) != len(spec.arg_kinds):
        return "invalid_reference"
    for a, kind in zip(t.args, spec.arg_kinds):
        if a in candidates(kind, robot, scene):
            continue
        # a frame slot naming a frame this robot lacks is unknown_frame; everything else is a bad pointer
        return "unknown_frame" if kind == "frame" else "invalid_reference"
    for k, v in t.params.items():
        lo, hi = spec.params.get(k, (float("nan"), float("nan")))  # unknown param names fail the range test
        if not lo <= v <= hi:
            return "param_out_of_range"
    return None


def validate(program: Program, robot: RobotSpec, scene: Scene) -> list[str]:
    # error codes, at most one per term (empty list means valid):
    # unknown_term, unknown_frame, invalid_reference, param_out_of_range, wrong_family
    errs = []
    for st in program.stages:
        for t in st.terms:
            if (e := _check_term(t, VOCAB, robot, scene)) is not None:
                errs.append(e)
        if (e := _check_term(st.until, UNTIL, robot, scene)) is not None:
            errs.append(e)
    return errs
