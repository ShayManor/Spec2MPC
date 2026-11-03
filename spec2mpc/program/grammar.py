from __future__ import annotations

from spec2mpc.core import RobotSpec, Scene
from spec2mpc.program.validate import candidates
from spec2mpc.program.vocab import UNTIL, VOCAB, TermSpec

# JSON schema for vLLM guided decoding (xgrammar backend). Only flat keywords are used:
# type, enum, properties, required, items, prefixItems, minItems/maxItems, minimum/maximum,
# anyOf, additionalProperties false. No $ref. The decoded JSON has the Program.to_json() shape.


def _args_schema(spec: TermSpec, robot: RobotSpec, scene: Scene) -> dict | None:
    # None when some arg slot has no allowed value on this robot and scene
    slots = []
    for kind in spec.arg_kinds:
        vals = candidates(kind, robot, scene)
        if not vals:
            return None
        slots.append({"type": "string", "enum": vals})
    n = len(slots)
    out = {"type": "array", "minItems": n, "maxItems": n}
    if n:
        out["prefixItems"] = slots
    out["items"] = False
    return out


def _term_variants(table: dict[str, TermSpec], robot: RobotSpec, scene: Scene) -> list[dict]:
    out = []
    for name, spec in table.items():
        if spec.families is not None and robot.family not in spec.families:
            continue
        args = _args_schema(spec, robot, scene)
        if args is None:
            continue
        params = {"type": "object", "additionalProperties": False,
                  "properties": {k: {"type": "number", "minimum": lo, "maximum": hi}
                                 for k, (lo, hi) in spec.params.items()}}
        out.append({"type": "object", "additionalProperties": False, "required": ["type", "args", "params"],
                    "properties": {"type": {"type": "string", "enum": [name]}, "args": args, "params": params}})
    return out


def json_schema(robot: RobotSpec, scene: Scene) -> dict:
    # schema restricting term types to VOCAB for this family, pointer args to scene ids and robot frames,
    # and params to their ranges
    stage = {"type": "object", "additionalProperties": False, "required": ["terms", "until"],
             "properties": {"terms": {"type": "array", "items": {"anyOf": _term_variants(VOCAB, robot, scene)}},
                            "until": {"anyOf": _term_variants(UNTIL, robot, scene)}}}
    return {"type": "object", "additionalProperties": False, "required": ["stages"],
            "properties": {"stages": {"type": "array", "minItems": 1, "items": stage}}}
