import random
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from spec2mpc.core import Entity, Program, RobotSpec, Scene, Stage, Term
from spec2mpc.program import FAMILIES, GOTO_TERMS, UNTIL, VOCAB, json_schema, validate

ROOT = Path(__file__).parent.parent.parent


def _robot(family: str = "arm", frames: tuple[str, ...] = ("ee", "base")) -> RobotSpec:
    return RobotSpec(family=family, name=f"{family}_0", mjcf_path="x.xml", params={}, frames=list(frames),
                     actuators=["a0"], limits={}, floors={"min_clearance_m": 0.02}, runtime={}, perception={})


def _franka() -> RobotSpec:
    return _robot("arm", ("ee", "base", "link7"))


def _scene() -> Scene:
    def ent(id: str, label: str, kind: str, pts: np.ndarray | None = None) -> Entity:
        return Entity(id, label, kind, np.array([0.5, 0.0, 0.1, 1, 0, 0, 0.0]), np.array([0.05, 0.05, 0.05]),
                      pts, None)

    return Scene([ent("mug_0", "mug", "object"), ent("table_0", "table", "fixture"),
                  ent("mug_0_grasp_0", "grasp", "grasp"),
                  ent("path_0", "path", "path", np.zeros((5, 3))), ent("goal_0", "goal", "region")], None, 0.0)


def _pick() -> Program:
    return Program([
        Stage([Term("pose_to", ["ee", "mug_0_grasp_0"], {"z_offset_m": 0.1, "w": 1.0}),
               Term("gripper", ["open"], {}), Term("clearance", ["robot", "table_0"], {"margin_m": 0.03})],
              Term("dist_below", ["ee", "mug_0_grasp_0"], {"d": 0.02})),
        Stage([Term("approach_axis", ["ee", "mug_0_grasp_0", "-z"], {"dist_m": 0.1}),
               Term("limit", ["speed"], {"value": 0.3})],
              Term("at_pose", ["ee", "mug_0_grasp_0"], {"tol_m": 0.005})),
        Stage([Term("gripper", ["close"], {"force_N": 20.0})], Term("gripper_closed", [], {})),
        Stage([Term("relative_pose", ["mug_0", "ee"], {}), Term("pose_to", ["ee", "goal_0"], {"z_offset_m": 0.2}),
               Term("smooth", [], {"w": 0.1})],
              Term("timeout", [], {"t_s": 5.0})),
    ])


def _goto(family: str) -> Program:
    # a go-to goal using the first go-to term this family may use
    term = next(t for t in GOTO_TERMS if VOCAB[t].families is None or family in VOCAB[t].families)
    t = {"pose_to": Term("pose_to", ["base", "goal_0"], {}),
         "follow_path": Term("follow_path", ["base", "path_0"], {"v_ref_frac": 0.5}),
         "velocity_to": Term("velocity_to", ["base"], {"vx_mps": 1.0})}[term]
    return Program([Stage([t], Term("dist_below", ["base", "goal_0"], {"d": 0.2}))])


def _schema_ok(schema: dict, inst: dict) -> bool:
    try:
        jsonschema.Draft202012Validator(schema).validate(inst)
        return True
    except jsonschema.ValidationError:
        return False


def test_valid_pick_program_passes():
    assert validate(_pick(), _franka(), _scene()) == []


def test_gripper_on_quadruped_rejected():
    p = Program([Stage([Term("pose_to", ["base", "goal_0"], {}), Term("gripper", ["close"], {})],
                       Term("timeout", [], {"t_s": 3.0}))])
    assert validate(p, _robot("quadruped", ("base", "foot_fl")), _scene()) == ["wrong_family"]


def test_unknown_frame():
    p = Program([Stage([Term("pose_to", ["left_hand", "mug_0"], {})], Term("timeout", [], {"t_s": 3.0}))])
    assert validate(p, _franka(), _scene()) == ["unknown_frame"]


def test_param_clipped_range():
    p = Program([Stage([Term("limit", ["speed"], {"value": 5.0})], Term("timeout", [], {"t_s": 3.0}))])
    assert VOCAB["limit"].params["value"] == (0.0, 1.0)
    assert validate(p, _franka(), _scene()) == ["param_out_of_range"]


def test_other_error_codes():
    s, r = _scene(), _franka()
    until = Term("timeout", [], {"t_s": 1.0})

    def one(t: Term) -> list[str]:
        return validate(Program([Stage([t], until)]), r, s)

    assert one(Term("teleport", ["ee"], {})) == ["unknown_term"]
    assert one(Term("pose_to", ["ee", "mug_9"], {})) == ["invalid_reference"]  # entity dropped by perception
    assert one(Term("follow_path", ["ee", "mug_0"], {})) == ["invalid_reference"]  # not a path entity
    assert one(Term("pose_to", ["ee"], {})) == ["invalid_reference"]  # wrong arity
    assert one(Term("gripper", ["ajar"], {})) == ["invalid_reference"]  # unknown symbol
    assert one(Term("smooth", [], {"gain": 1.0})) == ["param_out_of_range"]  # unknown param
    assert one(Term("smooth", [], {"w": float("nan")})) == ["param_out_of_range"]
    bad_until = Program([Stage([], Term("pose_to", ["ee", "mug_0"], {}))])
    assert validate(bad_until, r, s) == ["unknown_term"]  # pose_to is a term, not a condition


def test_schema_accepts_valid_rejects_invalid():
    r, s = _franka(), _scene()
    schema = json_schema(r, s)
    jsonschema.Draft202012Validator.check_schema(schema)
    good = _pick().to_json()
    jsonschema.validate(good, schema, cls=jsonschema.Draft202012Validator)

    def with_term(t: Term) -> dict:
        return Program([Stage([t], Term("timeout", [], {"t_s": 1.0}))]).to_json()

    bad = [
        with_term(Term("teleport", ["ee"], {})),
        with_term(Term("pose_to", ["left_hand", "mug_0"], {})),
        with_term(Term("pose_to", ["ee", "mug_9"], {})),
        with_term(Term("pose_to", ["ee"], {})),
        with_term(Term("pose_to", ["ee", "mug_0", "mug_0"], {})),
        with_term(Term("limit", ["speed"], {"value": 5.0})),
        with_term(Term("smooth", [], {"gain": 1.0})),
        {"stages": []},
        {"stages": [{"terms": []}]},
    ]
    for inst in bad:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(inst, schema, cls=jsonschema.Draft202012Validator)
    # family restriction is in the schema too
    quad = json_schema(_robot("quadruped", ("base", "foot_fl")), s)
    assert not _schema_ok(quad, with_term(Term("gripper", ["close"], {})))
    assert _schema_ok(quad, with_term(Term("gait", ["trot"], {"freq_hz": 2.0})))


def test_schema_uses_flat_subset():
    # vLLM/xgrammar: no $ref, no string patterns or formats, no recursion
    allowed = {"type", "enum", "properties", "required", "items", "prefixItems", "minItems", "maxItems", "minimum",
               "maximum", "anyOf", "additionalProperties"}

    def walk(x: object) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "properties":
                    for sub in v.values():
                        walk(sub)
                    continue
                assert k in allowed, k
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(json_schema(_franka(), _scene()))


def _sample(schema: object, rng: random.Random) -> object:
    # random instance of the flat schema, stands in for constrained decoding
    if not isinstance(schema, dict):
        raise ValueError(schema)
    if "anyOf" in schema:
        return _sample(rng.choice(schema["anyOf"]), rng)
    if "enum" in schema:
        return rng.choice(schema["enum"])
    t = schema["type"]
    if t == "object":
        props = schema.get("properties", {})
        keys = [k for k in props if k in schema.get("required", []) or rng.random() < 0.5]
        return {k: _sample(props[k], rng) for k in keys}
    if t == "array":
        if "prefixItems" in schema:
            return [_sample(s, rng) for s in schema["prefixItems"]]
        if schema.get("items") is False:
            return []
        n = rng.randint(schema.get("minItems", 0), 3)
        return [_sample(schema["items"], rng) for _ in range(n)]
    if t == "number":
        return round(rng.uniform(schema["minimum"], schema["maximum"]), 6)  # xgrammar bounded floats: <= 6 decimals
    raise ValueError(t)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_schema_samples_always_validate(family: str):
    # anything the grammar admits passes validate
    r, s = _robot(family, ("base", "ee")), _scene()
    schema = json_schema(r, s)
    rng = random.Random(0)
    for _ in range(200):
        inst = _sample(schema, rng)
        assert _schema_ok(schema, inst)
        assert validate(Program.from_json(inst), r, s) == [], inst


def test_every_term_has_family_coverage():
    s = _scene()
    families_on_disk = {p.stem for p in (ROOT / "configs" / "v2" / "robots").glob("*.yaml")}
    assert families_on_disk == set(FAMILIES)
    for fam in FAMILIES:
        r = _robot(fam, ("base",))
        p = _goto(fam)
        assert validate(p, r, s) == [], fam
        assert _schema_ok(json_schema(r, s), p.to_json()), fam
    for name, spec in {**VOCAB, **UNTIL}.items():
        assert spec.families is None or spec.families <= FAMILIES, name
