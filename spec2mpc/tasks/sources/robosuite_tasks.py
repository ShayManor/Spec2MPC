"""robosuite (and robomimic) manipulation tasks as training cards.

Success checks were read from robosuite master (pinned in envs/robocasa) environments/manipulation/*.py. Entity
ids are the robosuite object names; "table_top" is the arena table-top point, "bin2" the PickPlace goal-bin body,
"pot_bottom" the TwoArmLift pot bottom (pot centre site minus top offset). One-sided height checks and bin bounds
are region boxes (region convention in spec2mpc.tasks.sources). "gripper away from the object" terms
(r_reach < 0.6) and "not grasping" terms have no predicate and are dropped and counted.
"""
from __future__ import annotations

import importlib.util

from spec2mpc.core import RobotSpec, TaskCard
from spec2mpc.tasks.sources import SLAB_M, Translation, make_cards, pred, region, source_robot, write_coverage

SOURCE = "robosuite"
CONTROL_HZ = 20.0  # robosuite default control frequency
S = SLAB_M
_BIN = (0.39, 0.49)  # BinsArena table_full_size x, y: PickPlace bin size
_OBJ_BIN = {"Milk": 0, "Bread": 1, "Cereal": 2, "Can": 3}  # PickPlace object_to_id


def _bin(obj: str) -> dict:
    # PickPlace not_in_bin(): quadrant i of bin2, from the bin2 position up 0.1 m
    i = _OBJ_BIN[obj]
    x0 = -_BIN[0] / 2 if i in (0, 2) else 0.0
    y0 = -_BIN[1] / 2 if i < 2 else 0.0
    return region(obj, f"{obj.lower()}_bin", "bin2", [x0, y0, 0.0], [x0 + _BIN[0] / 2, y0 + _BIN[1] / 2, 0.1])


def _nut(nut: str, peg: str) -> list[dict]:
    # NutAssembly on_peg(): |xy - peg xy| < 0.03 and z < table + 0.05
    return [region(nut, f"{nut}_over_{peg}", peg, [-0.03, -0.03, -S], [0.03, 0.03, S]),
            region(nut, f"{nut}_low", "table_top", [-S, -S, -S], [S, S, 0.05])]


def _lifted(obj: str, dz: float) -> dict:
    return region(obj, f"{obj}_lifted", "table_top", [-S, -S, dz], [S, S, S])


# name -> (predicates, dropped conjuncts, goal type, command, horizon steps, robot) or a skip reason
TASKS: dict[str, tuple[list[dict], list[str], str, str, int, str] | str] = {
    "Lift": ([_lifted("cube", 0.04)], [], "pick", "Lift the cube off the table.", 400, "panda"),
    "Stack": ([pred("on", "cubeA", "cubeB")], ["not_grasping_cubeA"], "stack",
              "Stack the red cube on the green cube.", 400, "panda"),
    "NutAssembly": (_nut("SquareNut", "peg1") + _nut("RoundNut", "peg2"), ["gripper_away", "gripper_away"],
                    "insert", "Put each nut on its matching peg.", 800, "panda"),
    "NutAssemblySingle": "single_object_mode_unset",  # nut chosen at reset; the Square and Round variants fix it
    "NutAssemblySquare": (_nut("SquareNut", "peg1"), ["gripper_away"], "insert",
                          "Put the square nut on the square peg.", 400, "panda"),
    "NutAssemblyRound": (_nut("RoundNut", "peg2"), ["gripper_away"], "insert",
                         "Put the round nut on the round peg.", 400, "panda"),
    "PickPlace": ([_bin(o) for o in _OBJ_BIN], ["gripper_away"] * 4, "rearrange",
                  "Sort each item into its own section of the bin.", 1000, "panda"),
    "PickPlaceSingle": "single_object_mode_unset",
    "PickPlaceMilk": ([_bin("Milk")], ["gripper_away"], "pick_place", "Put the milk carton in its bin.", 400,
                      "panda"),
    "PickPlaceBread": ([_bin("Bread")], ["gripper_away"], "pick_place", "Put the bread in its bin.", 400, "panda"),
    "PickPlaceCereal": ([_bin("Cereal")], ["gripper_away"], "pick_place", "Put the cereal box in its bin.", 400,
                        "panda"),
    "PickPlaceCan": ([_bin("Can")], ["gripper_away"], "pick_place", "Put the can in its bin.", 400, "panda"),
    "Door": ([pred("open_frac", "door", min=0.75)], [], "open", "Open the door.", 500, "panda"),  # hinge > 0.3 of 0.4
    "Wipe": "marker_count",
    "ToolHang": "compound_frame_and_tool",
    "TwoArmLift": ([_lifted("pot_bottom", 0.10)], [], "carry", "Lift the pot with both arms.", 500, "two_panda"),
    "TwoArmPegInHole": "relative_axis_alignment",
    "TwoArmHandover": "per_arm_grasp",
    "TwoArmTransport": ([pred("inside", "payload", "target_bin"), pred("inside", "trash", "trash_bin")], [],
                        "rearrange", "Move the payload to the target bin and throw the trash in the trash bin.",
                        700, "two_panda"),
}


def installed() -> bool:
    return importlib.util.find_spec("robosuite") is not None


def env_names() -> list[str]:
    # registered manipulation envs when robosuite is importable, else this module's table
    if installed():
        from robosuite.environments.base import REGISTERED_ENVS
        return sorted(k for k, v in REGISTERED_ENVS.items()
                      if v.__module__.startswith("robosuite.environments.manipulation."))
    return sorted(TASKS)


def robots() -> dict[str, RobotSpec]:
    return {"panda": source_robot("arm", "robosuite_panda", "robosuite:Panda", ["ee", "gripper"], CONTROL_HZ),
            "two_panda": source_robot("arm", "robosuite_two_panda", "robosuite:Panda+Panda",
                                      ["ee0", "gripper0", "ee1", "gripper1"], CONTROL_HZ)}


def translations() -> tuple[list[Translation], dict[str, str], list[str]]:
    names = env_names()
    out, skipped = [], {}
    for n in names:
        t = TASKS.get(n, "not_in_table")
        if isinstance(t, str):
            skipped[n] = t
            continue
        ps, dropped, goal, cmd, horizon, robot = t
        ents = sorted({a for d in ps for a in d["args"]} | {d["at"] for d in ps if "at" in d})
        out.append(Translation(n, ps, ents, cmd, goal, horizon / CONTROL_HZ, list(dropped), robot))
    return out, skipped, names


def coverage() -> dict:
    tr, sk, names = translations()
    return write_coverage(SOURCE, names, tr, sk, {"registry": "installed" if installed() else "table"})


def cards(n: int, seed: int) -> list[TaskCard]:
    tr, sk, names = translations()
    write_coverage(SOURCE, names, tr, sk, {"registry": "installed" if installed() else "table"})
    return make_cards(SOURCE, tr, robots(), n, seed)


if __name__ == "__main__":
    import json
    c = coverage()
    print(json.dumps({k: c[k] for k in ("total", "translated", "skipped", "skip_reasons")}, indent=1))
