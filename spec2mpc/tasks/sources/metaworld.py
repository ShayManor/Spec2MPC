"""Meta-World MT50 (v3 envs, metaworld 3.0.0) as training cards.

Every Meta-World success check compares one distance against a threshold. Entity ids: "obj" is the task's main
object point (obs[4:7]: puck, handle, button, lid, ...), "goal" is the env's _target_pos, "ee" the gripper tcp,
"obj2" the second object (obs[11:14]) in the stick tasks.
  3-D distance |obj - goal| <= r          -> at_pose(obj, goal; tol_m=r)
  1-D distance |obj[k] - goal[k]| <= r    -> in_region(obj, goal_slab): box at goal, +-r along k, +-SLAB_M else
  one-sided obj[2] > goal[2]              -> in_region(obj, above_goal): box at goal from 0 to SLAB_M in z
Scaled norms (an ellipsoid, not a sphere or box), angles, joint positions, and checks combined with a hidden
insertion test are skipped. Thresholds were read from metaworld 3.0.0 envs/sawyer_*_v3.py.
"""
from __future__ import annotations

import importlib.util

from spec2mpc.core import RobotSpec, TaskCard
from spec2mpc.tasks.sources import SLAB_M, Translation, make_cards, pred, region, source_robot, write_coverage

SOURCE = "metaworld"
STEP_S = 0.0125  # frame_skip 5 x timestep 0.0025 s
MAX_PATH_LENGTH = 500
_AX = {"x": 0, "y": 1, "z": 2}

# name -> (check, entity, threshold m, goal type, command); check "at" | "slab:<axis>" | "above" | "stick" | "skip:..."
TASKS: dict[str, tuple[str, str, float, str, str]] = {
    "assembly-v3": ("skip:compound", "", 0, "insert", "Put the ring onto the peg."),
    "basketball-v3": ("skip:scaled_norm", "", 0, "pick_place", "Dunk the ball into the basket."),
    "bin-picking-v3": ("at", "obj", 0.05, "pick_place", "Move the cube from one bin to the other bin."),
    "box-close-v3": ("at", "obj", 0.08, "pick_place", "Put the lid on the box."),
    "button-press-topdown-v3": ("slab:z", "obj", 0.024, "push", "Press the button from the top."),
    "button-press-topdown-wall-v3": ("slab:z", "obj", 0.024, "push", "Press the button from the top, past the wall."),
    "button-press-v3": ("slab:y", "obj", 0.02, "push", "Press the button."),
    "button-press-wall-v3": ("slab:y", "obj", 0.03, "push", "Press the button behind the wall."),
    "coffee-button-v3": ("slab:y", "obj", 0.02, "push", "Push the button on the coffee machine."),
    "coffee-pull-v3": ("skip:scaled_norm", "", 0, "pull", "Pull the mug out from the coffee machine."),
    "coffee-push-v3": ("skip:scaled_norm", "", 0, "push", "Push the mug under the coffee machine."),
    "dial-turn-v3": ("at", "obj", 0.07, "turn", "Rotate the dial."),
    "disassemble-v3": ("above", "obj", 0, "pick", "Pick the nut off the peg."),
    "door-close-v3": ("at", "obj", 0.08, "close", "Close the door."),
    "door-lock-v3": ("slab:z", "obj", 0.02, "turn", "Lock the door by turning the lock down."),
    "door-open-v3": ("slab:x", "obj", 0.08, "open", "Open the door."),
    "door-unlock-v3": ("slab:x", "obj", 0.02, "turn", "Unlock the door by turning the lock."),
    "drawer-close-v3": ("at", "obj", 0.055, "close", "Push the drawer closed."),
    "drawer-open-v3": ("at", "obj", 0.03, "open", "Open the drawer."),
    "faucet-close-v3": ("at", "obj", 0.07, "turn", "Turn the faucet handle to close it."),
    "faucet-open-v3": ("at", "obj", 0.07, "turn", "Turn the faucet handle to open it."),
    "hammer-v3": ("skip:joint_position", "", 0, "tool_use", "Hammer the nail into the wall."),
    "hand-insert-v3": ("at", "obj", 0.05, "insert", "Insert the object into the hole."),
    "handle-press-side-v3": ("slab:z", "obj", 0.02, "push", "Press the handle down from the side."),
    "handle-press-v3": ("slab:z", "obj", 0.02, "push", "Press the handle down."),
    "handle-pull-side-v3": ("skip:scaled_norm", "", 0, "pull", "Pull the handle up from the side."),
    "handle-pull-v3": ("slab:z", "obj", 0.05, "pull", "Pull the handle up."),
    "lever-pull-v3": ("skip:angle", "", 0, "pull", "Pull the lever up."),
    "peg-insert-side-v3": ("skip:scaled_norm", "", 0, "insert", "Insert the peg into the hole from the side."),
    "peg-unplug-side-v3": ("at", "obj", 0.07, "pull", "Unplug the peg from the side."),
    "pick-out-of-hole-v3": ("at", "obj", 0.07, "pick_place", "Pick the object out of the hole."),
    "pick-place-v3": ("at", "obj", 0.07, "pick_place", "Pick up the puck and place it at the goal."),
    "pick-place-wall-v3": ("at", "obj", 0.07, "pick_place", "Pick up the puck and place it at the goal past the wall."),
    "plate-slide-back-side-v3": ("at", "obj", 0.07, "slide", "Slide the plate back into the cabinet from the side."),
    "plate-slide-back-v3": ("at", "obj", 0.07, "slide", "Slide the plate back into the cabinet."),
    "plate-slide-side-v3": ("at", "obj", 0.07, "slide", "Slide the plate into the cabinet from the side."),
    "plate-slide-v3": ("at", "obj", 0.07, "slide", "Slide the plate into the cabinet."),
    "push-back-v3": ("at", "obj", 0.07, "push", "Push the puck back to the goal."),
    "push-v3": ("at", "obj", 0.05, "push", "Push the puck to the goal."),
    "push-wall-v3": ("at", "obj", 0.07, "push", "Push the puck to the goal past the wall."),
    "reach-v3": ("at", "ee", 0.05, "reach", "Reach the goal position."),
    "reach-wall-v3": ("at", "ee", 0.05, "reach", "Reach the goal position past the wall."),
    "shelf-place-v3": ("at", "obj", 0.07, "pick_place", "Pick up the puck and place it on the shelf."),
    "soccer-v3": ("skip:scaled_norm", "", 0, "push", "Kick the ball into the goal."),
    "stick-pull-v3": ("skip:compound", "", 0, "tool_use", "Grasp the stick and pull the box with it."),
    "stick-push-v3": ("stick", "obj2", 0.12, "tool_use", "Grasp the stick and push the box with it."),
    "sweep-into-v3": ("at", "obj", 0.05, "push", "Sweep the puck into the hole."),
    "sweep-v3": ("at", "obj", 0.05, "push", "Sweep the puck off the table."),
    "window-close-v3": ("slab:x", "obj", 0.05, "slide", "Push the window closed."),
    "window-open-v3": ("slab:x", "obj", 0.05, "slide", "Push the window open."),
}


def installed() -> bool:
    return importlib.util.find_spec("metaworld") is not None


def env_names() -> list[str]:
    # the installed MT50 list when metaworld is importable, else this module's table
    if installed():
        import metaworld
        return sorted(metaworld.ALL_V3_ENVIRONMENTS)
    return sorted(TASKS)


def robots() -> dict[str, RobotSpec]:
    return {"default": source_robot("arm", "metaworld_sawyer", "metaworld:sawyer_xyz_base", ["ee", "gripper"],
                                    1.0 / STEP_S)}


def translate(name: str) -> Translation:
    if name not in TASKS:
        raise KeyError(name)
    check, ent, r, goal, cmd = TASKS[name]
    if check.startswith("skip:"):
        raise ValueError(check[5:])
    dropped: list[str] = []
    if check == "at":
        ps = [pred("at_pose", ent, "goal", tol_m=r)]
    elif check.startswith("slab:"):
        hi = [SLAB_M] * 3
        hi[_AX[check[5:]]] = r
        ps = [region(ent, "goal_slab", "goal", [-h for h in hi], hi)]
    elif check == "above":
        ps = [region(ent, "above_goal", "goal", [-SLAB_M, -SLAB_M, 0.0], [SLAB_M, SLAB_M, SLAB_M])]
    else:  # stick: box at goal and the stick grasped ("lifted above its start" has no predicate)
        ps = [pred("at_pose", ent, "goal", tol_m=r), pred("held", "obj")]
        dropped = ["stick_lifted"]
    ents = sorted({a for d in ps for a in d["args"]} | {d["at"] for d in ps if "at" in d})
    return Translation(name, ps, ents, cmd, goal, MAX_PATH_LENGTH * STEP_S, dropped)


def translations() -> tuple[list[Translation], dict[str, str], list[str]]:
    names = env_names()
    out, skipped = [], {}
    for n in names:
        try:
            out.append(translate(n))
        except KeyError:
            skipped[n] = "not_in_table"
        except ValueError as e:
            skipped[n] = str(e)
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
