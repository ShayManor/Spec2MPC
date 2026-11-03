"""Executor prompt: chat messages built from the command, the robot description, and the scene entity list.

Only public inputs enter the prompt (command, RobotSpec, Scene, feedback). Task tests never reach this module.
"""
from __future__ import annotations

import json

import numpy as np

from spec2mpc.core import Entity, Program, RobotSpec, Scene

SYSTEM = (
    "You are the executor of a robot. You turn one command into an MPC program for this robot. "
    "A program is a list of stages run in order; each stage holds cost and constraint terms from a fixed "
    "vocabulary and ends when its until condition fires. Term arguments point at scene entity ids, robot "
    "frames, or listed symbols. Parameters are optional; omitted ones take the controller defaults. "
    "Safety floors are enforced by the controller and cannot be changed by the program.\n"
    "First reason briefly inside <think> ... </think>: ground the command to entities, pick the stages, "
    "and pick terms and limits that suit this robot. Then output only the program as JSON of the form "
    '{"stages": [{"terms": [{"type": ..., "args": [...], "params": {...}}], '
    '"until": {"type": ..., "args": [...], "params": {...}}}]}.'
)

MAX_PATH_POINTS = 20  # path waypoints shown per path entity


def _f(x: np.ndarray, nd: int = 3) -> str:
    return "(" + ", ".join(f"{float(v):.{nd}f}" for v in np.asarray(x).ravel()) + ")"


def describe_entity(e: Entity) -> str:
    # one line per entity; pose is xyz + quaternion wxyz in the world frame
    s = f"- {e.id} [{e.kind}] label={e.label!r} pos={_f(e.pose[:3])} quat={_f(e.pose[3:])}"
    if e.extent is not None:
        s += f" half_extent={_f(e.extent)}"
    if e.parent is not None:
        s += f" parent={e.parent}"
    if e.points is not None and len(e.points):
        if e.kind == "path":
            idx = np.unique(np.linspace(0, len(e.points) - 1, min(len(e.points), MAX_PATH_POINTS)).astype(int))
            s += " waypoints=[" + ", ".join(_f(e.points[i], 2) for i in idx) + "]"
        else:
            s += f" n_points={len(e.points)}"
    return s


def describe_scene(scene: Scene) -> str:
    return "\n".join([f"scene entities at t={scene.t_s:.2f} s:"] + [describe_entity(e) for e in scene.entities])


def vocab_doc(robot: RobotSpec | None = None) -> str:
    # vocabulary documentation from Task 1's VOCAB and UNTIL, restricted to the robot's family when given
    from spec2mpc.program import SYMBOLS, UNTIL, VOCAB

    def rows(table: dict) -> list[str]:
        out = []
        for name, spec in table.items():
            if robot is not None and spec.families is not None and robot.family not in spec.families:
                continue
            ps = ", ".join(f"{k} in [{lo:g}, {hi:g}]" for k, (lo, hi) in spec.params.items()) or "none"
            out.append(f"- {name}({', '.join(spec.arg_kinds)}); params: {ps}")
        return out

    kinds = ["argument kinds: frame = a robot frame; pointer = a scene entity id or robot frame; "
             "path = a scene entity of kind path; body = a robot frame or 'robot'; "
             "cloud = a scene entity id or 'scene'"]
    kinds += [f"{k}: {' | '.join(v)}" for k, v in SYMBOLS.items()]
    return "\n".join(["terms:"] + rows(VOCAB) + ["until conditions:"] + rows(UNTIL) + kinds)


# three hand-written generic examples (not drawn from any task suite); valid under Task 1's validate
EXAMPLES: list[dict] = [
    {"robot": "arm (frames: ee, base)",
     "entities": ["mug_1 [object] label='red mug'", "mug_1_grasp_0 [grasp] parent=mug_1",
                  "table_1 [fixture] label='table'"],
     "command": "pick up the red mug",
     "program": "STAGE\n  TERM pose_to(ee,mug_1_grasp_0; z_offset_m=0.1)\n  TERM gripper(open)\n"
                "  UNTIL dist_below(ee,mug_1_grasp_0; d=0.12)\nEND\n"
                "STAGE\n  TERM approach_axis(ee,mug_1_grasp_0,-z; dist_m=0.1)\n  TERM limit(speed; value=0.3)\n"
                "  UNTIL dist_below(ee,mug_1_grasp_0; d=0.01)\nEND\n"
                "STAGE\n  TERM gripper(close; force_N=20.0)\n  UNTIL gripper_closed()\nEND\n"
                "STAGE\n  TERM pose_to(ee,table_1; z_offset_m=0.3)\n  TERM smooth(; w=1.0)\n"
                "  UNTIL timeout(; t_s=3.0)\nEND"},
    {"robot": "mobile_base (frames: base)",
     "entities": ["dock_1 [fixture] label='charging dock'", "box_1 [object] label='box'"],
     "command": "drive to the charging dock, carefully",
     "program": "STAGE\n  TERM pose_to(base,dock_1)\n  TERM clearance(robot,scene; margin_m=0.3)\n"
                "  TERM limit(speed; value=0.4)\n  TERM smooth(; w=5.0)\n"
                "  UNTIL dist_below(base,dock_1; d=0.1)\nEND"},
    {"robot": "quadruped (frames: base, foot_fl, foot_fr, foot_rl, foot_rr)",
     "entities": ["cone_1 [object] label='orange cone'"],
     "command": "trot over to the orange cone",
     "program": "STAGE\n  TERM gait(trot; freq_hz=2.0)\n  TERM balance(; margin_m=0.05)\n"
                "  TERM pose_to(base,cone_1)\n  TERM clearance(robot,scene; margin_m=0.2)\n"
                "  UNTIL dist_below(base,cone_1; d=0.3)\nEND"},
]


def examples_text(n: int = 3) -> str:
    out = []
    for i, ex in enumerate(EXAMPLES[:n]):
        prog = json.dumps(Program.from_text(ex["program"]).to_json(), separators=(",", ":"))
        out.append(f"example {i + 1}\nrobot: {ex['robot']}\nentities: {'; '.join(ex['entities'])}\n"
                   f"command: {ex['command']}\noutput: {prog}")
    return "\n\n".join(out)


def build_prompt(command: str, robot: RobotSpec, scene: Scene, feedback: str | None, *,
                 docs: bool = False, n_examples: int = 0) -> list[dict]:
    """Chat messages; the camera image is attached as the first user content part when present."""
    system = SYSTEM
    if docs:
        system += "\n\nvocabulary\n" + vocab_doc(robot)
    if n_examples:
        system += "\n\nexamples\n" + examples_text(n_examples)
    text = f"{robot.describe()}\n\n{describe_scene(scene)}\n\ncommand: {command}"
    if feedback:
        text += f"\n\nfeedback from execution so far:\n{feedback}"
    user: list[dict] = []
    if scene.image is not None:
        from PIL import Image

        user.append({"type": "image", "image": Image.fromarray(np.asarray(scene.image, dtype=np.uint8))})
    user.append({"type": "text", "text": text})
    return [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": user}]


def prompt_text(messages: list[dict]) -> str:
    # all text parts joined (for checks and text-only tokenizers)
    return "\n\n".join(p["text"] for m in messages for p in m["content"] if p["type"] == "text")
