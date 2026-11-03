"""MuJoCo Playground locomotion environments and terrains (quadruped and humanoid families).

Assets: <ASSET_ROOT>/playground/mujoco_playground (sparse clone of google-deepmind/mujoco_playground, locomotion
only). The terrain comes from Playground (flat plane, or the rough-terrain heightfield `hfield.png` with its
`size="10 10 .05 1.0"`); the robot is the family member from our registry (Go1 and Barkour scenes go to the
quadruped members, Spot to spot, G1 and H1 to their own, the other humanoids to g1 or h1). A joystick task tracks a
commanded base velocity for an episode; its translation is a go-to: hold one command sampled from the env's own
command ranges for a window, and reach the displaced goal without falling. Get-up, handstand, footstand, and
gait-tracking envs have no predicate translation and are skipped. No HumanoidBench content is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spec2mpc.core import Entity, Scene, TaskCard, TaskPublic
from spec2mpc.tasks.sources import Translation, pred, write_coverage
from spec2mpc.tasks.sources import _nav as N

SOURCE = "playground"
ROOT = N.ASSET_ROOT / "playground" / "mujoco_playground" / "mujoco_playground" / "_src" / "locomotion"
HFIELD = ROOT / "go1" / "xmls" / "assets" / "hfield.png"  # one heightfield shared by every rough-terrain env
HFIELD_SIZE = (10.0, 10.0, 0.05, 1.0)  # Playground rough-terrain scenes
WINDOW_S = (4.0, 8.0)  # how long the sampled command is held
MIN_GOAL_M, EDGE_MARGIN_M = 1.0, 1.5
GOAL_R_M = 0.5
SLACK = 2.0  # horizon = SLACK * window + 5 s


@dataclass(frozen=True)
class Env:
    family: str
    members: tuple[str, ...] | None  # None: every family member
    terrain: str  # flat | rough
    vx: tuple[float, float]  # command ranges, m/s (Playground default_config)
    vy: tuple[float, float]


# Playground locomotion registry (mujoco_playground/_src/locomotion/__init__.py), translated envs
ENVS: dict[str, Env] = {
    "ApolloJoystickFlatTerrain": Env("humanoid", None, "flat", (-1.5, 1.5), (-0.8, 0.8)),
    "BarkourJoystick": Env("quadruped", ("go2", "anymal_c"), "flat", (-0.6, 1.5), (-0.8, 0.8)),
    "BerkeleyHumanoidJoystickFlatTerrain": Env("humanoid", None, "flat", (-1.0, 1.0), (-1.0, 1.0)),
    "BerkeleyHumanoidJoystickRoughTerrain": Env("humanoid", None, "rough", (-1.0, 1.0), (-1.0, 1.0)),
    "G1JoystickFlatTerrain": Env("humanoid", ("g1",), "flat", (-1.0, 1.0), (-0.5, 0.5)),
    "G1JoystickRoughTerrain": Env("humanoid", ("g1",), "rough", (-1.0, 1.0), (-0.5, 0.5)),
    "Go1JoystickFlatTerrain": Env("quadruped", ("go2", "anymal_c"), "flat", (-1.5, 1.5), (-0.8, 0.8)),
    "Go1JoystickRoughTerrain": Env("quadruped", ("go2", "anymal_c"), "rough", (-1.5, 1.5), (-0.8, 0.8)),
    "Op3Joystick": Env("humanoid", None, "flat", (-0.6, 1.5), (-0.8, 0.8)),
    "SpotFlatTerrainJoystick": Env("quadruped", ("spot",), "flat", (-1.0, 1.0), (-0.8, 0.8)),
    "T1JoystickFlatTerrain": Env("humanoid", None, "flat", (-1.0, 1.0), (-0.8, 0.8)),
    "T1JoystickRoughTerrain": Env("humanoid", None, "rough", (-1.0, 1.0), (-0.8, 0.8)),
}
SKIPPED: dict[str, str] = {
    "Go1Getup": "initial_state: starts fallen; TaskPublic has no initial-state field",
    "SpotGetup": "initial_state: starts fallen; TaskPublic has no initial-state field",
    "Go1Handstand": "no_predicate: body-posture reward, no success test",
    "Go1Footstand": "no_predicate: body-posture reward, no success test",
    "H1InplaceGaitTracking": "no_predicate: contact-schedule tracking",
    "H1JoystickGaitTracking": "no_predicate: contact-schedule tracking",
    "SpotJoystickGaitTracking": "no_predicate: contact-schedule tracking",
}

STYLES = {
    "default": ["Walk over to the marked goal spot.", "Go to the goal marker and stop there.",
                "Head to the goal area without falling over."],
    "fast": ["Get to the goal marker as fast as you can without falling.",
             "Hurry over to the goal spot, but stay on your feet."],
    "careful": ["Walk carefully to the goal and keep your body steady.",
                "Make your way to the goal marker slowly and steadily, no stumbling."],
}


def available() -> bool:
    return HFIELD.exists()


def _goal(env: Env, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    # (goal xy, commanded speed, window) from one command held for a window, kept on the terrain
    lim = HFIELD_SIZE[0] - EDGE_MARGIN_M
    for _ in range(100):
        v = np.array([rng.uniform(*env.vx), rng.uniform(*env.vy)])
        win = rng.uniform(*WINDOW_S)
        d = v * win
        if MIN_GOAL_M <= np.hypot(*d) and np.all(np.abs(d) <= lim):
            return d, float(np.hypot(*v)), float(win)
    raise RuntimeError("no goal sample")


def _build(pub: TaskPublic) -> tuple[str, Scene, dict]:
    _, task, sseed = N.parse_task_id(pub.task_id)
    env = ENVS[task]
    rng = np.random.default_rng(sseed)
    goal, speed, win = _goal(env, rng)
    w = N.World(f"playground_{task}", floor_half_m=HFIELD_SIZE[0])
    floor = None
    if env.terrain == "rough":
        w.assets.append(f'<hfield name="hfield" file="{HFIELD}" size="{" ".join(map(str, HFIELD_SIZE))}"/>')
        floor = '<geom name="floor" type="hfield" hfield="hfield" rgba="0.4 0.35 0.3 1"/>'
    terrain = Entity("terrain", f"{env.terrain} terrain", "fixture", N.pose7([0, 0, 0]),
                     np.array([HFIELD_SIZE[0], HFIELD_SIZE[1], HFIELD_SIZE[2] if env.terrain == "rough" else 0.0]),
                     None, None)
    pts = np.c_[np.linspace([0, 0], goal, 10), np.zeros(10)]
    ents = [N.start([0, 0, 0], 0.0), N.region("goal", N.circle(goal, GOAL_R_M)), N.path("route", pts), terrain]
    return w.xml(floor), N.scene_of(ents), {"dist_m": float(np.hypot(*goal)), "speed": speed, "window_s": win}


def scene(card: TaskCard | TaskPublic) -> tuple[str, Scene]:
    """MJCF world xml (no robot) and the Scene for a playground card."""
    xml, sc, _ = _build(card.public if isinstance(card, TaskCard) else card)
    return xml, sc


def _test(info: dict, style: str) -> tuple[list[dict], list[dict], float]:
    win = info["window_s"]
    preds = [pred("in_region", "base", "goal"), pred("not_fallen", "robot")]
    graded = [{"term": "time", "args": [], "good": win, "bad": SLACK * win + 5.0}]
    if style == "fast":
        preds.append(pred("time_below", t_s=round(1.5 * win, 2)))
    if style == "careful":
        graded = [{"term": "tilt", "args": ["base"], "good": 0.1, "bad": 0.5},
                  {"term": "jerk", "args": ["base"], "good": 5.0, "bad": 50.0}]
    return preds, graded, SLACK * win + 5.0


def translations() -> list[Translation]:
    preds = [pred("in_region", "base", "goal"), pred("not_fallen", "robot")]
    return [Translation(k, preds, ["base", "goal"], STYLES["default"][0], "walk_to", SLACK * WINDOW_S[1] + 5.0,
                        ["velocity_tracking_reward: approximated by reaching the displaced goal"])
            for k in ENVS]


def coverage() -> dict:
    return write_coverage(SOURCE, sorted(ENVS) + sorted(SKIPPED), translations(), SKIPPED,
                          {"terrains": ["flat", "rough"]})


def cards(n: int, seed: int) -> list[TaskCard]:
    """n go-to cards cycling through the translated Playground envs, goals sampled from their command ranges."""
    if not available():
        raise RuntimeError(f"MuJoCo Playground assets missing under {ROOT}")
    rng = np.random.default_rng(seed)
    names = sorted(ENVS)
    order = rng.permutation(len(names))
    coverage()
    out = []
    for j in range(n):
        task = names[int(order[j % len(names)])]
        env = ENVS[task]
        style = N.pick(rng, sorted(STYLES))
        sseed = int(rng.integers(0, 2**31 - 1))
        rb = N.robot(env.family, N.pick(rng, list(env.members or N.members(env.family))), sseed)
        pub = TaskPublic(N.task_id(SOURCE, task, sseed), rb.family, rb, sseed, "", None)
        _, _, info = _build(pub)
        preds, graded, horizon = _test(info, style)
        out.append(N.card(SOURCE, task, sseed, rb, N.pick(rng, STYLES[style]), "walk_to", preds, graded, horizon))
    return out
