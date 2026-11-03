from __future__ import annotations

import mujoco

from spec2mpc.core import RobotSpec


def actuator_names(model: mujoco.MjModel) -> list[str]:
    return [model.actuator(i).name or f"act{i}" for i in range(model.nu)]


def actuator_limits(model: mujoco.MjModel) -> dict[str, float]:
    # control and force ranges per actuator, in the actuator's own units
    out: dict[str, float] = {}
    for i, a in enumerate(actuator_names(model)):
        if model.actuator_ctrllimited[i]:
            lo, hi = model.actuator_ctrlrange[i]
            out[f"{a}_ctrl_min"], out[f"{a}_ctrl_max"] = float(lo), float(hi)
        if model.actuator_forcelimited[i]:
            out[f"{a}_force_max"] = float(abs(model.actuator_forcerange[i]).max())
    return out


def describe_robot(robot: RobotSpec, model: mujoco.MjModel | None = None) -> str:
    # RobotSpec.describe() plus links and joints, the robot text the executor reads
    if model is None:
        from spec2mpc.robots.v2.registry import build_model
        model = build_model(robot)
    links = [model.body(i).name for i in range(1, model.nbody)]
    joints = [model.joint(i).name or f"joint{i}" for i in range(model.njnt)]
    return "\n".join([robot.describe(), f"links: {', '.join(links)}", f"joints: {', '.join(joints)}"])
