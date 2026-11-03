from __future__ import annotations

from dataclasses import dataclass

# robot families in configs/v2/robots/
FAMILIES = frozenset({"arm", "mobile_base", "mobile_manipulator", "rover2d", "kart", "quadruped", "humanoid",
                      "quadrotor"})
ARM_LIKE = frozenset({"arm", "mobile_manipulator"})
LEGGED = frozenset({"quadruped", "humanoid"})
MOBILE = frozenset({"mobile_base", "mobile_manipulator", "rover2d", "kart"})

# argument kinds:
#   frame    a robot frame (robot.frames)
#   pointer  a scene entity id or a robot frame
#   path     a scene entity of kind "path"
#   body     a robot frame, or "robot" for the whole robot
#   cloud    a scene entity id, or "scene" for every entity
#   any other kind is symbolic, with its allowed values in SYMBOLS
SYMBOLS: dict[str, list[str]] = {
    "gripper_state": ["open", "close", "hold"],
    "axis": ["x", "y", "z", "-x", "-y", "-z"],
    "quantity": ["speed", "accel", "jerk", "effort", "force", "tilt"],
    "gait_pattern": ["stand", "walk", "trot", "pace", "bound"],
}


@dataclass(frozen=True)
class TermSpec:
    arg_kinds: list[str]  # one kind per positional arg
    params: dict[str, tuple[float, float]]  # name -> (lo, hi); every param is optional
    families: set[str] | None  # None: every family


_W = (0.0, 100.0)  # cost weight
_OFF = (-2.0, 2.0)  # target offset, m
_V = (-30.0, 30.0)  # linear velocity, m/s
_F = (0.0, 200.0)  # force, N

# Spec Section 5.3 table
VOCAB: dict[str, TermSpec] = {
    "pose_to": TermSpec(["frame", "pointer"], {"x_offset_m": _OFF, "y_offset_m": _OFF, "z_offset_m": _OFF,
                                               "w": _W, "w_pos": _W, "w_rot": _W}, None),
    "relative_pose": TermSpec(["pointer", "pointer"], {"w": _W, "w_pos": _W, "w_rot": _W}, None),
    "approach_axis": TermSpec(["frame", "pointer", "axis"], {"dist_m": (0.0, 1.0), "w": _W}, None),
    "follow_path": TermSpec(["frame", "path"], {"w_progress": _W, "w_contour": _W, "w_lag": _W,
                                                "v_ref_frac": (0.0, 1.0)}, None),
    "velocity_to": TermSpec(["frame"], {"vx_mps": _V, "vy_mps": _V, "vz_mps": _V, "wz_radps": (-10.0, 10.0),
                                        "w": _W}, set(MOBILE | LEGGED | {"quadrotor"})),
    "contact": TermSpec(["frame", "pointer"], {"force_N": _F, "w": _W}, None),
    "gripper": TermSpec(["gripper_state"], {"force_N": _F}, set(ARM_LIKE)),
    "gait": TermSpec(["gait_pattern"], {"freq_hz": (0.5, 4.0), "w": _W}, set(LEGGED)),
    "balance": TermSpec([], {"margin_m": (0.0, 0.3), "w": _W}, set(LEGGED)),
    "clearance": TermSpec(["body", "cloud"], {"margin_m": (0.0, 1.0), "w": _W}, None),
    "limit": TermSpec(["quantity"], {"value": (0.0, 1.0)}, None),  # value is a fraction of the robot limit
    "smooth": TermSpec([], {"w": _W}, None),
}

# stage transition conditions (UNTIL ...)
UNTIL: dict[str, TermSpec] = {
    "contact": TermSpec(["pointer", "pointer"], {"force_N": _F}, None),
    "dist_below": TermSpec(["pointer", "pointer"], {"d": (0.0, 10.0)}, None),
    "gripper_closed": TermSpec([], {}, set(ARM_LIKE)),
    "timeout": TermSpec([], {"t_s": (0.0, 600.0)}, None),
    "at_pose": TermSpec(["pointer", "pointer"], {"tol_m": (0.0, 1.0), "tol_rad": (0.0, 3.2)}, None),
}

# terms that alone can express "go to a goal"
GOTO_TERMS = ("pose_to", "follow_path", "velocity_to")
