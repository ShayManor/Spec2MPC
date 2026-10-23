from dataclasses import dataclass, fields, replace
from typing import Union

from torch import Tensor

DONE_TIMEOUT = 0
DONE_COMPLETED = 1
DONE_FAILURE = 2

_STEP_FIELDS = (
    "t_s", "x", "u", "s", "a_long", "a_lat", "clearance_true", "solver_ok", "valid", "p_drive_W",
)


@dataclass
class Trajectory:
    t_s: Tensor  # [B, T]
    x: Tensor  # [B, T, 8] X, Y, psi, vx, vy, omega, delta, F
    u: Tensor  # [B, T, 2] delta_rate, F_rate
    s: Tensor  # [B, T] unwrapped progress, m
    a_long: Tensor  # [B, T] m/s^2
    a_lat: Tensor  # [B, T] m/s^2
    clearance_true: Tensor  # [B, T] m
    solver_ok: Tensor  # [B, T] bool
    valid: Tensor  # [B, T] bool, False after termination
    p_drive_W: Tensor  # [B, T]
    done_reason: Tensor  # [B] long
    completion_time_s: Tensor  # [B] NaN if not completed
    laps_done: Tensor  # [B] long
    dt_s: float

    def index(self, idx: Union[Tensor, slice]) -> "Trajectory":
        # idx over B axis
        return replace(
            self,
            **{f.name: getattr(self, f.name)[idx] for f in fields(self) if f.name != "dt_s"},
        )


def downsample(traj: Trajectory, hz: float) -> Trajectory:
    # [B, T, ...] -> [B, ceil(T/k), ...], k = round(1/(hz*dt_s)) >= 1
    k = max(1, round(1.0 / (hz * traj.dt_s)))
    return replace(
        traj,
        dt_s=traj.dt_s * k,
        **{n: getattr(traj, n)[:, ::k] for n in _STEP_FIELDS},
    )
