"""Robust per-feature scaling for heavy-tailed episode features: z = asinh((phi - median) / (IQR / 1.349))."""
from typing import Tuple

import torch
from torch import Tensor

SCALE_FLOOR = 1e-3  # raw phi units; keeps z finite for features constant on train
IQR_TO_STD = 1.349  # IQR of a unit normal


def robust_stats(phi: Tensor) -> Tuple[Tensor, Tensor]:
    # phi [E, D] raw -> center [D] (median), scale [D] (IQR / 1.349, floored)
    q = phi.quantile(torch.tensor([0.25, 0.5, 0.75], dtype=phi.dtype, device=phi.device), dim=0)  # [3, D]
    return q[1], ((q[2] - q[0]) / IQR_TO_STD).clamp_min(SCALE_FLOOR)


def robust_z(phi: Tensor, center: Tensor, scale: Tensor) -> Tensor:
    # phi [..., D] raw -> z [..., D]
    return torch.asinh((phi - center) / scale)


def robust_inv(z: Tensor, center: Tensor, scale: Tensor) -> Tensor:
    # z [..., D] -> phi [..., D] raw, exact inverse of robust_z
    return center + scale * torch.sinh(z)
