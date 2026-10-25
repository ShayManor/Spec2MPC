from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.batch import Batched


@dataclass
class Conditions(Batched):
    friction_scale: Tensor  # [B]
    mass_scale: Tensor  # [B]
    latency_s: Tensor  # [B] extra plant input delay, added to actuator_delay_s
    init_offset: Tensor  # [B, 3] lateral m, heading rad, speed m/s
    seed: Tensor  # [B] int64


def sample_conditions(n: int, cfg: DictConfig, gen: torch.Generator) -> Conditions:
    # cfg: the sim config (reads cfg.conditions) -> Conditions with batch n, uniform within each range
    c = cfg.conditions

    def u(rng) -> Tensor:
        lo, hi = float(rng[0]), float(rng[1])
        return lo + (hi - lo) * torch.rand(n, generator=gen, dtype=torch.float32)

    friction, mass, latency = u(c.friction_scale), u(c.mass_scale), u(c.latency_s)
    offset = torch.stack([u(c.init_lateral_m), u(c.init_heading_rad), u(c.init_speed_mps)], dim=1)
    seed = torch.randint(0, 2**62, (n,), generator=gen, dtype=torch.int64)
    return Conditions(friction_scale=friction, mass_scale=mass, latency_s=latency, init_offset=offset, seed=seed)
