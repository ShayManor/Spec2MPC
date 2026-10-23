import hashlib
from dataclasses import dataclass
from typing import List, Union

import torch
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.batch import Batched


@dataclass
class KartParams(Batched):
    mass_kg: Tensor
    yaw_inertia_kgm2: Tensor
    wheelbase_m: Tensor
    lf_m: Tensor
    lr_m: Tensor
    cf_Npr: Tensor
    cr_Npr: Tensor
    pac_B: Tensor
    pac_C: Tensor
    pac_E: Tensor
    mu_nominal: Tensor
    max_steer_rad: Tensor
    max_steer_rate_radps: Tensor
    max_drive_force_N: Tensor
    max_brake_force_N: Tensor
    max_force_rate_Nps: Tensor
    actuator_delay_s: Tensor
    v_max_mps: Tensor
    a_lat_max_mps2: Tensor
    a_long_max_mps2: Tensor
    width_m: Tensor
    length_m: Tensor
    c_rr: Tensor
    c_drag: Tensor

    @property
    def radius_m(self) -> Tensor:
        # [B]
        return 0.5 * torch.hypot(self.width_m, self.length_m)

    @property
    def min_turn_radius_m(self) -> Tensor:
        # [B]
        return self.wheelbase_m / torch.tan(self.max_steer_rad)


@dataclass
class PerceptionProfile(Batched):
    range_m: Tensor
    fov_rad: Tensor
    loc_noise_std_m: Tensor
    loc_bias_m: Tensor
    latency_s: Tensor
    p_false_occ: Tensor
    p_false_free: Tensor
    pred_error_m: Tensor
    has_prediction: Tensor  # bool


@dataclass
class RuntimeConfig:
    control_rate_hz: float
    solver_iters: int
    perception_rate_hz: float


@dataclass
class Robot:
    params: KartParams
    perception: PerceptionProfile
    runtime: RuntimeConfig
    robot_id: List[str]

    def index(self, idx: Union[Tensor, slice]) -> "Robot":
        # idx: slice | long tensor [M] | bool mask [B] -> Robot with batch M
        if isinstance(idx, slice):
            ids = self.robot_id[idx]
        else:
            sel = idx.nonzero().flatten() if idx.dtype == torch.bool else idx
            ids = [self.robot_id[int(i)] for i in sel]
        return Robot(self.params.index(idx), self.perception.index(idx), self.runtime, ids)

    def to(self, device: Union[str, torch.device]) -> "Robot":
        # every tensor field moved to device; shapes unchanged
        return Robot(self.params.to(device), self.perception.to(device), self.runtime, self.robot_id)


def _uniform(rng, n: int, gen: torch.Generator) -> Tensor:
    # -> [n]
    lo, hi = float(rng[0]), float(rng[1])
    return lo + (hi - lo) * torch.rand(n, generator=gen, dtype=torch.float32)


def _build_params(v: dict, g: float) -> KartParams:
    # v: name -> [B] tensors for every sampled range
    L = v["wheelbase_m"]
    lf = v["lf_frac"] * L
    return KartParams(
        mass_kg=v["mass_kg"], yaw_inertia_kgm2=v["yaw_inertia_kgm2"], wheelbase_m=L,
        lf_m=lf, lr_m=L - lf, cf_Npr=v["cf_Npr"], cr_Npr=v["cr_Npr"],
        pac_B=v["pac_B"], pac_C=v["pac_C"], pac_E=v["pac_E"], mu_nominal=v["mu_nominal"],
        max_steer_rad=v["max_steer_rad"], max_steer_rate_radps=v["max_steer_rate_radps"],
        max_drive_force_N=v["max_drive_force_N"], max_brake_force_N=v["max_brake_force_N"],
        max_force_rate_Nps=v["max_force_rate_Nps"], actuator_delay_s=v["actuator_delay_s"],
        v_max_mps=v["v_max_mps"],
        a_lat_max_mps2=v["mu_nominal"] * g,
        a_long_max_mps2=v["max_drive_force_N"] / v["mass_kg"],
        width_m=v["width_m"], length_m=v["length_m"], c_rr=v["c_rr"], c_drag=v["c_drag"],
    )


def sample_runtime(cfg: DictConfig, gen: torch.Generator) -> RuntimeConfig:
    def pick(options):
        return options[int(torch.randint(len(options), (1,), generator=gen))]

    return RuntimeConfig(
        control_rate_hz=float(pick(cfg.runtime.control_rate_hz)),
        solver_iters=int(pick(cfg.runtime.solver_iters)),
        perception_rate_hz=float(pick(cfg.runtime.perception_rate_hz)),
    )


def sample_karts(n: int, cfg: DictConfig, runtime: RuntimeConfig, gen: torch.Generator) -> Robot:
    # returns Robot with params/perception fields [n]
    draw = int(torch.randint(0, 2**62, (1,), generator=gen))
    seed_hash = hashlib.sha1(draw.to_bytes(8, "little")).hexdigest()[:8]
    v = {k: _uniform(r, n, gen) for k, r in cfg.ranges.items()}
    p = {k: _uniform(r, n, gen) for k, r in cfg.perception.items() if k != "p_has_prediction"}
    has_pred = torch.rand(n, generator=gen) < float(cfg.perception.p_has_prediction)
    return Robot(
        params=_build_params(v, float(cfg.gravity_mps2)),
        perception=PerceptionProfile(**p, has_prediction=has_pred),
        runtime=runtime,
        robot_id=[f"kart_{seed_hash}_{i}" for i in range(n)],
    )


def load_fixed_kart(cfg: DictConfig, name: str, runtime: RuntimeConfig) -> Robot:
    # returns Robot with params/perception fields [1]
    f = cfg.fixed[name]
    t = lambda x: torch.tensor([float(x)], dtype=torch.float32)
    v = {k: t(f[k]) for k in cfg.ranges.keys()}
    pf = f.perception
    p = {k: t(pf[k]) for k in cfg.perception.keys() if k != "p_has_prediction"}
    has_pred = torch.tensor([bool(pf.has_prediction)])
    return Robot(
        params=_build_params(v, float(cfg.gravity_mps2)),
        perception=PerceptionProfile(**p, has_prediction=has_pred),
        runtime=runtime,
        robot_id=[f"kart_{name}_0"],
    )
