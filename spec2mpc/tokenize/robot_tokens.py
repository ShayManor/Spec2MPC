from dataclasses import fields
from typing import Dict, Mapping

import torch
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.robots.kart import Robot

ROBOT_TOKEN_TYPES = ("chassis", "wheel_front", "wheel_rear", "limits", "perception", "runtime")
N_ROBOT_TOK = len(ROBOT_TOKEN_TYPES)
F_MAX = 9
D_ROBOT_TOK = N_ROBOT_TOK + 2 * F_MAX


def robot_columns(robot: Robot) -> Dict[str, Tensor]:
    # robot [B] -> {episodes.parquet column: [B]} for the robot fields
    b = len(robot.robot_id)
    cols: Dict[str, Tensor] = {}
    for f in fields(robot.params):
        cols[f"kart_{f.name}"] = getattr(robot.params, f.name)
    for f in fields(robot.perception):
        cols[f"perc_{f.name}"] = getattr(robot.perception, f.name)
    rt = robot.runtime
    dev = robot.params.mass_kg.device
    cols["rt_control_rate_hz"] = torch.full((b,), rt.control_rate_hz, dtype=torch.float32, device=dev)
    cols["rt_solver_iters"] = torch.full((b,), rt.solver_iters, dtype=torch.int64, device=dev)
    cols["rt_perception_rate_hz"] = torch.full((b,), rt.perception_rate_hz, dtype=torch.float32, device=dev)
    return cols


def robot_tokens_from_columns(cols: Mapping[str, Tensor], tok_cfg: DictConfig) -> Tensor:
    # cols: column -> [E] -> tokens [E, N_ROBOT_TOK, D_ROBOT_TOK] = [type one-hot | values padded to F_MAX | value mask]
    f_max = int(tok_cfg.f_max)
    ref = next(iter(cols.values()))
    e = ref.shape[0]
    tokens = torch.zeros(e, N_ROBOT_TOK, N_ROBOT_TOK + 2 * f_max, dtype=torch.float32, device=ref.device)
    for i, name in enumerate(ROBOT_TOKEN_TYPES):
        names = list(tok_cfg.tokens[name])
        if len(names) > f_max:
            raise ValueError(f"token {name} has {len(names)} fields, f_max is {f_max}")
        tokens[:, i, i] = 1.0
        for j, col in enumerate(names):
            if col not in cols:
                raise KeyError(f"missing robot column {col}")
            center, half = (float(x) for x in tok_cfg.norm[col])
            tokens[:, i, N_ROBOT_TOK + j] = (cols[col].to(torch.float32) - center) / half
            tokens[:, i, N_ROBOT_TOK + f_max + j] = 1.0
    return tokens


def robot_tokens(robot: Robot, tok_cfg: DictConfig) -> Tensor:
    # robot [B] -> tokens [B, N_ROBOT_TOK, D_ROBOT_TOK]
    return robot_tokens_from_columns(robot_columns(robot), tok_cfg)
