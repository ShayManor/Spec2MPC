import torch
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.robots.kart import KartParams
from spec2mpc.sim.paths import RefPath, path_lookup
from spec2mpc.sim.scenes import Scene, true_clearance

PATH_TOKEN_FIELDS = ("curvature_x_Rmin", "clear_left_over_L", "clear_right_over_L", "seg_len_over_L", "closed", "is_goal")

def _march_clearance(scene: Scene, origin: Tensor, direction: Tensor, cfg: DictConfig) -> Tensor:
    # origin, direction [B, K, 2] -> distance [B, K] along direction to the first occupied point, capped at clear_cap_m
    d = torch.zeros(origin.shape[:2], dtype=origin.dtype, device=origin.device)
    for _ in range(int(cfg.march_max_steps)):
        c = true_clearance(scene, origin + d[..., None] * direction, 0.0)  # [B, K]
        live = (c >= float(cfg.march_eps_m)) & (d < float(cfg.clear_cap_m))
        if not live.any():
            break
        d = torch.where(live, d + c, d)
    return d.clamp(max=float(cfg.clear_cap_m))


def path_tokens(path: RefPath, scene: Scene, p: KartParams, tok_cfg: DictConfig) -> Tensor:
    # path [B, M], scene (true, t = 0), p [B] -> tokens [B, n_path_tokens, 6] ordered as PATH_TOKEN_FIELDS
    cfg = tok_cfg
    n_tokens = int(cfg.n_path_tokens)
    if n_tokens < 2:
        raise ValueError(f"n_path_tokens must be >= 2, got {n_tokens}")
    b = path.batch_size
    u = torch.arange(n_tokens, dtype=path.xy.dtype, device=path.xy.device)  # [K]
    denom = torch.where(path.closed, torch.full_like(path.length_m, float(n_tokens)), torch.full_like(path.length_m, float(n_tokens - 1)))  # [B]
    spacing = path.length_m / denom  # [B]
    s = u[None] * spacing[:, None]  # [B, K]
    xy, heading, curv = path_lookup(path, s)
    normal = torch.stack([-torch.sin(heading), torch.cos(heading)], dim=-1)  # [B, K, 2]
    left = _march_clearance(scene, xy, normal, cfg)  # [B, K]
    right = _march_clearance(scene, xy, -normal, cfg)
    wb = p.wheelbase_m[:, None]  # [B, 1]
    is_goal = torch.zeros(b, n_tokens, dtype=xy.dtype, device=xy.device)
    is_goal[:, -1] = (~path.closed).to(xy.dtype)
    return torch.stack(
        [
            curv * p.min_turn_radius_m[:, None],
            left / wb,
            right / wb,
            (spacing[:, None] / wb).expand(-1, n_tokens),
            path.closed[:, None].to(xy.dtype).expand(-1, n_tokens),
            is_goal,
        ],
        dim=-1,
    )
