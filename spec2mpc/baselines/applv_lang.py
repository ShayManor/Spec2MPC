"""applv_lang baseline: direct regression (statement, robot, path) -> normalized theta_n in [-1, 1]^14.

BC on theta*, then REINFORCE on a reward callable injected by the training script. No world model, no critic,
and no rubric code: the reward arrives as a function of theta_n only.
"""
from typing import Callable, Dict, Optional

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from spec2mpc.model.policy import ContextEncoder, PolicyContext
from spec2mpc.model.world_model import D_THETA


class APPLVRegressor(nn.Module):
    def __init__(self, encoder: ContextEncoder, hidden: int):
        super().__init__()
        d = encoder.cls.shape[-1]
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, D_THETA))

    def forward(self, ctx: PolicyContext) -> Tensor:
        # ctx [B] -> theta_n [B, 14] in (-1, 1)
        return torch.tanh(self.head(self.encoder(ctx)))


def build_applv(cfg: DictConfig) -> APPLVRegressor:
    # cfg: train/applv config with d_text and d_slot filled by the training script; encoder as the policy, text_mode full
    m = cfg.model
    enc = ContextEncoder(int(cfg.d_text), int(cfg.d_slot), int(m.d_model), int(m.n_layers), int(m.n_heads),
                         float(m.dropout), "full")
    return APPLVRegressor(enc, int(cfg.hidden))


def bc_loss(model: APPLVRegressor, ctx: PolicyContext, theta_n: Tensor) -> Tensor:
    # ctx [B], theta_n [B, 14] -> scalar MSE
    return ((model(ctx) - theta_n) ** 2).mean()


def reinforce_step(model: APPLVRegressor, ctx: PolicyContext, reward_fn: Callable[[Tensor], Tensor], sigma: float,
                   n_samples: int, opt: torch.optim.Optimizer,
                   gen: Optional[torch.Generator] = None) -> Dict[str, float]:
    # ctx [B]; reward_fn(theta_n [B, n, 14]) -> [B, n], called once; Gaussian policy N(mu, sigma^2 I) around
    # mu = model(ctx); log-prob on the unclamped sample, reward_fn sees it clamped to the box; per-row standardized
    # advantage; one optimizer step; also returns the fraction of rows with zero advantage (all n rewards equal)
    assert n_samples >= 2, "per-row advantage needs n_samples >= 2"
    mu = model(ctx)  # [B, 14]
    shape = (mu.shape[0], n_samples, D_THETA)
    eps = torch.randn(shape) if gen is None else torch.randn(shape, generator=gen, device=gen.device)
    raw = mu.detach()[:, None] + sigma * eps.to(mu.device)  # [B, n, 14] unclamped Gaussian sample
    reward = reward_fn(raw.clamp(-1.0, 1.0)).to(mu.device, mu.dtype)  # [B, n]; the clamp is part of the environment
    adv = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-8)  # [B, n]
    logp = -((raw - mu[:, None]) ** 2).sum(-1) / (2 * sigma ** 2)  # [B, n], log N(raw; mu, sigma) + const
    loss = -(adv.detach() * logp).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    zero_adv = (reward.max(1).values == reward.min(1).values).float().mean()  # rows whose n rewards are all equal
    return {"reward": reward.mean().item(), "loss": loss.item(), "zero_adv_frac": zero_adv.item()}
