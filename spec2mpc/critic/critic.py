"""Intent critic V_psi (S2): (statement embedding, slots, episode features phi) -> normalized rubric score.

Inputs carry no rubric content; labels are passed in as tensors by the training script.
"""
from typing import Optional

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional as F

from spec2mpc.features.phi import PHI_NAMES
from spec2mpc.features.robust import robust_stats, robust_z

D_PHI = len(PHI_NAMES)


def _mlp(d_in: int, hidden: int, n_layers: int) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.GELU()]
        d = hidden
    return nn.Sequential(*layers, nn.Linear(d, 1))


class CriticEnsemble(nn.Module):
    def __init__(self, d_text: int, d_slot: int, d_phi: int, n_members: int, hidden: int, n_layers: int):
        super().__init__()
        self.members = nn.ModuleList(_mlp(d_text + d_slot + d_phi, hidden, n_layers) for _ in range(n_members))
        self.register_buffer("phi_center", torch.zeros(d_phi))
        self.register_buffer("phi_scale", torch.ones(d_phi))

    def set_phi_stats(self, phi: Tensor) -> None:
        # phi [E, d_phi] raw, train blocks only -> per-feature median and IQR / 1.349 (floored)
        center, scale = robust_stats(phi)
        self.phi_center.copy_(center)
        self.phi_scale.copy_(scale)

    def phi_to_z(self, phi: Tensor) -> Tensor:
        # phi [..., d_phi] raw -> z [..., d_phi] = asinh((phi - center) / scale), robust to heavy tails
        return robust_z(phi, self.phi_center, self.phi_scale)

    def _inputs(self, text_emb: Tensor, slots: Tensor, phi: Tensor) -> Tensor:
        # [B, d_text], [B, d_slot], [B, d_phi] raw -> [B, d_text + d_slot + d_phi]
        return torch.cat([text_emb, slots, self.phi_to_z(phi)], dim=-1)

    def member(self, m: int, text_emb: Tensor, slots: Tensor, phi: Tensor) -> Tensor:
        # member m only: [B, d_text], [B, d_slot], [B, d_phi] raw -> [B]
        return self.members[m](self._inputs(text_emb, slots, phi)).squeeze(-1)

    def forward(self, text_emb: Tensor, slots: Tensor, phi: Tensor) -> Tensor:
        # text_emb [B, d_text], slots [B, d_slot], phi [B, d_phi] raw -> [M, B]
        x = self._inputs(text_emb, slots, phi)
        return torch.stack([net(x).squeeze(-1) for net in self.members])

    def score(self, text_emb: Tensor, slots: Tensor, phi: Tensor, lam: float) -> Tensor:
        # -> [B] = ensemble mean - lam * ensemble std
        out = self(text_emb, slots, phi)  # [M, B]
        return out.mean(0) - lam * out.std(0)


def critic_loss(pred: Tensor, target: Tensor, group: Tensor, w_rank: float, rank_margin: float,
                pair_weight: Optional[Tensor] = None, mse_target: Optional[Tensor] = None) -> Tensor:
    # pred [M, B], target [B], group [B] long -> scalar, sum over members of
    # MSE + w_rank * mean logistic loss over ordered same-group pairs with target_i - target_j > rank_margin;
    # pair_weight [B, B] turns the pair mean into a weighted mean, mse_target [B] replaces target in the MSE term
    mse = ((pred - (target if mse_target is None else mse_target)) ** 2).mean(1)  # [M]
    pairs = (group[:, None] == group[None, :]) & (target[:, None] - target[None, :] > rank_margin)  # [B, B]
    if not pairs.any():
        return mse.sum()
    i, j = pairs.nonzero(as_tuple=True)
    loss = F.softplus(-(pred[:, i] - pred[:, j]))  # [M, P]
    if pair_weight is None:
        rank = loss.mean(1)  # [M]
    else:
        w = pair_weight[i, j]  # [P]
        rank = (loss * w).sum(1) / w.sum()
    return (mse + w_rank * rank).sum()


def flip_pair_weight(j_all: Tensor, ok_all: Tensor, rank_margin: float, upweight: float) -> Tensor:
    # j_all [B, K] labels of K cards per row, ok_all [B, K] in conditions -> [B, B] weights: upweight where two cards
    # (both in conditions for rows i and j) order the pair oppositely by > rank_margin, else 1
    both = ok_all[:, None, :] & ok_all[None, :, :]  # [B, B, K]
    d = torch.nan_to_num(j_all[:, None, :] - j_all[None, :, :])  # [B, B, K]
    flip = ((d > rank_margin) & both).any(-1) & ((d < -rank_margin) & both).any(-1)  # [B, B]
    return 1.0 + (upweight - 1.0) * flip.to(j_all.dtype)


def build_critic(cfg: DictConfig) -> CriticEnsemble:
    # cfg: train/critic config with d_text and d_slot filled by the training script
    return CriticEnsemble(int(cfg.d_text), int(cfg.d_slot), D_PHI, int(cfg.members), int(cfg.hidden), int(cfg.n_layers))
