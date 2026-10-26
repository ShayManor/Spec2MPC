"""Policy (S4): context (robot, statement, slots, path) -> latent z -> CVAE head -> normalized theta_n in [-1, 1]^14.

No rubric input; theta* labels and the frozen W and V are passed in by the training code.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from spec2mpc.batch import Batched
from spec2mpc.model.world_model import D_PATH, D_ROBOT, D_THETA, N_PATH, N_ROBOT

TEXT_MODES = ("full", "zero")
N_TYPES = 5  # cls, robot, text, slots, path
LOGSTD_MIN, LOGSTD_MAX = -5.0, 2.0  # latent log std clamp for prior and posterior


@dataclass
class PolicyContext(Batched):
    robot_tok: Tensor  # [B, 6, 24]
    text_emb: Tensor  # [B, d_text]
    slots: Tensor  # [B, d_slot]
    path_tok: Tensor  # [B, 64, 6]


@dataclass
class ContextTable:
    # contexts stored as (problem, statement) index pairs, so a cross product costs no copies
    robot_tok: Tensor  # [P, 6, 24]
    path_tok: Tensor  # [P, 64, 6]
    text_emb: Tensor  # [S, d_text]
    slots: Tensor  # [S, d_slot]
    pair: Tensor  # [N, 2] long, (problem index, statement index)

    @property
    def batch_size(self) -> int:
        return int(self.pair.shape[0])

    def index(self, idx: Tensor) -> PolicyContext:
        # idx [M] long -> PolicyContext [M]
        p, s = self.pair[idx, 0], self.pair[idx, 1]
        return PolicyContext(self.robot_tok[p], self.text_emb[s], self.slots[s], self.path_tok[p])

    def to(self, device: torch.device) -> "ContextTable":
        return ContextTable(self.robot_tok.to(device), self.path_tok.to(device), self.text_emb.to(device),
                            self.slots.to(device), self.pair.to(device))


class ContextEncoder(nn.Module):
    def __init__(self, d_text: int, d_slot: int, d_model: int, n_layers: int, n_heads: int, dropout: float,
                 text_mode: str):
        super().__init__()
        assert text_mode in TEXT_MODES, f"text_mode must be one of {TEXT_MODES}, got {text_mode}"
        self.text_scale = 1.0 if text_mode == "full" else 0.0
        self.robot_in = nn.Linear(D_ROBOT, d_model)
        self.text_in = nn.Linear(d_text, d_model)
        self.slot_in = nn.Linear(d_slot, d_model)
        self.path_in = nn.Linear(D_PATH, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.path_pos = nn.Parameter(0.02 * torch.randn(1, N_PATH, d_model))
        self.type_emb = nn.Parameter(0.02 * torch.randn(N_TYPES, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        types = [0] + [1] * N_ROBOT + [2, 3] + [4] * N_PATH
        self.register_buffer("types", torch.tensor(types), persistent=False)

    def forward(self, ctx: PolicyContext) -> Tensor:
        # robot_tok [B,6,24], text_emb [B,d_text], slots [B,d_slot], path_tok [B,64,6] -> z [B, d_model]
        B = ctx.robot_tok.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), self.robot_in(ctx.robot_tok),
                       self.text_in(ctx.text_emb * self.text_scale)[:, None], self.slot_in(ctx.slots)[:, None],
                       self.path_in(ctx.path_tok) + self.path_pos], dim=1)  # [B, 73, d]
        x = self.encoder(x + self.type_emb[self.types])  # [B, 73, d]
        return x[:, 0]


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d_out))


def _randn(shape: Tuple[int, ...], gen: Optional[torch.Generator], device: torch.device) -> Tensor:
    # standard normal drawn on the generator's device, then moved
    if gen is None:
        return torch.randn(shape, device=device)
    return torch.randn(shape, generator=gen, device=gen.device).to(device)


class Spec2MPCPolicy(nn.Module):
    def __init__(self, encoder: ContextEncoder, d_latent: int, hidden: int):
        super().__init__()
        d = encoder.cls.shape[-1]
        self.encoder = encoder
        self.d_latent = d_latent
        self.prior_net = _mlp(d, hidden, 2 * d_latent)
        self.posterior_net = _mlp(d + D_THETA, hidden, 2 * d_latent)
        self.decoder = _mlp(d + d_latent, hidden, D_THETA)

    def prior(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        # z [B, d] -> mu, logstd [B, d_latent]
        mu, logstd = self.prior_net(z).chunk(2, dim=-1)
        return mu, logstd.clamp(LOGSTD_MIN, LOGSTD_MAX)

    def posterior(self, z: Tensor, theta_n: Tensor) -> Tuple[Tensor, Tensor]:
        # z [B, d], theta_n [B, 14] -> mu, logstd [B, d_latent]
        mu, logstd = self.posterior_net(torch.cat([z, theta_n], dim=-1)).chunk(2, dim=-1)
        return mu, logstd.clamp(LOGSTD_MIN, LOGSTD_MAX)

    def decode(self, z: Tensor, eps: Tensor) -> Tensor:
        # z [B, d], eps [B, N, d_latent] latent samples -> theta_n [B, N, 14] in (-1, 1)
        N = eps.shape[1]
        return torch.tanh(self.decoder(torch.cat([z[:, None].expand(-1, N, -1), eps], dim=-1)))

    def sample(self, ctx: PolicyContext, n: int, gen: Optional[torch.Generator]) -> Tensor:
        # ctx [B] -> theta_n [B, n, 14], latents reparameterized from the prior (differentiable)
        z = self.encoder(ctx)  # [B, d]
        mu, logstd = self.prior(z)
        noise = _randn((z.shape[0], n, self.d_latent), gen, z.device)
        return self.decode(z, mu[:, None] + logstd.exp()[:, None] * noise)

    def amortized(self, ctx: PolicyContext) -> Tensor:
        # ctx [B] -> theta_n [B, 14] decoded at the prior mean
        z = self.encoder(ctx)
        return self.decode(z, self.prior(z)[0][:, None])[:, 0]

    def elbo_loss(self, ctx: PolicyContext, theta_n: Tensor, beta: float, gen: Optional[torch.Generator] = None) -> Tensor:
        # ctx [B], theta_n [B, 14] -> scalar: mean over B of squared error summed over theta + beta * KL(q || prior)
        z = self.encoder(ctx)
        mq, lq = self.posterior(z, theta_n)
        mp, lp = self.prior(z)
        lat = mq + lq.exp() * _randn(tuple(mq.shape), gen, z.device)  # [B, d_latent]
        recon = ((self.decode(z, lat[:, None])[:, 0] - theta_n) ** 2).sum(-1)  # [B]
        kl = (lp - lq + ((2 * lq).exp() + (mq - mp) ** 2) / (2 * (2 * lp).exp()) - 0.5).sum(-1)  # [B]
        return (recon + beta * kl).mean()


def draw_score(V: nn.Module, text_emb: Tensor, slots: Tensor, phi: Tensor, lam: float) -> Tensor:
    # text_emb [B, d_text], slots [B, d_slot], phi [B, X, 20] one feature vector per xi draw -> [B] mean over X of
    # V.score per draw (V is trained on per-episode phi, so it scores draws, never their mean)
    B, X = phi.shape[:2]
    rep = lambda t: t.repeat_interleave(X, dim=0)  # noqa: E731
    return V.score(rep(text_emb), rep(slots), phi.reshape(B * X, -1), lam).view(B, X).mean(1)


def expected_score(W: nn.Module, V: nn.Module, robot_tok: Tensor, path_tok: Tensor, theta_n: Tensor, xi_n: Tensor,
                   text_emb: Tensor, slots: Tensor, lam: float) -> Tensor:
    # robot_tok [B,6,24], path_tok [B,64,6], theta_n [B,14], xi_n [B,X,6], text_emb [B,d_text], slots [B,d_slot]
    # -> [B] mean over the X draws of V.score(text, slots, W(robot, path, theta, xi_x), lam)
    B, X = xi_n.shape[:2]
    rep = lambda t: t.repeat_interleave(X, dim=0)  # noqa: E731
    phi = W(rep(robot_tok), rep(path_tok), rep(theta_n), xi_n.reshape(B * X, -1))  # [B*X, 20]
    return draw_score(V, text_emb, slots, phi.view(B, X, -1), lam)


def vw_objective(policy: Spec2MPCPolicy, W: nn.Module, V: nn.Module, ctx: PolicyContext, xi_n: Tensor,
                 n_samples: int, lam: float, gen: Optional[torch.Generator]) -> Tensor:
    # ctx [B], xi_n [B, X, 6] from the global prior -> [B] mean over n_samples prior draws of
    # expected_score(W, V, robot, path, theta, xi, text, slots, lam); differentiable in the policy parameters
    B = xi_n.shape[0]
    theta = policy.sample(ctx, n_samples, gen).reshape(B * n_samples, D_THETA)  # [B*N, 14]
    rep = lambda t: t.repeat_interleave(n_samples, dim=0)  # noqa: E731
    return expected_score(W, V, rep(ctx.robot_tok), rep(ctx.path_tok), theta, rep(xi_n), rep(ctx.text_emb),
                          rep(ctx.slots), lam).view(B, n_samples).mean(1)


def build_policy(cfg: DictConfig) -> Spec2MPCPolicy:
    # cfg: train/policy config with d_text and d_slot filled by the training script
    m = cfg.model
    enc = ContextEncoder(int(cfg.d_text), int(cfg.d_slot), int(m.d_model), int(m.n_layers), int(m.n_heads),
                         float(m.dropout), str(cfg.text_mode))
    return Spec2MPCPolicy(enc, int(m.d_latent), int(m.hidden))
