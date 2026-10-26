"""Inference modes (C4): amortized, bon_pred, refine, bon_true. Each returns (theta_n [B,14] in (-1, 1), pred_score [B])."""
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Tuple

import torch
from torch import Tensor, nn

from spec2mpc.model.policy import PolicyContext, Spec2MPCPolicy, draw_score, expected_score

MODES = ("amortized", "bon_pred", "refine", "bon_true")  # budget K = n for best-of-N, steps for refine
BOX_EPS = 1e-6


@contextmanager
def _eval(*mods: nn.Module) -> Iterator[None]:
    # W and V run in eval mode (no dropout) and get their training flags back afterwards
    flags = [m.training for m in mods]
    for m in mods:
        m.eval()
    try:
        yield
    finally:
        for m, f in zip(mods, flags):
            m.train(f)


def _score_pred(W: nn.Module, V: nn.Module, ctx: PolicyContext, theta_n: Tensor, xi_n: Tensor, lam: float) -> Tensor:
    # ctx [B], theta_n [B,14], xi_n [B,X,6] -> [B] mean over X of V per draw
    return expected_score(W, V, ctx.robot_tok, ctx.path_tok, theta_n, xi_n, ctx.text_emb, ctx.slots, lam)


@torch.no_grad()
def amortized(policy: Spec2MPCPolicy, W: nn.Module, V: nn.Module, ctx: PolicyContext, xi_n: Tensor,
              lam: float) -> Tuple[Tensor, Tensor]:
    theta = policy.amortized(ctx)  # [B,14]
    with _eval(W, V):
        return theta, _score_pred(W, V, ctx, theta, xi_n, lam)


@torch.no_grad()
def bon_pred(policy: Spec2MPCPolicy, W: nn.Module, V: nn.Module, ctx: PolicyContext, n: int, xi_n: Tensor,
             lam: float, gen: Optional[torch.Generator]) -> Tuple[Tensor, Tensor]:
    B = ctx.batch_size
    cand = policy.sample(ctx, n, gen)  # [B,n,14]
    rep = lambda t: t.repeat_interleave(n, dim=0)  # noqa: E731
    flat = cand.reshape(B * n, -1)
    with _eval(W, V):
        s = expected_score(W, V, rep(ctx.robot_tok), rep(ctx.path_tok), flat, rep(xi_n), rep(ctx.text_emb),
                           rep(ctx.slots), lam).view(B, n)
    best = s.argmax(1)
    return cand[torch.arange(B), best], s[torch.arange(B), best]


def refine(policy: Spec2MPCPolicy, W: nn.Module, V: nn.Module, ctx: PolicyContext, steps: int, lr: float,
           xi_n: Tensor, lam: float, trust_radius: float) -> Tuple[Tensor, Tensor]:
    # Adam on u with theta = tanh(u), L-inf ball around the amortized start; returns the best finite theta visited per row
    with _eval(W, V):
        theta0, s0 = amortized(policy, W, V, ctx, xi_n, lam)
        lo = (theta0 - trust_radius).clamp(min=-1 + BOX_EPS)
        hi = (theta0 + trust_radius).clamp(max=1 - BOX_EPS)
        best_theta = theta0.clone()
        best_s = torch.where(torch.isfinite(s0), s0, torch.full_like(s0, -float("inf")))
        u = torch.atanh(theta0.clamp(-1 + BOX_EPS, 1 - BOX_EPS)).detach().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=lr)
        frozen = torch.zeros(theta0.shape[0], dtype=torch.bool, device=theta0.device)

        def visit(theta: Tensor, s: Tensor, ok: Tensor) -> None:
            better = ok & (s > best_s)
            best_theta[better] = torch.max(torch.min(theta, hi), lo)[better]
            best_s[better] = s[better]

        for _ in range(steps):
            theta = torch.tanh(u)
            s = _score_pred(W, V, ctx, theta, xi_n, lam)  # [B]
            ok = torch.isfinite(s) & ~frozen
            g = torch.autograd.grad(torch.where(ok, -s, torch.zeros_like(s)).sum(), u)[0]
            ok = ok & torch.isfinite(g).all(-1)
            frozen = frozen | ~ok
            visit(theta.detach(), s.detach(), ok)
            u.grad = torch.where(frozen[:, None], torch.zeros_like(g), g)
            u_prev = u.detach().clone()
            opt.step()
            with torch.no_grad():
                proj = torch.max(torch.min(torch.tanh(u), hi), lo)  # [B,14]
                u.copy_(torch.atanh(proj))
                u[frozen] = u_prev[frozen]
        with torch.no_grad():
            theta = torch.tanh(u)
            s = _score_pred(W, V, ctx, theta, xi_n, lam)
            visit(theta, s, torch.isfinite(s) & ~frozen)
            # reported score is V(W(returned theta)) for every row, including non-finite amortized rows
            return best_theta, _score_pred(W, V, ctx, best_theta, xi_n, lam)


@torch.no_grad()
def bon_true(policy: Spec2MPCPolicy, V: nn.Module, ctx: PolicyContext, n: int,
             rollout_phi: Callable[[Tensor], Tensor], lam: float,
             gen: Optional[torch.Generator]) -> Tuple[Tensor, Tensor]:
    B = ctx.batch_size
    cand = policy.sample(ctx, n, gen)  # [B,n,14]
    phi = rollout_phi(cand.reshape(B * n, -1))  # [B*n,X,20], one true phi per xi draw
    rep = lambda t: t.repeat_interleave(n, dim=0)  # noqa: E731
    with _eval(V):
        s = draw_score(V, rep(ctx.text_emb), rep(ctx.slots), phi, lam).view(B, n)
    best = s.argmax(1)
    return cand[torch.arange(B), best], s[torch.arange(B), best]
