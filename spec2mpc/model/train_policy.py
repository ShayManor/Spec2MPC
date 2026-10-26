"""S4 training: stage A distills theta* with the CVAE ELBO; stage B maximizes V(W(theta)) over the global xi prior
with W and V frozen, plus alpha_distill * ELBO on distill batches."""
import json
import math
import time
from typing import List, Optional, Sequence, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from spec2mpc.model.data import RolloutTensors, normalize_xi, xi_from_conditions
from spec2mpc.model.policy import ContextTable, Spec2MPCPolicy, build_policy, vw_objective
from spec2mpc.sim.conditions import sample_conditions


def global_xi(n: int, n_xi: int, sim_cfg: DictConfig, gen: torch.Generator) -> Tensor:
    # -> xi_n [n, n_xi, 6] drawn from the global condition prior (never a card's ranges), normalized
    cond = sample_conditions(n * n_xi, sim_cfg, gen)
    return normalize_xi(xi_from_conditions(cond), sim_cfg).view(n, n_xi, -1)


def check_split(cfg: DictConfig, w_cfg: DictConfig, v_cfg: DictConfig) -> None:
    # the policy's train blocks must be the blocks W (data.val_frac, data.split_seed) and V (val_frac, split_seed) trained on
    want = (float(cfg.data.val_frac), int(cfg.data.split_seed))
    for name, got in (("world model", (float(w_cfg.data.val_frac), int(w_cfg.data.split_seed))),
                      ("critic", (float(v_cfg.val_frac), int(v_cfg.split_seed)))):
        if got != want:
            raise ValueError(f"{name} split (val_frac, split_seed) = {got} differs from the policy's {want}")


def distill_table(problem_ids: Sequence[str], robot_tok: Tensor, path_tok: Tensor, text_emb: Tensor, slots: Tensor,
                  stmt_card: Sequence[str], oracle_pid: Sequence[str], oracle_cid: Sequence[str],
                  theta_n: Tensor) -> Tuple[ContextTable, Tensor]:
    # problems [P] (robot_tok [P,6,24], path_tok [P,64,6]); train statements [S] (text_emb, slots, card of each);
    # oracle records [R] (problem id, card id, theta*_n [R,14]) -> one row per (record, train statement of its card)
    p_index = {p: i for i, p in enumerate(problem_ids)}
    pair, rows = [], []
    for r, (pid, cid) in enumerate(zip(oracle_pid, oracle_cid)):
        stmts = [s for s, c in enumerate(stmt_card) if c == cid]
        assert stmts, f"card {cid} has no train statements"
        pair += [(p_index[pid], s) for s in stmts]
        rows += [r] * len(stmts)
    table = ContextTable(robot_tok, path_tok, text_emb, slots, torch.tensor(pair, dtype=torch.long).view(-1, 2))
    return table, theta_n[torch.tensor(rows, dtype=torch.long)]


def vw_table(data: RolloutTensors, train: Tensor, text_emb: Tensor, slots: Tensor, stmt_task: Tensor) -> ContextTable:
    # data [E], train [E] bool (train blocks), train statements [S] with task ids stmt_task [S] ->
    # first episode of every train block crossed with the train statements of the block's task
    e = train.nonzero().squeeze(1)
    order = e[torch.argsort(data.block_id[e], stable=True)]
    b = data.block_id[order]
    first = order[torch.cat([torch.ones(1, dtype=torch.bool), b[1:] != b[:-1]])]  # [P] one episode per block
    task = data.task[first]
    pair = (task[:, None] == stmt_task[None, :]).nonzero()  # [N, 2] (block, statement)
    return ContextTable(data.robot_tok[first], data.path_tok[first], text_emb, slots, pair)


def train_policy(cfg: DictConfig, distill_data: Tuple[ContextTable, Tensor], vw_contexts: ContextTable,
                 W: nn.Module, V: nn.Module, log: Optional[List[dict]] = None) -> Spec2MPCPolicy:
    # distill_data: contexts [R] and theta*_n [R, 14]; vw_contexts [N]; log (optional) receives one dict per
    # stage-A epoch and per stage-B step
    device = torch.device(cfg.device)
    table, theta_star = distill_data[0].to(device), distill_data[1].to(device)
    vw_contexts = vw_contexts.to(device)
    W, V = W.to(device).eval().requires_grad_(False), V.to(device).eval().requires_grad_(False)
    policy = build_policy(cfg).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=float(cfg.opt.lr))
    gen = torch.Generator().manual_seed(int(cfg.seed))  # batches and xi, CPU
    gen_s = torch.Generator().manual_seed(int(cfg.seed) + 1)  # latent noise, CPU
    bs, beta = int(cfg.opt.batch), float(cfg.beta)
    log = [] if log is None else log
    R, t0 = table.batch_size, time.perf_counter()
    every = int(cfg.get("log_every", 0))  # print every `every` epochs / steps; 0 is silent

    for epoch in range(int(cfg.epochs_distill)):
        policy.train()
        perm = torch.randperm(R, generator=gen).to(device)
        n_b, tot = math.ceil(R / bs), 0.0
        for k in range(n_b):
            idx = perm[k * bs:(k + 1) * bs]
            loss = policy.elbo_loss(table.index(idx), theta_star[idx], beta, gen_s)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        log.append({"stage": "distill", "epoch": epoch + 1, "elbo": tot / n_b, "time_s": time.perf_counter() - t0})
        if every and (epoch + 1) % every == 0:
            print(json.dumps(log[-1]), flush=True)

    N = vw_contexts.batch_size
    for step in range(int(cfg.steps_vw)):
        policy.train()
        ci = torch.randint(0, N, (min(bs, N),), generator=gen).to(device)
        di = torch.randint(0, R, (min(bs, R),), generator=gen).to(device)
        xi = global_xi(ci.numel(), int(cfg.n_xi), cfg.sim, gen).to(device)  # [b, X, 6]
        vw = vw_objective(policy, W, V, vw_contexts.index(ci), xi, int(cfg.n_samples), float(cfg.lam), gen_s)  # [b]
        elbo = policy.elbo_loss(table.index(di), theta_star[di], beta, gen_s)
        loss = -vw.mean() + float(cfg.alpha_distill) * elbo
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        log.append({"stage": "vw", "step": step + 1, "vw": vw.mean().item(), "elbo": elbo.item(),
                    "time_s": time.perf_counter() - t0})
        if every and (step + 1) % every == 0:
            print(json.dumps(log[-1]), flush=True)
    return policy.eval()
