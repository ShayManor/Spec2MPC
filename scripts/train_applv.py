"""applv_lang baseline: stage=bc fits theta* on the distill rows; stage=rl runs REINFORCE from the bc checkpoint on
train-card rubric rewards from true sim rollouts (no world model, no critic).

Usage: python scripts/train_applv.py stage=bc|rl device=cuda oracle=.../train/oracle.parquet out_dir=... seed=0
stage=rl checkpoints every iteration to {out_dir}/seed{seed}/rl_state.pt; RESUME=1 continues it, without RESUME an
existing rl_state.pt is an error. The reward is built here from the train cards and injected as a callable, so
spec2mpc/baselines/applv_lang.py never sees rubric content.
"""
import json
import os
import time
from pathlib import Path
from typing import Callable, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor

from spec2mpc.baselines.applv_lang import bc_loss, build_applv, reinforce_step
from spec2mpc.model.data import load_theta_star
from spec2mpc.model.io import load_checkpoint, save_checkpoint
from spec2mpc.model.policy import ContextTable
from spec2mpc.model.train_policy import distill_table
from spec2mpc.mpc.theta import ThetaBatch, denormalize_theta, normalize_theta
from spec2mpc.rubric.cards import Card, load_cards
from spec2mpc.rubric.scoring import card_max_score, sample_card_conditions, score
from spec2mpc.rubric.splits import statement_records
from spec2mpc.sim.conditions import Conditions
from spec2mpc.sim.evaluate import rollout_metrics
from spec2mpc.sim.problems import Problems, load_oracle_problems, stable_seed
from spec2mpc.sim.storage import code_digest, git_hash
from spec2mpc.text.encoder import make_text_encoder, text_features
from spec2mpc.tokenize.robot_tokens import robot_tokens

STATE_FILE = "rl_state.pt"
RESUME_KEYS = ("rl.sigma", "rl.n_samples", "rl.xi_per_sample", "rl.rows_per_iter", "rl.lr", "oracle",
               "bc_checkpoint")  # rl.iterations may grow on resume


def rollout_cfg(cfg: DictConfig, ocfg: DictConfig) -> DictConfig:
    # RL rollouts run under the oracle run's resolved sim, robots and mpc sections (its runtime is in the problems);
    # only the device and mpc.solver.compile (speed knobs) come from the applv config
    rc = OmegaConf.create({"sim": ocfg.sim, "robots": ocfg.robots, "mpc": ocfg.mpc, "device": cfg.device})
    rc.mpc.solver.compile = bool(cfg.mpc.solver.compile)
    return rc


def card_reward_fn(problems: Problems, row_problem: List[int], row_card: List[Card], cfg: DictConfig, X: int,
                   gen: torch.Generator, rollout_gen: torch.Generator) -> Callable[[Tensor], Tensor]:
    # cfg: rollout_cfg; rows [B] (problem index, card) -> reward_fn(theta_n [B, n, 14]) -> [B, n]: mean over X draws
    # of sample_card_conditions(card) of score(card) / card_max_score(card); the n samples of a row share its draws;
    # all B * n * X episodes in one rollout_metrics call

    def reward_fn(theta_n: Tensor) -> Tensor:
        B, n, D = theta_n.shape
        rep = torch.tensor(row_problem).repeat_interleave(n * X)  # [E] problem index, row-major, then sample, then xi
        robot, scene = problems.expand(rep)
        values = denormalize_theta(theta_n.detach().float().cpu(), cfg.mpc)  # [B, n, 14]
        th = ThetaBatch(values=values.repeat_interleave(X, dim=1).reshape(-1, D))  # [E, 14]
        cd = Conditions.cat([sample_card_conditions(c, X, cfg.sim, gen).index(torch.arange(X).repeat(n))
                             for c in row_card])  # [E]
        metrics = rollout_metrics(robot, scene, th, cd, cfg, rollout_gen, torch.device(cfg.device)).metrics
        r = torch.empty(B, n, dtype=torch.float64)
        for i, c in enumerate(row_card):
            sl = slice(i * n * X, (i + 1) * n * X)
            r[i] = score(c, {k: v[sl] for k, v in metrics.items()}).view(n, X).mean(1) / card_max_score(c)
        return r.float()

    return reward_fn


def run_bc(cfg: DictConfig, table: ContextTable, theta_n: Tensor, out: Path, extra: dict) -> None:
    device = torch.device(cfg.device)
    table, theta_n = table.to(device), theta_n.to(device)
    model = build_applv(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.bc.lr))
    gen = torch.Generator().manual_seed(int(cfg.seed))
    R, bs, every, t0, history = table.batch_size, int(cfg.bc.batch), int(cfg.log_every), time.perf_counter(), []
    for epoch in range(int(cfg.bc.epochs)):
        model.train()
        perm, tot, n_b = torch.randperm(R, generator=gen).to(device), 0.0, 0
        for k in range(0, R, bs):
            idx = perm[k:k + bs]
            loss = bc_loss(model, table.index(idx), theta_n[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, n_b = tot + loss.item(), n_b + 1
        history.append({"epoch": epoch + 1, "mse": tot / n_b, "time_s": time.perf_counter() - t0})
        if every and (epoch + 1) % every == 0:
            print(json.dumps(history[-1]), flush=True)
    save_checkpoint(out / "bc.pt", model.cpu().eval(), cfg, {**extra, "stage": "bc", "last": history[-1] if history else {}})
    (out / "bc_metrics.json").write_text(json.dumps({"history": history, "n_rows": R, "git_hash": git_hash(),
                                                     "code_digest": code_digest(),
                                                     "config": OmegaConf.to_container(cfg, resolve=True)}, indent=2))
    print(f"wrote {out / 'bc.pt'}", flush=True)


def run_rl(cfg: DictConfig, ocfg: DictConfig, table: ContextTable, problems: Problems, row_card: List[Card],
           out: Path, extra: dict) -> None:
    device = torch.device(cfg.device)
    model, _, bc_extra = load_checkpoint(Path(cfg.bc_checkpoint), build_applv)
    assert bc_extra["encoder"] == extra["encoder"], f"bc used {bc_extra['encoder']}, rl uses {extra['encoder']}"
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.rl.lr))
    rl = cfg.rl
    view = {k: OmegaConf.select(cfg, k) for k in RESUME_KEYS}
    it0, history, state_path = 0, [], out / STATE_FILE
    if state_path.exists():
        if os.environ.get("RESUME", "0") != "1":
            raise FileExistsError(f"{state_path} exists; set RESUME=1 to continue it or pick another out_dir")
        st = torch.load(state_path, map_location="cpu", weights_only=True)
        if st["view"] != view:
            raise ValueError(f"{state_path} was made with rl settings {st['view']}, current {view}")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        it0, history = int(st["iteration"]), st["history"]
    pair = table.pair.tolist()  # [R] (problem index, statement index)
    table = table.to(device)
    seed, R, n, X = int(cfg.seed), table.batch_size, int(rl.n_samples), int(rl.xi_per_sample)
    rc = rollout_cfg(cfg, ocfg)
    for it in range(it0, int(rl.iterations)):
        t0 = time.perf_counter()
        gen = torch.Generator().manual_seed(stable_seed(seed, "rl", it))  # rows and card conditions
        idx = torch.randperm(R, generator=gen)[:int(rl.rows_per_iter)]
        rows = idx.tolist()
        reward_fn = card_reward_fn(problems, [pair[i][0] for i in rows], [row_card[i] for i in rows], rc, X, gen,
                                   torch.Generator().manual_seed(stable_seed(seed, "rollout", it)))
        model.train()
        step = reinforce_step(model, table.index(idx.to(device)), reward_fn, float(rl.sigma), n, opt,
                              torch.Generator().manual_seed(stable_seed(seed, "noise", it)))
        dt, eps = time.perf_counter() - t0, idx.numel() * n * X
        history.append({"iteration": it + 1, **step, "episodes": eps, "time_s": dt, "eps_per_s": eps / dt})
        print(json.dumps(history[-1]), flush=True)
        save_checkpoint(out / "model.pt", model, cfg, {**extra, "stage": "rl", "iteration": it + 1,
                                                       "bc_git_hash": bc_extra["git_hash"]})
        tmp = state_path.with_name(f".tmp_{STATE_FILE}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iteration": it + 1, "history": history,
                    "view": view}, tmp)
        os.replace(tmp, state_path)
    (out / "metrics.json").write_text(json.dumps({"history": history, "n_rows": R, "git_hash": git_hash(),
                                                  "code_digest": code_digest(),
                                                  "config": OmegaConf.to_container(cfg, resolve=True)}, indent=2))
    print(f"wrote {out / 'model.pt'}", flush=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train/applv")
def main(cfg: DictConfig) -> None:
    assert cfg.stage in ("bc", "rl"), f"stage must be bc or rl, got {cfg.stage}"
    torch.manual_seed(int(cfg.seed))
    recs = [r for r in statement_records(cfg.splits) if r.role == "train"]
    encoder = make_text_encoder(cfg.text)
    emb, slots = text_features(encoder, [r.text for r in recs], cfg.text.slots)  # [S, d_text], [S, d_slot]

    # rows: (oracle train problem, train statement of the record's card); problems rebuilt from the oracle run's own
    # config.yaml (digest-checked), theta* normalized with its mpc section
    problems, ocfg = load_oracle_problems(Path(cfg.oracle).parent)
    assert str(ocfg.split) == "train", f"applv trains on the train oracle, got split {ocfg.split}"
    pid, cid, task, theta = load_theta_star(Path(cfg.oracle))
    p_task = dict(zip(problems.problem_id, problems.task))
    assert all(p_task.get(p) == t for p, t in zip(pid, task)), "oracle problems differ from the rebuilt problems"
    table, theta_n = distill_table(problems.problem_id, robot_tokens(problems.robot, cfg.robot_tokens),
                                   problems.path_tok.float(), emb, slots, [r.card_id for r in recs], pid, cid,
                                   normalize_theta(theta, ocfg.mpc))
    print(f"stage {cfg.stage}: {table.batch_size} rows from {len(pid)} oracle records x train statements", flush=True)

    with open_dict(cfg):
        cfg.d_text, cfg.d_slot = int(emb.shape[1]), int(slots.shape[1])
    out = Path(cfg.out_dir) / f"seed{int(cfg.seed)}"
    out.mkdir(parents=True, exist_ok=True)
    extra = {"encoder": encoder.name}
    if cfg.stage == "bc":
        run_bc(cfg, table, theta_n, out, extra)
    else:
        cards = {c.id: c for c in load_cards(Path(cfg.splits.cards_root), "train")}
        row_card = [cards[recs[s].card_id] for s in table.pair[:, 1].tolist()]
        run_rl(cfg, ocfg, table, problems, row_card, out, extra)


if __name__ == "__main__":
    main()
