"""S1 oracle: per (problem, card), the theta* maximizing the true rubric score J under true batched sim rollouts.

The one baselines module allowed to read rubric content (CLAUDE.md section 11).
"""
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from spec2mpc.eval.transfer import transfer_rollouts
from spec2mpc.mpc.theta import THETA_NAMES, ThetaBatch, default_theta, denormalize_theta, normalize_theta
from spec2mpc.robots.kart import RuntimeConfig
from spec2mpc.rubric.cards import Card, load_cards
from spec2mpc.rubric.scoring import sample_card_conditions, score
from spec2mpc.search.cmaes import BatchedCMAES
from spec2mpc.sim.conditions import Conditions
from spec2mpc.sim.evaluate import rollout_metrics
from spec2mpc.sim.problems import Problems, problem_digest, sample_problems, stable_seed
from spec2mpc.sim.storage import code_digest, git_hash

STATE_FILE = "state.pt"
META_FILE = "meta.json"
# settings a saved CMA state depends on; cmaes.generations may grow on resume
RESUME_KEYS = ("split", "problems", "runtime", "xi_seed", "cmaes.lam", "cmaes.sigma0", "cmaes.xi_per_candidate",
               "cmaes.seed", "sim", "mpc", "robots", "tokenize", "final_xi", "at_k")
FINAL_SOURCES = ("mean", "best", "default")


@dataclass
class OracleJob:
    problems: Problems  # [P]
    cards: List[Card]
    pairs: List[Tuple[int, int]]  # (problem index, card index), every card whose task equals the problem's task


def build_job(cfg: DictConfig) -> OracleJob:
    # cfg: root oracle config -> problems for cfg.split and the cards it optimizes (val: train + val cards)
    split = str(cfg.split)
    assert split in ("train", "val"), f"oracle split must be train or val, got {split}"
    root = Path(cfg.splits.cards_root)
    cards = load_cards(root, "train") + (load_cards(root, "val") if split == "val" else [])
    rt = RuntimeConfig(control_rate_hz=float(cfg.runtime.control_rate_hz), solver_iters=int(cfg.runtime.solver_iters),
                       perception_rate_hz=float(cfg.runtime.perception_rate_hz))
    problems = sample_problems(cfg.problems.n_per_task, cfg, int(cfg.problems[f"seed_{split}"]), rt, split)
    pairs = [(p, c) for p in range(problems.n) for c in range(len(cards)) if problems.task[p] == cards[c].task]
    return OracleJob(problems=problems, cards=cards, pairs=pairs)


def final_conditions(job: OracleJob, pair_problem: int, rubric_card: int, n: int, sim_cfg: DictConfig,
                     seed: int) -> Conditions:
    # -> Conditions [n] for (problem, rubric card); the finals and the transfer matrix share these draws
    gen = torch.Generator().manual_seed(stable_seed(seed, job.problems.problem_id[pair_problem],
                                                    job.cards[rubric_card].id))
    return sample_card_conditions(job.cards[rubric_card], n, sim_cfg, gen)


def final_layout(job: OracleJob, cfg: DictConfig) -> Tuple[Tensor, Conditions]:
    # -> rep long [n_pairs * final_xi] problem index per episode, cond [n_pairs * final_xi] (final_conditions, pair-major)
    nf = int(cfg.final_xi)
    rep = torch.tensor([p for p, _ in job.pairs]).repeat_interleave(nf)
    cond = Conditions.cat([final_conditions(job, p, k, nf, cfg.sim, int(cfg.xi_seed)) for p, k in job.pairs])
    return rep, cond


def final_gen(cfg: DictConfig) -> torch.Generator:
    # a fresh generator per finals call: every call on final_layout sees the same perception noise per env position
    return torch.Generator().manual_seed(stable_seed(int(cfg.xi_seed), "final"))


def final_scores(job: OracleJob, cfg: DictConfig, theta: Tensor, device: torch.device) -> Tensor:
    # theta: [n_pairs, D_THETA] Theta-box values, one per pair -> J [n_pairs, final_xi] float64, one rollout_metrics call
    # on final_layout with final_gen, so calls for different theta are paired episode by episode
    nf = int(cfg.final_xi)
    rep, cond = final_layout(job, cfg)
    robot, scene = job.problems.expand(rep)
    th = ThetaBatch(values=theta.float().repeat_interleave(nf, dim=0))  # [n_pairs * nf, D]
    metrics = rollout_metrics(robot, scene, th, cond, cfg, final_gen(cfg), device).metrics  # {name: [E]} on CPU
    J = torch.empty(len(job.pairs), nf, dtype=torch.float64)
    for i, (_, c) in enumerate(job.pairs):
        J[i] = score(job.cards[c], {k: v[i * nf:(i + 1) * nf] for k, v in metrics.items()})
    return J


def _evaluate(job: OracleJob, cfg: DictConfig, theta: Tensor, cond: List[Conditions], gen: torch.Generator,
              device: torch.device) -> Tensor:
    # theta: [n_pairs, M, D_THETA] Theta-box values; cond: per pair Conditions [X], shared by the M candidates
    # -> J [n_pairs, M, X] float64; all n_pairs * M * X episodes in one rollout_metrics call
    # candidates share xi but not perception noise (drawn per env position from gen): partial common random numbers
    n_pairs, M, D = theta.shape
    X = cond[0].batch_size
    rep = torch.tensor([p for p, _ in job.pairs]).repeat_interleave(M * X)  # [E] problem index
    robot, scene = job.problems.expand(rep)
    th = ThetaBatch(values=theta.float().repeat_interleave(X, dim=1).reshape(-1, D))  # [E, D]
    cd = Conditions.cat([c.index(torch.arange(X).repeat(M)) for c in cond])  # [E], candidate-major per pair
    metrics = rollout_metrics(robot, scene, th, cd, cfg, gen, device).metrics  # {name: [E]} on CPU
    J = torch.empty(n_pairs, M * X, dtype=torch.float64)
    for i, (_, c) in enumerate(job.pairs):
        sl = slice(i * M * X, (i + 1) * M * X)
        J[i] = score(job.cards[c], {k: v[sl] for k, v in metrics.items()})
    return J.view(n_pairs, M, X)


def _resume_view(cfg: DictConfig) -> Dict[str, object]:
    return {k: OmegaConf.to_container(OmegaConf.create({"v": OmegaConf.select(cfg, k)}), resolve=True)["v"]
            for k in RESUME_KEYS}


def _atomic_save(obj: object, path: Path) -> None:
    tmp = path.with_name(f".tmp_{path.name}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_text(text: str, path: Path) -> None:
    tmp = path.with_name(f".tmp_{path.name}")
    tmp.write_text(text)
    os.replace(tmp, path)


def _write_parquet(cols: Dict[str, list], path: Path) -> None:
    tmp = path.with_name(f".tmp_{path.name}")
    pq.write_table(pa.table(cols), tmp)
    os.replace(tmp, path)


def run_oracle(job: OracleJob, cfg: DictConfig, out_dir: Path, device: torch.device, resume: bool = False) -> None:
    # writes {out_dir}/{split}/{oracle.parquet, at_k.parquet, state.pt, config.yaml, problems.pt, meta.json}
    # and, for split train, transfer.parquet (spec2mpc.eval.transfer)
    out = Path(out_dir) / str(cfg.split)
    out.mkdir(parents=True, exist_ok=True)
    c = cfg.cmaes
    lam, xi, n_gen, seed = int(c.lam), int(c.xi_per_candidate), int(c.generations), int(cfg.xi_seed)
    mpc_cfg = cfg.mpc
    assert all(int(k) % lam == 0 for k in cfg.at_k), f"every at_k entry must be a multiple of lam={lam}: {list(cfg.at_k)}"
    n_pairs = len(job.pairs)
    pair_ids = [(job.problems.problem_id[p], job.cards[k].id) for p, k in job.pairs]
    mean0 = normalize_theta(default_theta(n_pairs, mpc_cfg).values, mpc_cfg)  # [n_pairs, D]
    cma = BatchedCMAES(mean0, float(c.sigma0), lam, torch.Generator().manual_seed(int(c.seed)))
    at_k: Dict[int, Tensor] = {}  # k -> CMA mean [n_pairs, D] (normalized) after k / lam generations
    episodes, wall = 0, 0.0
    state_path = out / STATE_FILE
    problems_path = out / "problems.pt"
    digests = problem_digest(job.problems)
    if state_path.exists():
        if not resume:
            raise FileExistsError(f"{state_path} exists; set RESUME=1 to continue it or pick another out_dir")
        st = torch.load(state_path, weights_only=True)
        diff = {k: (st["resume_view"].get(k), v) for k, v in _resume_view(cfg).items() if st["resume_view"].get(k) != v}
        if diff or st["pairs"] != pair_ids:
            raise ValueError(f"{state_path} was made with different settings (saved, current): {diff}; "
                             f"use the original settings or another out_dir")
        saved = torch.load(problems_path, weights_only=True)
        if saved["problem_id"] != list(job.problems.problem_id) or saved["digest"] != digests:
            raise ValueError(f"problems rebuilt for this run differ from {problems_path}; use another out_dir")
        cma.load_state_dict(st["cma"])
        at_k, episodes, wall = st["at_k"], st["episodes"], st["wall_s"]
        if cma.generation >= n_gen and (out / META_FILE).exists():
            print(f"oracle {cfg.split} complete at gen={cma.generation} (RESUME=1)", flush=True)
            return
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    if not state_path.exists():
        # lets downstream stages verify problems rebuilt from config.yaml (sim.problems.load_oracle_problems)
        _atomic_save({"problem_id": list(job.problems.problem_id), "digest": digests}, problems_path)

    def save_state() -> None:
        _atomic_save({"cma": cma.state_dict(), "generation": cma.generation, "at_k": at_k, "episodes": episodes,
                      "wall_s": wall, "pairs": pair_ids, "resume_view": _resume_view(cfg)}, state_path)

    while cma.generation < n_gen:
        t0 = time.perf_counter()
        g = cma.generation + 1
        x = cma.ask()  # [n_pairs, lam, D] normalized, CPU float64
        cond = [sample_card_conditions(job.cards[k], xi, cfg.sim, torch.Generator().manual_seed(
            stable_seed(seed, "gen", g, *pair_ids[i]))) for i, (_, k) in enumerate(job.pairs)]
        gen = torch.Generator().manual_seed(stable_seed(seed, "rollout", g))
        J = _evaluate(job, cfg, denormalize_theta(x, mpc_cfg), cond, gen, device)  # [n_pairs, lam, xi]
        cma.tell(x, J.mean(2))
        if g * lam in [int(k) for k in cfg.at_k]:
            at_k[g * lam] = cma.mean.clone()
        dt = time.perf_counter() - t0
        n_eps = n_pairs * lam * xi
        episodes, wall = episodes + n_eps, wall + dt
        save_state()
        print(f"gen={g} eps={n_eps} wall_s={dt:.1f} eps_per_s={n_eps / dt:.2f} "
              f"mean_best_J={float(cma.best_f.mean()):.4f}", flush=True)

    nf = int(cfg.final_xi)
    fin = pq.read_table(out / "oracle.parquet").to_pydict() if resume and (out / "oracle.parquet").exists() else None
    if fin is not None and fin["n_evals"] == [cma.n_evals] * n_pairs:
        # a resumed run that stopped after the finals of its last generation: reuse them (same layout and
        # generator, so identical)
        cols = fin
        if list(zip(cols["problem_id"], cols["card_id"])) != [tuple(pk) for pk in pair_ids]:
            raise ValueError(f"{out / 'oracle.parquet'} pairs differ from this run; use another out_dir")
        theta_star = torch.tensor([cols[n] for n in THETA_NAMES], dtype=torch.float32).T  # [n_pairs, D]
        dt = 0.0
        episodes += n_pairs * 3 * nf
    else:
        # finals: theta_mean, theta_best, theta_default, one paired call each (same layout, xi and perception noise)
        t0 = time.perf_counter()
        th = torch.stack([denormalize_theta(cma.mean, mpc_cfg), denormalize_theta(cma.best_x, mpc_cfg),
                          default_theta(n_pairs, mpc_cfg).values.double()], dim=1)  # [n_pairs, 3, D]
        Jf = torch.stack([final_scores(job, cfg, th[:, j], device).mean(1) for j in range(3)], 1)  # [n_pairs, 3]
        src = Jf.argmax(1)  # [n_pairs]; ties go to the first of mean, best, default
        theta_star = th[torch.arange(n_pairs), src].float()  # [n_pairs, D]
        dt = time.perf_counter() - t0
        episodes, wall = episodes + n_pairs * 3 * nf, wall + dt

        cols: Dict[str, list] = {"problem_id": [a for a, _ in pair_ids], "card_id": [b for _, b in pair_ids],
                                 "task": [job.problems.task[p] for p, _ in job.pairs]}
        cols.update({n: theta_star[:, j].tolist() for j, n in enumerate(THETA_NAMES)})
        cols.update(theta_star_source=[FINAL_SOURCES[int(i)] for i in src],
                    J_oracle=Jf.max(1).values.tolist(), J_default=Jf[:, 2].tolist(), J_mean=Jf[:, 0].tolist(),
                    J_best=Jf[:, 1].tolist(), n_evals=[cma.n_evals] * n_pairs)
        _write_parquet(cols, out / "oracle.parquet")
    if str(cfg.split) == "train":
        t1 = time.perf_counter()
        ts = {pk: theta_star[i] for i, pk in enumerate(pair_ids)}
        tr = transfer_rollouts(job, ts, cfg, device, cols)
        _write_parquet(tr.to_pydict(), out / "transfer.parquet")
        n_tr = (max(sum(1 for k in job.cards if k.task == t) for t in set(cols["task"])) - 1) * n_pairs * nf
        episodes, wall = episodes + n_tr, wall + time.perf_counter() - t1
        print(f"transfer eps={n_tr} wall_s={time.perf_counter() - t1:.1f} rows={tr.num_rows}", flush=True)
    ks = sorted(at_k)
    kc: Dict[str, list] = {"problem_id": [a for _ in ks for a, _ in pair_ids],
                           "card_id": [b for _ in ks for _, b in pair_ids], "k": [k for k in ks for _ in pair_ids]}
    if ks:
        th_k = torch.cat([denormalize_theta(at_k[k], mpc_cfg) for k in ks]).float()  # [len(ks) * n_pairs, D]
        kc.update({n: th_k[:, j].tolist() for j, n in enumerate(THETA_NAMES)})
    else:
        kc.update({n: [] for n in THETA_NAMES})
    _write_parquet(kc, out / "at_k.parquet")
    save_state()
    meta = {"git_hash": git_hash(), "code_digest": code_digest(), "host": socket.gethostname(), "device": str(device),
            "wall_s": wall, "episodes": episodes, "eps_per_s": episodes / max(wall, 1e-9), "n_pairs": n_pairs,
            "generations": cma.generation}
    _write_text(json.dumps(meta, indent=1), out / META_FILE)
    print(f"finals eps={n_pairs * 3 * nf} wall_s={dt:.1f} mean_J_oracle={sum(cols['J_oracle']) / n_pairs:.4f} "
          f"mean_J_default={sum(cols['J_default']) / n_pairs:.4f}", flush=True)
