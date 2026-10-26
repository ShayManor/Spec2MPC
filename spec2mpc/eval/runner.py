"""Evaluation runner: val eval units, a fixed episode layout shared by every method, true rollouts scored by the true
rubric, and the theta of each method (non-learned baselines and the learned inference modes).

Every method sees identical problems, statements, condition draws (xi), rollout generator seed and batch layout.
Learned modules receive tensors only; rubric content stays in this module and spec2mpc/rubric.
"""
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from spec2mpc.baselines.oracle_reader import load_oracle_thetas
from spec2mpc.baselines.simple import MeanTheta, NNRetrieval, default_theta_n
from spec2mpc.eval.metrics import attach_jhat, critic_gap, critic_target_asinh, floor_violation_rate, floor_violations
from spec2mpc.model.data import TASK_IDS, load_theta_star, normalize_xi, xi_from_conditions
from spec2mpc.model.policy import PolicyContext, draw_score, expected_score
from spec2mpc.mpc.theta import ThetaBatch, denormalize_theta, normalize_theta, theta_bounds
from spec2mpc.rubric.cards import Card
from spec2mpc.rubric.scoring import card_max_score, sample_card_conditions, score, tier_passes
from spec2mpc.rubric.splits import statement_records
from spec2mpc.search.inference import amortized, bon_pred, bon_true, refine
from spec2mpc.sim.conditions import Conditions, sample_conditions
from spec2mpc.sim.evaluate import rollout_metrics
from spec2mpc.sim.problems import Problems, stable_seed

SUMMARY_FILE = "summary.parquet"
KINDS = {"val_paraphrase": "paraphrase", "val_intent": "intent"}
SINGLE_SHOT = ("default", "oracle", "mean_theta", "nn_retrieval", "slot_only", "applv_lang", "amortized")  # K = 0
LEARNED = ("slot_only", "applv_lang", "amortized", "bon_pred", "refine", "bon_true")  # one result per training seed


@dataclass(frozen=True)
class EvalUnit:
    statement: str
    card_id: str
    task: str
    kind: str  # "paraphrase" | "intent"


def val_units(split_cfg: DictConfig) -> List[EvalUnit]:
    # -> the val_paraphrase and val_intent statements of statement_records, in record order
    return [EvalUnit(r.text, r.card_id, r.task, KINDS[r.role]) for r in statement_records(split_cfg) if r.role in KINDS]


@dataclass
class EvalLayout:
    unit_idx: Tensor  # [E] long
    problem_idx: Tensor  # [E] long
    cond: Conditions  # [E]
    gen_seed: int  # rollout generator seed, the same for every method
    unit_card: List[str]  # [U] card id of each unit

    def pairs(self) -> Tuple[Tensor, Tensor]:
        # -> unit, problem [n_pairs] long in episode order (each pair holds a contiguous run of xi draws)
        key = self.unit_idx * (int(self.problem_idx.max()) + 1) + self.problem_idx
        k = torch.unique_consecutive(key)
        first = torch.searchsorted(key, k)  # key is sorted (unit-major, then problem)
        return self.unit_idx[first], self.problem_idx[first]


def make_layout(units: Sequence[EvalUnit], problems: Problems, cards: Mapping[str, Card], n_xi: int, seed: int,
                sim_cfg: DictConfig) -> EvalLayout:
    # every (unit, problem) with matching task, n_xi card-condition draws seeded by (seed, card, problem), so the
    # statements of one card share xi; episodes ordered unit-major, then problem, then xi
    u_idx, p_idx, conds = [], [], []
    for u, unit in enumerate(units):
        for p in range(problems.n):
            if problems.task[p] != unit.task:
                continue
            gen = torch.Generator().manual_seed(stable_seed(seed, unit.card_id, problems.problem_id[p]))
            conds.append(sample_card_conditions(cards[unit.card_id], n_xi, sim_cfg, gen))
            u_idx += [u] * n_xi
            p_idx += [p] * n_xi
    return EvalLayout(unit_idx=torch.tensor(u_idx), problem_idx=torch.tensor(p_idx), cond=Conditions.cat(conds),
                      gen_seed=stable_seed(seed, "rollout"), unit_card=[u.card_id for u in units])


def run_thetas(theta_n: Tensor, layout: EvalLayout, problems: Problems, cards: Mapping[str, Card], cfg: DictConfig,
               device: torch.device) -> Dict[str, Tensor]:
    # theta_n [U, P, 14] normalized (entries of off-task pairs unused); cfg: root config with sim, mpc, robots
    # -> per (unit, problem) pair: unit_idx, problem_idx [n_pairs], J [n_pairs] (mean over xi), tier_rate
    # [n_pairs, max_tiers] (NaN past a card's tiers), boundary_violation_rate, completion_rate [n_pairs];
    # one rollout_metrics call with a fixed generator, so every method gets identical perception noise
    theta = denormalize_theta(theta_n[layout.unit_idx, layout.problem_idx].float(), cfg.mpc)  # [E, 14]
    robot, scene = problems.expand(layout.problem_idx)
    m = rollout_metrics(robot, scene, ThetaBatch(values=theta), layout.cond, cfg,
                        torch.Generator().manual_seed(layout.gen_seed), device).metrics  # {name: [E]} on CPU
    pu, pp = layout.pairs()
    n = pu.numel()
    n_xi = layout.unit_idx.numel() // n
    max_t = max(len(cards[c].rubric.tiers) for c in layout.unit_card)
    J = torch.empty(n, dtype=torch.float64)
    tier = torch.full((n, max_t), float("nan"), dtype=torch.float64)
    for i in range(n):
        sl = slice(i * n_xi, (i + 1) * n_xi)
        card = cards[layout.unit_card[int(pu[i])]]
        mi = {k: v[sl] for k, v in m.items()}
        J[i] = score(card, mi).mean()
        tp = tier_passes(card, mi).double().mean(0)  # [n_tiers]
        tier[i, :tp.numel()] = tp
    per_pair = lambda x: x.double().view(n, n_xi).mean(1)  # noqa: E731
    return {"unit_idx": pu, "problem_idx": pp, "J": J, "tier_rate": tier,
            "boundary_violation_rate": per_pair(m["boundary_violations"] > 0),
            "completion_rate": per_pair(m["completion"])}


def result_table(method: str, K: int, seed: int, units: Sequence[EvalUnit], problems: Problems,
                 cards: Mapping[str, Card], out: Dict[str, Tensor], pred_score: Tensor, gap: Tensor,
                 pred_mean: Tensor, gap_mean: Tensor, floor_viol: Tensor) -> pa.Table:
    # out: run_thetas output; pred_score, gap (critic mean - lam * std) and pred_mean, gap_mean (ensemble mean)
    # [n_pairs], NaN for methods without a critic; floor_viol [n_pairs] bool
    # -> result rows; Jhat and valid are NaN / False here and filled by metrics.attach_jhat
    pu, pp = out["unit_idx"].tolist(), out["problem_idx"].tolist()
    J = out["J"].tolist()
    max_score = [card_max_score(cards[units[u].card_id]) for u in pu]
    cols = {"method": [method] * len(pu), "K": [int(K)] * len(pu), "seed": [int(seed)] * len(pu), "unit": pu,
            "card_id": [units[u].card_id for u in pu], "kind": [units[u].kind for u in pu],
            "problem_id": [problems.problem_id[p] for p in pp], "J": J, "J_norm": [j / s for j, s in zip(J, max_score)],
            "Jhat": [float("nan")] * len(pu), "valid": [False] * len(pu),
            "pred_score": pred_score.double().tolist(), "critic_gap": gap.double().tolist(),
            "pred_mean": pred_mean.double().tolist(), "critic_gap_mean": gap_mean.double().tolist(),
            "boundary_violation_rate": out["boundary_violation_rate"].tolist(),
            "completion_rate": out["completion_rate"].tolist(), "floor_violation": floor_viol.tolist()}
    cols.update({f"tier_rate_{t}": out["tier_rate"][:, t].tolist() for t in range(out["tier_rate"].shape[1])})
    return pa.table(cols)


def settings(cfg: DictConfig) -> List[Tuple[str, int, int]]:
    # -> (method, K, seed) for cfg.methods; default and oracle first, non-learned methods at seed 0 only
    ms = sorted(cfg.methods, key=lambda m: (m not in ("default", "oracle"), m != "default"))
    out = []
    for m in ms:
        ks = [0] if m in SINGLE_SHOT else [int(k) for k in cfg.budgets[m]]
        seeds = [int(s) for s in cfg.seeds] if m in LEARNED else [0]
        out += [(m, k, s) for s in seeds for k in ks]
    return out


@dataclass
class EvalContext:
    # everything a method needs to produce theta for the eval pairs; built once per run
    units: List[EvalUnit]
    problems: Problems  # val oracle problems
    layout: EvalLayout
    cards: Dict[str, Card]
    rcfg: DictConfig  # val oracle run config: sim, mpc, robots of every rollout
    cfg: DictConfig  # eval config
    device: torch.device
    unit_emb: Tensor  # [U, d_text]
    unit_slots: Tensor  # [U, d_slot]
    train_emb: Tensor  # [S, d_text] train statements
    train_card: List[str]  # [S]
    encoder_name: str
    robot_tok: Tensor  # [P, 6, 24]
    loaders: Dict[str, Callable[[int], Tuple[torch.nn.Module, DictConfig, dict]]]  # name -> seed -> checkpoint

    @property
    def pair_idx(self) -> Tuple[Tensor, Tensor]:
        return self.layout.pairs()

    def context(self) -> PolicyContext:
        # -> PolicyContext [n_pairs] on the eval device
        pu, pp = self.pair_idx
        return PolicyContext(self.robot_tok[pp], self.unit_emb[pu], self.unit_slots[pu],
                             self.problems.path_tok[pp].float()).to(self.device)

    def model_xi(self) -> Tensor:
        # -> xi_n [n_pairs, n_xi_model, 6] from the global prior, the same for every learned method
        n = self.pair_idx[0].numel()
        X = int(self.cfg.n_xi_model)
        c = sample_conditions(n * X, self.rcfg.sim, torch.Generator().manual_seed(stable_seed(int(self.cfg.xi_seed), "model_xi")))
        return normalize_xi(xi_from_conditions(c), self.rcfg.sim).view(n, X, -1).to(self.device)


def _check_box(ctx: EvalContext, ck_cfg: DictConfig, what: str) -> None:
    # a learned model's normalized theta must use the same Theta box as the rollouts
    if "mpc" in ck_cfg and not all(torch.equal(a, b) for a, b in zip(theta_bounds(ck_cfg.mpc), theta_bounds(ctx.rcfg.mpc))):
        raise ValueError(f"{what} was trained with a different theta box than the oracle run")


def _load(ctx: EvalContext, name: str, seed: int) -> Tuple[torch.nn.Module, DictConfig]:
    model, ck_cfg, extra = ctx.loaders[name](seed)
    if "encoder" in extra and extra["encoder"] != ctx.encoder_name:
        raise ValueError(f"{name} seed {seed} was trained with text encoder {extra['encoder']}, eval uses {ctx.encoder_name}")
    _check_box(ctx, ck_cfg, f"{name} seed {seed}")
    return model.to(ctx.device).eval(), ck_cfg


def _oracle_theta(ctx: EvalContext, k: Optional[int]) -> Tensor:
    # -> [n_pairs, 14] normalized theta* (k None) or CMA mean at k for each (val problem, unit card)
    th = load_oracle_thetas(Path(ctx.cfg.oracle_root) / "val", k)
    pu, pp = ctx.pair_idx
    keys = [(ctx.problems.problem_id[p], ctx.units[u].card_id) for u, p in zip(pu.tolist(), pp.tolist())]
    missing = [kk for kk in keys if kk not in th]
    if missing:
        raise KeyError(f"val oracle has no theta for {missing[:4]}")
    return normalize_theta(torch.stack([th[kk] for kk in keys]), ctx.rcfg.mpc)


def _train_theta_star(ctx: EvalContext) -> Tuple[List[str], List[str], Tensor]:
    # -> card_id, task [R] and normalized theta* [R, 14] of the train oracle (theta columns and ids only)
    d = Path(ctx.cfg.oracle_root) / "train"
    _, cid, task, theta = load_theta_star(d / "oracle.parquet")
    return cid, task, normalize_theta(theta, OmegaConf.load(d / "config.yaml").mpc)


def _bon_true_rollout(ctx: EvalContext, n: int, seed: int) -> Callable[[Tensor], Tensor]:
    # -> theta_n [n_pairs * n, 14] (pair-major) -> phi [n_pairs * n, bon_true_xi, 20], one true rollout per
    # global-prior draw (the draws of a pair are shared by its candidates) on the pair's problem
    pu, pp = ctx.pair_idx
    X = int(ctx.cfg.bon_true_xi)
    xi_seed = int(ctx.cfg.xi_seed)

    def rollout_phi(theta_n: Tensor) -> Tensor:
        B = theta_n.shape[0]
        conds = [sample_conditions(X, ctx.rcfg.sim, torch.Generator().manual_seed(
            stable_seed(xi_seed, "bon_true", ctx.problems.problem_id[p]))) for p in pp.tolist()]
        cond = Conditions.cat([c.index(torch.arange(X).repeat(n)) for c in conds])  # [B * X], pair-major
        rep = pp.repeat_interleave(n * X)
        robot, scene = ctx.problems.expand(rep)
        th = ThetaBatch(values=denormalize_theta(theta_n.detach().cpu().float(), ctx.rcfg.mpc).repeat_interleave(X, 0))
        gen = torch.Generator().manual_seed(stable_seed(xi_seed, "bon_true_rollout", n, seed))
        phi = rollout_metrics(robot, scene, th, cond, ctx.rcfg, gen, ctx.device).phi  # [B * X, 20]
        return phi.view(B, X, -1).to(ctx.device)
    return rollout_phi


def method_thetas(ctx: EvalContext, method: str, K: int, seed: int) -> Tuple[Tensor, Tensor, Tensor, Optional[bool]]:
    # -> theta_n [n_pairs, 14], pred_score [n_pairs] (critic mean - lam * std, as the mode returns it) and pred_mean
    # [n_pairs] (critic ensemble mean of the same prediction), NaN without a critic, on CPU; and whether the critic
    # predicts asinh(J / card_max_score) (None without a critic)
    pu, pp = ctx.pair_idx
    n = pu.numel()
    nan = torch.full((n,), float("nan"))
    task = torch.tensor([TASK_IDS[ctx.units[u].task] for u in pu.tolist()])
    if method == "default":
        return default_theta_n(n, ctx.rcfg.mpc), nan, nan, None
    if method == "oracle":
        return _oracle_theta(ctx, None), nan, nan, None
    if method == "oracle_bo":
        return _oracle_theta(ctx, K), nan, nan, None
    if method == "mean_theta":
        _, t, th = _train_theta_star(ctx)
        return MeanTheta().fit(th, torch.tensor([TASK_IDS[x] for x in t])).predict(task), nan, nan, None
    if method == "nn_retrieval":
        cid, t, th = _train_theta_star(ctx)
        card_theta = {c: th[[i for i, x in enumerate(cid) if x == c]].mean(0) for c in set(cid)}
        card_task = dict(zip(cid, t))
        rows = [i for i, c in enumerate(ctx.train_card) if c in card_theta]
        nn = NNRetrieval().fit(ctx.train_emb[rows], torch.tensor([TASK_IDS[card_task[ctx.train_card[i]]] for i in rows]),
                               torch.stack([card_theta[ctx.train_card[i]] for i in rows]))
        return nn.predict(ctx.unit_emb[pu], task), nan, nan, None
    pctx = ctx.context()
    if method == "slot_only":
        policy, _ = _load(ctx, "policy_zero", seed)
        with torch.no_grad():
            return policy.amortized(pctx).cpu(), nan, nan, None
    if method == "applv_lang":
        model, _ = _load(ctx, "applv", seed)
        with torch.no_grad():
            return model(pctx).cpu(), nan, nan, None
    policy, _ = _load(ctx, "policy_full", seed)
    W, _ = _load(ctx, "world_model", seed)
    V, v_cfg = _load(ctx, "critic", seed)
    asinh = critic_target_asinh(v_cfg)
    lam = float(ctx.cfg.lam)
    gen = torch.Generator().manual_seed(stable_seed(seed, method, K))
    xi = ctx.model_xi()
    if method in ("amortized", "bon_pred", "refine"):
        if method == "amortized":
            theta, s = amortized(policy, W, V, pctx, xi, lam)
        elif method == "bon_pred":
            theta, s = bon_pred(policy, W, V, pctx, K, xi, lam, gen)
        else:
            r = ctx.cfg.refine
            theta, s = refine(policy, W, V, pctx, K, float(r.lr), xi, lam, float(r.trust_radius))
        with torch.no_grad():  # lam = 0: ensemble mean of the same V(W(theta)) prediction
            m = expected_score(W, V, pctx.robot_tok, pctx.path_tok, theta.detach(), xi, pctx.text_emb, pctx.slots, 0.0)
    elif method == "bon_true":
        rollout = _bon_true_rollout(ctx, K, seed)
        cache: List[Tensor] = []

        def cached(th: Tensor) -> Tensor:
            cache.append(rollout(th))  # [n_pairs * K, bon_true_xi, 20]
            return cache[-1]
        theta, s = bon_true(policy, V, pctx, K, cached, lam, gen)
        rep = lambda t: t.repeat_interleave(K, dim=0)  # noqa: E731
        with torch.no_grad():  # the selected candidate is the argmax of the lam score on the cached true phi
            pen = draw_score(V, rep(pctx.text_emb), rep(pctx.slots), cache[0], lam).view(n, K)
            ens = draw_score(V, rep(pctx.text_emb), rep(pctx.slots), cache[0], 0.0).view(n, K)
            m = ens[torch.arange(n), pen.argmax(1).to(ens.device)]
    else:
        raise ValueError(f"unknown method {method}")
    return theta.detach().cpu(), s.detach().cpu().float(), m.detach().cpu().float(), asinh


def evaluate_setting(ctx: EvalContext, method: str, K: int, seed: int) -> pa.Table:
    # theta for (method, K, seed) -> floor check (asserted 0) -> true rollouts on the shared layout -> result rows
    theta, pred, pred_mean, asinh = method_thetas(ctx, method, K, seed)
    pu, pp = ctx.pair_idx
    params = ctx.problems.robot.params.index(pp)
    rate = floor_violation_rate(theta, params, ctx.rcfg.mpc)
    assert rate == 0.0, f"{method} K={K} seed={seed}: safety-floor violation rate {rate} (must be 0)"
    full = torch.zeros(len(ctx.units), ctx.problems.n, theta.shape[1])
    full[pu, pp] = theta.float()
    out = run_thetas(full, ctx.layout, ctx.problems, ctx.cards, ctx.rcfg, ctx.device)
    J_norm = out["J"] / torch.tensor([card_max_score(ctx.cards[ctx.units[u].card_id]) for u in pu.tolist()],
                                     dtype=torch.float64)
    nan = torch.full_like(J_norm, float("nan"))
    gap = nan if asinh is None else critic_gap(pred.double(), J_norm, asinh)
    gap_mean = nan if asinh is None else critic_gap(pred_mean.double(), J_norm, asinh)
    return result_table(method, K, seed, ctx.units, ctx.problems, ctx.cards, out, pred, gap, pred_mean, gap_mean,
                        floor_violations(theta, params, ctx.rcfg.mpc))


def anchor_files(group: str) -> Tuple[str, str]:
    # -> the default and oracle result files of an eval group; rows of the group are normalized by these only
    return f"default_0_0_{group}.parquet", f"oracle_0_0_{group}.parquet"


def result_files(out: Path, n_xi: int, xi_seed: int) -> Tuple[List[Path], List[Path]]:
    # -> result files in out whose .json records this n_xi and xi_seed, and the others (no json, or other settings);
    # temp files and the summary are not results
    keep, skipped = [], []
    for p in sorted(out.glob("*.parquet")):
        if p.name.startswith(".") or p.name == SUMMARY_FILE:
            continue
        js = p.with_suffix(".json")
        c = json.loads(js.read_text()).get("config", {}) if js.exists() else {}
        ok = c.get("n_xi") == n_xi and c.get("xi_seed") == xi_seed
        (keep if ok else skipped).append(p)
    return keep, skipped


def write_table(table: pa.Table, path: Path, tmp_tag: str = "") -> None:
    # atomic parquet write; tmp_tag keeps concurrent writers of the same path off each other's temp file
    tmp = path.with_name(f".tmp_{tmp_tag}{path.name}")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def run_settings(ctx: EvalContext, todo: Sequence[Tuple[str, int, int]], out: Path, min_gap: float,
                 meta: dict, group: str, tags: Mapping[str, str]) -> List[str]:
    # evaluates each (method, K, seed) not yet in out into {method}_{K}_{seed}_{group}.parquet (+ .json with wall
    # time, meta and tags); every row carries anchor_group = group and the tags (device name, code digest) as
    # columns; a failing setting is logged with its traceback and skipped -> names of the failed settings
    failed = []
    names = anchor_files(group)
    for method, K, seed in todo:
        f = out / f"{method}_{K}_{seed}_{group}.parquet"
        if f.exists():
            print(f"skip {f.name} (exists)", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            tab = evaluate_setting(ctx, method, K, seed)
        except Exception:  # noqa: BLE001 - one broken setting must not stop the others
            print(f"FAILED {f.stem}:\n{traceback.format_exc()}", flush=True)
            failed.append(f.stem)
            continue
        for k, v in {"anchor_group": group, **tags}.items():
            tab = tab.append_column(k, pa.array([str(v)] * tab.num_rows, pa.string()))
        anchors = [pq.read_table(out / a) for a in names if (out / a).exists() and a != f.name]
        if {method} | {a.column("method")[0].as_py() for a in anchors} >= {"default", "oracle"}:
            # Jhat now when both anchors exist; summarize recomputes it over every file either way
            tab = attach_jhat(pa.concat_tables(anchors + [tab]), min_gap).slice(sum(a.num_rows for a in anchors))
        write_table(tab, f)
        dt = time.perf_counter() - t0
        f.with_suffix(".json").write_text(json.dumps({"method": method, "K": K, "seed": seed, "wall_s": dt, **meta,
                                                      "anchor_group": group, **tags}, indent=1))
        d = tab.to_pydict()
        print(f"{f.name}: mean J={sum(d['J']) / len(d['J']):.4f} wall_s={dt:.1f}", flush=True)
    return failed
