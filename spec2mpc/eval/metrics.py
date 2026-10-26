"""Evaluation metrics: normalized score, safety-floor check, critic gap, and the per-(method, K, kind) summary."""
from typing import Dict, List, Mapping, Optional, Tuple

import pyarrow as pa
import torch
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.eval.stats import bootstrap_ci
from spec2mpc.mpc.adapter import adapt
from spec2mpc.mpc.theta import ThetaBatch, denormalize_theta
from spec2mpc.robots.kart import KartParams

CI_ALPHA = 0.05  # 95% bootstrap CI (CLAUDE.md section 8.2)
ANCHORS = ("default", "oracle")  # methods whose J are J_default and J_oracle


def normalized_score(J: Tensor, J_default: Tensor, J_oracle: Tensor, min_gap: float) -> Tuple[Tensor, Tensor]:
    # J, J_default, J_oracle [N] -> Jhat [N] (NaN where invalid), valid [N] bool (J_oracle - J_default > min_gap)
    gap = J_oracle - J_default
    valid = gap > min_gap
    jhat = torch.where(valid, (J - J_default) / torch.where(valid, gap, torch.ones_like(gap)), torch.full_like(J, float("nan")))
    return jhat, valid


def floor_violations(theta_n: Tensor, params: KartParams, mpc_cfg: DictConfig) -> Tensor:
    # theta_n [N, 14], params [N] -> [N] bool: the adapted MPC weights break a safety floor
    w = adapt(ThetaBatch(values=denormalize_theta(theta_n.float(), mpc_cfg)), params, mpc_cfg)
    fl = mpc_cfg.floors
    return ((w.margin_m < float(fl.m_floor_m)) | (w.rho_slack < float(fl.rho_slack_min))
            | (w.v_cap_mps < float(fl.c_vmax) * params.v_max_mps)
            | (w.a_lat_cap_mps2 < float(fl.c_alat) * params.a_lat_max_mps2)
            | (w.a_long_cap_mps2 < float(fl.c_along) * params.a_long_max_mps2))


def floor_violation_rate(theta_n: Tensor, problems_params: KartParams, mpc_cfg: DictConfig) -> float:
    # theta_n [N, 14], problems_params [N] -> fraction of rows breaking a floor (must be 0.0)
    return float(floor_violations(theta_n, problems_params, mpc_cfg).float().mean())


def critic_target_asinh(critic_cfg: DictConfig) -> bool:
    # True when the critic was trained on asinh(J / card_max_score); the flag must be in its checkpoint config
    if "target_asinh" not in critic_cfg:
        raise KeyError("critic checkpoint config has no target_asinh flag; cannot place the critic gap in its space")
    return bool(critic_cfg.target_asinh)


def critic_gap(pred_score: Tensor, J_norm: Tensor, target_asinh: bool) -> Tensor:
    # pred_score [N] (critic space), J_norm [N] = J / card_max_score -> [N] predicted minus true, in the critic's space
    return pred_score - (torch.asinh(J_norm) if target_asinh else J_norm)


def _anchor_keys(d: Dict[str, list]) -> List[tuple]:
    # -> per row (anchor_group, device_name, code_digest, unit, problem_id); None for rows written without them, so
    # a group requeued onto another GPU model or code tree never pairs with the first job's anchors
    none = [None] * len(d["unit"])
    groups, dev, dig = (d.get(k, none) for k in ("anchor_group", "device_name", "code_digest"))
    return list(zip(groups, dev, dig, d["unit"], d["problem_id"]))


def drop_unanchored(results: pa.Table) -> Tuple[pa.Table, Dict[tuple, int]]:
    # -> rows whose (anchor_group, unit, problem_id) has both a default and an oracle row, and the count of the other
    # rows per (method, K, kind, anchor_group); anchors are never paired across groups
    d = results.to_pydict()
    key = _anchor_keys(d)
    have = {m: {k for mm, k in zip(d["method"], key) if mm == m} for m in ANCHORS}
    ok = [all(k in have[m] for m in ANCHORS) for k in key]
    dropped: Dict[tuple, int] = {}
    for i, o in enumerate(ok):
        if not o:
            k = (d["method"][i], d["K"][i], d["kind"][i], key[i][0])
            dropped[k] = dropped.get(k, 0) + 1
    return results.filter(pa.array(ok, pa.bool_())), dropped


def attach_jhat(results: pa.Table, min_gap: float) -> pa.Table:
    # results rows of every method -> same rows with Jhat and valid from the default and oracle rows of the same
    # (anchor_group, unit, problem_id); raises if an anchor row is missing (drop_unanchored removes those rows first)
    d = results.to_pydict()
    key = _anchor_keys(d)
    anchor: Dict[str, Dict[tuple, float]] = {m: {} for m in ANCHORS}
    for m, k, j in zip(d["method"], key, d["J"]):
        if m in anchor:
            anchor[m][k] = j
    missing = sorted({(m, k) for k in key for m in ANCHORS if k not in anchor[m]})
    if missing:
        raise ValueError(f"no anchor rows for {missing[:4]} (+{max(len(missing) - 4, 0)}); run default and oracle first")
    J = torch.tensor(d["J"], dtype=torch.float64)
    jd = torch.tensor([anchor["default"][k] for k in key], dtype=torch.float64)
    jo = torch.tensor([anchor["oracle"][k] for k in key], dtype=torch.float64)
    jhat, valid = normalized_score(J, jd, jo, min_gap)
    idx = results.column_names.index
    out = results.set_column(idx("Jhat"), "Jhat", pa.array(jhat.tolist(), pa.float64()))
    return out.set_column(idx("valid"), "valid", pa.array(valid.tolist(), pa.bool_()))


def _ci(values: Tensor, groups: List[str], n_boot: int, seed: int) -> Tuple[float, float, float]:
    # values [N], card id per row -> (mean, lo, hi) of the card-weighted mean: rows are averaged within each card
    # (statements, problems and seeds pooled), then cards are averaged with equal weight and resampled by the
    # bootstrap; NaN when empty
    if values.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    codes = {c: i for i, c in enumerate(sorted(set(groups)))}
    g = torch.tensor([codes[c] for c in groups])
    v = values.double()
    card_mean = torch.zeros(len(codes), dtype=torch.float64).index_add_(0, g, v) / torch.bincount(g).double()  # [G]
    return bootstrap_ci(card_mean, torch.arange(len(codes)), n_boot, CI_ALPHA, torch.Generator().manual_seed(seed))


def summarize(results: pa.Table, n_boot: int, seed: int, unanchored: Optional[Mapping[tuple, int]] = None) -> pa.Table:
    # results rows (with Jhat and valid attached) -> one row per (method, K, kind): card-weighted mean Jhat and
    # critic gaps (penalized: mean - lam * std; ens: ensemble mean) with a bootstrap CI over cards (all statements,
    # problems and training seeds of a card pooled in its card mean), n_cards behind the Jhat CI, mean tier success
    # per tier index, n_excluded (rows with J_oracle - J_default <= min_gap), n_unanchored (rows dropped for lack of
    # same-group anchors, from drop_unanchored), floor and boundary violation rates
    d = results.to_pydict()
    tiers = sorted((c for c in d if c.startswith("tier_rate_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    groups: Dict[tuple, List[int]] = {}
    for i, k in enumerate(zip(d["method"], d["K"], d["kind"])):
        groups.setdefault(k, []).append(i)
    out = []
    for (method, K, kind), rows in sorted(groups.items()):
        col = lambda n: torch.tensor([d[n][i] for i in rows], dtype=torch.float64)  # noqa: E731
        cards = [d["card_id"][i] for i in rows]
        valid = torch.tensor([bool(d["valid"][i]) for i in rows])
        jhat, gap, gap_ens = col("Jhat"), col("critic_gap"), col("critic_gap_mean")
        ok_gap, ok_ens = torch.isfinite(gap), torch.isfinite(gap_ens)
        pick = lambda ok: [c for c, v in zip(cards, ok.tolist()) if v]  # noqa: E731
        jm, jl, jh = _ci(jhat[valid], pick(valid), n_boot, seed)
        gm, gl, gh = _ci(gap[ok_gap], pick(ok_gap), n_boot, seed)
        em, el, eh = _ci(gap_ens[ok_ens], pick(ok_ens), n_boot, seed)
        rec = {"method": method, "K": int(K), "kind": kind, "n_rows": len(rows), "n_cards": len(set(pick(valid))),
               "n_seeds": len({d["seed"][i] for i in rows}), "n_excluded": int((~valid).sum()),
               "n_unanchored": sum(v for (m, k, kd, _), v in (unanchored or {}).items()
                                   if (m, k, kd) == (method, K, kind)),
               "Jhat_mean": jm, "Jhat_lo": jl, "Jhat_hi": jh, "gap_mean": gm, "gap_lo": gl, "gap_hi": gh,
               "gap_ens_mean": em, "gap_ens_lo": el, "gap_ens_hi": eh,
               "floor_violation_rate": float(col("floor_violation").mean()),
               "boundary_violation_rate": float(col("boundary_violation_rate").mean()),
               "completion_rate": float(col("completion_rate").mean())}
        rec.update({f"tier_{t.rsplit('_', 1)[1]}": float(col(t).nanmean()) for t in tiers})
        out.append(rec)
    return pa.Table.from_pylist(out)
