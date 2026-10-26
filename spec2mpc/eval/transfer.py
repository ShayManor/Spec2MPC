"""Cross-card transfer: theta* of card c scored by rubric card c' on the same problem, and the go/no-go test.

Every entry uses the oracle finals layout (final_layout / final_gen), so diagonal, default and off-diagonal entries
see identical xi and identical perception noise.
"""
from pathlib import Path
from typing import Dict, List, Mapping, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow as pa  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from torch import Tensor  # noqa: E402

DEFAULT = "default"  # theta_card of the theta_default rows


def transfer_scores(job, theta_star: Mapping[Tuple[str, str], Tensor], cfg: DictConfig, device: torch.device,
                    offset: int) -> Tuple[Tensor, Tensor, List[int]]:
    # pair (p, c') gets theta*_{p, c} with c = the card `offset` places after c' among its task's cards (cyclic)
    # -> J [n_pairs] mean over final_xi, valid [n_pairs] (offset < number of task cards), src card index per pair
    from spec2mpc.baselines.oracle_bo import final_scores  # oracle_bo imports this module

    task_cards: Dict[str, List[int]] = {}
    for _, c in job.pairs:
        cards = task_cards.setdefault(job.cards[c].task, [])
        if c not in cards:
            cards.append(c)
    src, valid, theta = [], [], []
    for p, c in job.pairs:
        cards = task_cards[job.cards[c].task]
        s = cards[(cards.index(c) + offset) % len(cards)]
        src.append(s)
        valid.append(offset < len(cards))
        theta.append(theta_star[(job.problems.problem_id[p], job.cards[s].id)])
    J = final_scores(job, cfg, torch.stack(theta), device).mean(1)  # [n_pairs]
    return J, torch.tensor(valid), src


def transfer_rollouts(job, theta_star: Mapping[Tuple[str, str], Tensor], cfg: DictConfig, device: torch.device,
                      finals: Mapping[str, list]) -> pa.Table:
    # theta_star: (problem_id, card_id) -> [D_THETA]; finals: oracle.parquet columns (diagonal and default rows)
    # -> rows problem_id, task, theta_card, rubric_card, J; one finals-layout call per card offset
    pid = [job.problems.problem_id[p] for p, _ in job.pairs]
    task = [job.problems.task[p] for p, _ in job.pairs]
    rub = [job.cards[c].id for _, c in job.pairs]
    fin = {(p, c): (jo, jd) for p, c, jo, jd in zip(finals["problem_id"], finals["card_id"], finals["J_oracle"],
                                                     finals["J_default"])}
    rows = [(p, t, DEFAULT, c, fin[(p, c)][1]) for p, t, c in zip(pid, task, rub)]
    rows += [(p, t, c, c, fin[(p, c)][0]) for p, t, c in zip(pid, task, rub)]
    n_max = max(sum(1 for c in job.cards if c.task == t) for t in set(task))
    for offset in range(1, n_max):
        J, valid, src = transfer_scores(job, theta_star, cfg, device, offset)
        rows += [(p, t, job.cards[s].id, c, float(j))
                 for p, t, c, s, j, v in zip(pid, task, rub, src, J.tolist(), valid.tolist()) if v]
    names = ("problem_id", "task", "theta_card", "rubric_card", "J")
    return pa.table({n: [r[i] for r in rows] for i, n in enumerate(names)})


def transfer_matrix(table: pa.Table, task_cards: List[str], min_gap: float = 0.05,
                    return_skipped: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    # -> T [C, C]: mean over problems of (J - J_default_col) / (J_oracle_col - J_default_col) for theta*_row under
    # rubric_col; problems with gap <= min_gap are skipped (NaN if all are); optionally the skip counts [C, C]
    J = {(r["problem_id"], r["theta_card"], r["rubric_card"]): r["J"] for r in table.to_pylist()}
    problems = sorted({p for p, _, _ in J})
    C = len(task_cards)
    T = torch.full((C, C), float("nan"), dtype=torch.float64)
    skipped = torch.zeros(C, C, dtype=torch.long)
    for i, a in enumerate(task_cards):
        for j, b in enumerate(task_cards):
            vals = []
            for p in problems:
                if (p, a, b) not in J:
                    continue
                jd, jo = J[(p, DEFAULT, b)], J[(p, b, b)]
                if jo - jd <= min_gap:
                    skipped[i, j] += 1
                    continue
                vals.append((J[(p, a, b)] - jd) / (jo - jd))
            if vals:
                T[i, j] = sum(vals) / len(vals)
    return (T, skipped) if return_skipped else T


def diag_margin(T: Tensor) -> Tensor:
    # T [C, C] -> [C]: T[c, c] - max over c' != c of T[c', c]
    C = T.shape[0]
    off = T.masked_fill(torch.eye(C, dtype=torch.bool), float("-inf"))
    return T.diagonal() - off.max(0).values


def transfer_verdict(margins: Tensor, go_margin: float) -> Tuple[str, int, float]:
    # margins [C] (NaN: card unscoreable) -> verdict, number of finite margins, their median;
    # INCONCLUSIVE when fewer than half the cards have a finite margin, else GO iff median >= go_margin
    ok = torch.isfinite(margins)
    n_valid = int(ok.sum())
    med = float(margins[ok].median()) if n_valid else float("nan")
    if n_valid < margins.numel() / 2 or n_valid == 0:
        return "INCONCLUSIVE", n_valid, med
    return ("GO" if med >= go_margin else "NO-GO"), n_valid, med


def go_no_go(margins: Tensor, go_margin: float) -> bool:
    # True only for a GO verdict (transfer_verdict)
    return transfer_verdict(margins, go_margin)[0] == "GO"


def short_name(card_id: str) -> str:
    return card_id.removeprefix("kart_").removesuffix("_001")


def plot_transfer(T_by_task: Dict[str, Tuple[Tensor, List[str]]], out_png: Path) -> None:
    # one heatmap per task (lap, goto): rows = theta from card, columns = scored by card, color on [0, 1]
    tasks = [t for t in ("lap", "goto") if t in T_by_task]
    sizes = [len(T_by_task[t][1]) for t in tasks]
    fig, axes = plt.subplots(1, len(tasks), figsize=(1.0 + 1.1 * sum(sizes), 0.9 + 1.0 * max(sizes)),
                             gridspec_kw={"width_ratios": sizes, "wspace": 0.45}, squeeze=False)
    cmap = plt.get_cmap("viridis").with_extremes(bad="#e0e0e0")  # NaN: every problem skipped (gap <= min_gap)
    for ax, t in zip(axes[0], tasks):
        T, ids = T_by_task[t]
        T = T.double()
        im = ax.imshow(T.clamp(0, 1).numpy(), cmap=cmap, vmin=0, vmax=1)
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                v = float(T[i, j])
                ax.text(j, i, "n/a" if v != v else f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if (v == v and v < 0.6) else "black")
        names = [short_name(c) for c in ids]
        ax.set_xticks(range(len(ids)), names, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(ids)), names, fontsize=8)
        ax.set_xlabel(f"{t}: scored by card", fontsize=9)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0][0].set_ylabel("theta from card", fontsize=9)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
