from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from hydra import compose, initialize_config_dir

from spec2mpc.baselines.oracle_bo import OracleJob, build_job, run_oracle
from spec2mpc.eval.transfer import (diag_margin, go_no_go, plot_transfer, transfer_matrix, transfer_rollouts,
                                    transfer_scores, transfer_verdict)
from spec2mpc.mpc.theta import THETA_NAMES
from spec2mpc.rubric.cards import load_cards

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
TINY = [  # tests/test_oracle.py TINY
    "sim.episode.timeout_cap_lap_s=0.3", "sim.episode.timeout_cap_goto_s=0.3", "sim.grid.res_m=0.5",
    "runtime.solver_iters=2", "problems.n_per_task.lap=1", "problems.n_per_task.goto=1",
    "cmaes.lam=4", "cmaes.generations=1", "cmaes.xi_per_candidate=2", "final_xi=2", "at_k=[4]",
]


def _table(rows):
    return pa.table({k: [r[i] for r in rows] for i, k in enumerate(("problem_id", "task", "theta_card", "rubric_card", "J"))})


def _two_card_rows(pid="p0", jo=(2.0, 4.0), jd=(0.0, 0.0), ab=1.0, ba=1.0):
    # theta of card a scores ab under b; theta of b scores ba under a
    return [(pid, "lap", "default", "a", jd[0]), (pid, "lap", "default", "b", jd[1]),
            (pid, "lap", "a", "a", jo[0]), (pid, "lap", "b", "b", jo[1]),
            (pid, "lap", "a", "b", ab), (pid, "lap", "b", "a", ba)]


def test_matrix_from_synthetic_table():
    # 2 cards, 1 problem: J_default=0, J_oracle=(2, 4); theta of card a scores 1 under b -> T[a, b] = 0.25
    T = transfer_matrix(_table(_two_card_rows()), ["a", "b"])
    assert T[0, 0] == 1.0 and T[1, 1] == 1.0 and T[0, 1] == pytest.approx(0.25) and T[1, 0] == pytest.approx(0.5)


def test_diag_margin_and_verdict():
    T = torch.tensor([[1.0, 0.25], [0.5, 1.0]])
    assert torch.allclose(diag_margin(T), torch.tensor([0.5, 0.75]))
    assert go_no_go(diag_margin(T), 0.2) and not go_no_go(torch.tensor([0.1, 0.1]), 0.2)


def test_verdict_ignores_unscoreable_cards():
    nan = float("nan")
    v, n, med = transfer_verdict(torch.tensor([nan, nan, nan]), 0.2)
    assert v == "INCONCLUSIVE" and n == 0 and med != med
    assert not go_no_go(torch.tensor([nan, nan]), 0.2)
    v, n, med = transfer_verdict(torch.tensor([nan, 0.3, 0.5, 0.1]), 0.2)
    assert (v, n) == ("GO", 3) and med == pytest.approx(0.3) and go_no_go(torch.tensor([nan, 0.3, 0.5, 0.1]), 0.2)
    assert transfer_verdict(torch.tensor([nan, 0.1, 0.1, 0.0]), 0.2)[0] == "NO-GO"
    assert transfer_verdict(torch.tensor([nan, nan, nan, 0.9]), 0.2)[0] == "INCONCLUSIVE"  # 1 of 4 < half


def test_small_gap_problem_skipped():
    # p1: card b's oracle equals its default (gap 0, default won) -> skipped in column b only
    rows = _two_card_rows("p0") + _two_card_rows("p1", jo=(2.0, 3.0), jd=(0.0, 3.0), ab=100.0, ba=0.0)
    T, skipped = transfer_matrix(_table(rows), ["a", "b"], return_skipped=True)
    assert T[0, 1] == pytest.approx(0.25) and T[1, 1] == 1.0
    assert T[1, 0] == pytest.approx(0.25)  # mean of 0.5 (p0) and 0.0 (p1)
    assert skipped.tolist() == [[0, 1], [0, 1]]


def _tiny_cfg():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name="oracle", overrides=TINY)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    # 3 lap + 2 goto train cards, 1 problem per task; full train oracle path (writes transfer.parquet)
    cfg = _tiny_cfg()
    job = build_job(cfg)
    cards = load_cards(ROOT / "data/cards", "train")
    keep = [c for t, n in (("lap", 3), ("goto", 2)) for c in [c for c in cards if c.task == t][:n]]
    pairs = [(p, c) for p in range(job.problems.n) for c in range(len(keep)) if job.problems.task[p] == keep[c].task]
    job = OracleJob(problems=job.problems, cards=keep, pairs=pairs)
    out = tmp_path_factory.mktemp("o")
    run_oracle(job, cfg, out, CPU)
    return job, cfg, out / "train"


def _theta_star(d):
    o = pq.read_table(d / "oracle.parquet").to_pydict()
    th = torch.tensor([o[n] for n in THETA_NAMES]).T
    return o, {(p, c): th[i] for i, (p, c) in enumerate(zip(o["problem_id"], o["card_id"]))}


def test_transfer_rollouts_cover_offdiagonal(tiny):
    job, cfg, d = tiny
    o, ts = _theta_star(d)
    t = pq.read_table(d / "transfer.parquet").to_pydict()
    expect = {(pid, a.id, b.id) for pid, task in zip(job.problems.problem_id, job.problems.task)
              for a in job.cards for b in job.cards if a.task == b.task == task}
    got = [(p, a, b) for p, a, b in zip(t["problem_id"], t["theta_card"], t["rubric_card"]) if a != "default"]
    assert len(got) == len(set(got)) and set(got) == expect  # every ordered same-task pair, diagonal included
    fin = {(p, c): (jo, jd) for p, c, jo, jd in zip(o["problem_id"], o["card_id"], o["J_oracle"], o["J_default"])}
    for p, a, b, j in zip(t["problem_id"], t["theta_card"], t["rubric_card"], t["J"]):
        if a == b:
            assert j == fin[(p, b)][0]
        elif a == "default":
            assert j == fin[(p, b)][1]
    again = transfer_rollouts(job, ts, cfg, CPU, o).to_pydict()
    assert again == t


def test_diagonal_through_transfer_path_reproduces_oracle(tiny):
    job, cfg, d = tiny
    o, ts = _theta_star(d)
    J, valid, src = transfer_scores(job, ts, cfg, CPU, 0)  # offset 0: theta*_c scored by c on the finals layout
    assert valid.all() and src == [c for _, c in job.pairs]
    assert J.tolist() == o["J_oracle"]


def test_plot_transfer_writes_png(tmp_path):
    T = torch.tensor([[1.0, 0.25], [0.5, 1.0]])
    out = tmp_path / "t.png"
    plot_transfer({"lap": (T, ["kart_fast_001", "kart_kids_fair_001"]), "goto": (T, ["a", "b"])}, out)
    assert out.stat().st_size > 0


def test_plot_script_prints_verdict(tiny, tmp_path):
    import subprocess
    import sys
    _, _, d = tiny
    r = subprocess.run([sys.executable, "scripts/plot_transfer.py", str(d), "--out", str(tmp_path / "t.png")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "margin=" in r.stdout and (tmp_path / "t.png").exists()
    for scope in ("lap", "goto", "overall"):
        assert any(l.startswith(f"{scope} verdict=") and "n_valid=" in l and "median=" in l for l in r.stdout.splitlines())
