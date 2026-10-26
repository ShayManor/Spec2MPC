import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.baselines.oracle_reader import load_oracle_thetas
from spec2mpc.baselines.simple import MeanTheta, NNRetrieval, default_theta_n
from spec2mpc.mpc.theta import THETA_NAMES, default_theta, denormalize_theta


@pytest.fixture
def mpc_cfg():
    return OmegaConf.load("configs/mpc/kart.yaml")


def test_default_maps_to_default_theta(mpc_cfg):
    assert torch.allclose(denormalize_theta(default_theta_n(3, mpc_cfg), mpc_cfg), default_theta(3, mpc_cfg).values, atol=1e-6)


def test_mean_theta_per_task():
    th = torch.tensor([[0.0] * 14, [1.0] * 14, [4.0] * 14])
    m = MeanTheta().fit(th, torch.tensor([0, 0, 1]))
    out = m.predict(torch.tensor([1, 0, 0]))
    assert torch.allclose(out[:, 0], torch.tensor([4.0, 0.5, 0.5]))


def test_nn_retrieval_same_task_only():
    emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    th = torch.tensor([[0.0] * 14, [1.0] * 14])
    r = NNRetrieval().fit(emb, torch.tensor([0, 1]), th)
    out = r.predict(torch.tensor([[1.0, 0.0]]), torch.tensor([1]))
    assert torch.allclose(out, th[1:2])


def test_nn_retrieval_exact_statement_returns_its_theta():
    emb = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    th = torch.randn(6, 14)
    task = torch.tensor([0, 0, 0, 1, 1, 1])
    out = NNRetrieval().fit(emb, task, th).predict(emb, task)
    assert torch.allclose(out, th)


def test_load_oracle_thetas_k(tmp_path):
    def table(vals, extra):
        d = {"problem_id": ["p0", "p1"], "card_id": ["c0", "c0"], **extra}
        d.update({n: [v, v + 1.0] for n, v in zip(THETA_NAMES, [vals] * len(THETA_NAMES))})
        return pa.table(d)
    pq.write_table(table(1.0, {"task": ["lap", "lap"]}), tmp_path / "oracle.parquet")
    kt = pa.concat_tables([table(2.0, {"k": [48, 48]}), table(3.0, {"k": [96, 96]})])
    pq.write_table(kt, tmp_path / "at_k.parquet")
    star = load_oracle_thetas(tmp_path, None)
    assert torch.allclose(star[("p1", "c0")], torch.full((14,), 2.0))
    at48 = load_oracle_thetas(tmp_path, 48)
    assert set(at48) == {("p0", "c0"), ("p1", "c0")}
    assert torch.allclose(at48[("p0", "c0")], torch.full((14,), 2.0))


def test_mean_theta_missing_task_raises():
    with pytest.raises(ValueError, match="1"):
        MeanTheta().fit(torch.zeros(2, 14), torch.tensor([0, 0]))


def test_mean_theta_unknown_task_raises():
    m = MeanTheta().fit(torch.zeros(2, 14), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="7"):
        m.predict(torch.tensor([7]))


def test_load_oracle_thetas_missing_k_raises(tmp_path):
    d = {"problem_id": ["p0"], "card_id": ["c0"], "k": [48], **{n: [1.0] for n in THETA_NAMES}}
    pq.write_table(pa.table(d), tmp_path / "at_k.parquet")
    with pytest.raises(ValueError, match="96"):
        load_oracle_thetas(tmp_path, 96)
