import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.sim.problems import sample_problems, stable_seed
from spec2mpc.sim.scenes import TASK_GOTO, TASK_LAP


@pytest.fixture
def root_cfg():
    return OmegaConf.create({"robots": OmegaConf.load("configs/robots/kart.yaml"),
                             "sim": OmegaConf.load("configs/sim/default.yaml"),
                             "tokenize": OmegaConf.load("configs/tokenize/path.yaml")})


def test_problems_deterministic_and_shaped(root_cfg, runtime):
    a = sample_problems({"lap": 2, "goto": 3}, root_cfg, 11, runtime, "t")
    b = sample_problems({"lap": 2, "goto": 3}, root_cfg, 11, runtime, "t")
    assert a.n == 5 and a.task == ["lap"] * 2 + ["goto"] * 3
    assert torch.equal(a.path_tok, b.path_tok) and a.path_tok.shape == (5, 64, 6)
    assert torch.equal(a.robot.params.mass_kg, b.robot.params.mass_kg)
    assert a.problem_id == ["t_lap_000", "t_lap_001", "t_goto_000", "t_goto_001", "t_goto_002"]
    assert a.scene.task.tolist() == [TASK_LAP] * 2 + [TASK_GOTO] * 3
    assert a.robot.runtime == runtime


def test_problems_index_and_expand(root_cfg, runtime):
    a = sample_problems({"lap": 2, "goto": 3}, root_cfg, 11, runtime, "t")
    sub = a.index(torch.tensor([4, 0]))
    assert sub.n == 2 and sub.problem_id == ["t_goto_002", "t_lap_000"] and sub.task == ["goto", "lap"]
    assert torch.equal(sub.path_tok[0], a.path_tok[4])
    robot, scene = a.expand(torch.tensor([1, 1, 3]))
    assert robot.params.batch_size == 3 and scene.batch_size == 3
    assert torch.equal(scene.seed, a.scene.seed[[1, 1, 3]])
    assert robot.robot_id == [a.robot.robot_id[i] for i in (1, 1, 3)]


def test_stable_seed():
    s = stable_seed("prob_lap_000", "kart_fast_001", 3)
    assert s == stable_seed("prob_lap_000", "kart_fast_001", 3) and 0 <= s < 2**63
    assert s != stable_seed("prob_lap_000", "kart_fast_001", 4)
    assert s == 7699035190336522817  # pinned: same value in every process (no Python hash)


def test_problem_digest_stable_and_sensitive(root_cfg, runtime):
    from spec2mpc.sim.problems import problem_digest
    a = sample_problems({"lap": 2, "goto": 1}, root_cfg, 11, runtime, "t")
    b = sample_problems({"lap": 2, "goto": 1}, root_cfg, 11, runtime, "t")
    da = problem_digest(a)
    assert da == problem_digest(b) and len(da) == 3 and len(set(da)) == 3
    a.path_tok[1, 0, 0] += 1e-3
    assert problem_digest(a) == [da[0], problem_digest(a)[1], da[2]] and problem_digest(a)[1] != da[1]
    a.robot.params.mass_kg[2] += 1.0
    assert problem_digest(a)[2] != da[2]


def oracle_dir(tmp_path, root_cfg, n_lap=1, n_goto=1, seed=5):
    cfg = OmegaConf.merge(root_cfg, {"split": "train",
                                     "problems": {"seed_train": seed, "seed_val": 9, "n_per_task": {"lap": n_lap, "goto": n_goto}},
                                     "runtime": {"control_rate_hz": 20.0, "solver_iters": 10, "perception_rate_hz": 10.0}})
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    return cfg


def test_load_oracle_problems_honors_run_config(tmp_path, root_cfg):
    from spec2mpc.robots.kart import RuntimeConfig
    from spec2mpc.sim.problems import load_oracle_problems
    cfg = oracle_dir(tmp_path, root_cfg)  # 1 + 1 problems, seed 5; configs/oracle.yaml has 4 + 4, seed 1001
    p, ocfg = load_oracle_problems(tmp_path)
    ref = sample_problems({"lap": 1, "goto": 1}, cfg, 5, RuntimeConfig(20.0, 10, 10.0), "train")
    assert p.problem_id == ["train_lap_000", "train_goto_000"] and torch.equal(p.path_tok, ref.path_tok)
    assert int(ocfg.problems.seed_train) == 5


def test_load_oracle_problems_checks_digest(tmp_path, root_cfg):
    from spec2mpc.sim.problems import load_oracle_problems, problem_digest
    oracle_dir(tmp_path, root_cfg)
    p, _ = load_oracle_problems(tmp_path)
    torch.save({"problem_id": p.problem_id, "digest": problem_digest(p)}, tmp_path / "problems.pt")
    load_oracle_problems(tmp_path)  # matching digests pass
    torch.save({"problem_id": p.problem_id, "digest": ["0" * 32, problem_digest(p)[1]]}, tmp_path / "problems.pt")
    with pytest.raises(ValueError, match="train_lap_000"):
        load_oracle_problems(tmp_path)


def test_problem_digest_ignores_float_noise(root_cfg, runtime):
    from spec2mpc.sim.problems import problem_digest
    a = sample_problems({"lap": 1, "goto": 1}, root_cfg, 11, runtime, "t")
    d0 = problem_digest(a)
    a.path_tok[0, 3, 2] = 0.123454  # away from a rounding boundary, so the noise below cannot cross one
    a.robot.params.mass_kg[0] = 200.000004
    d1 = problem_digest(a)
    a.path_tok = a.path_tok.double()  # float64 so the 1e-9 noise is representable
    a.path_tok[0, 3, 2] += 1e-9
    a.robot.params.mass_kg = a.robot.params.mass_kg.double()
    a.robot.params.mass_kg[0] += 1e-9
    assert problem_digest(a) == d1
    a.path_tok[0, 3, 2] += 1e-3
    assert problem_digest(a)[0] != d1[0] and problem_digest(a)[1] == d0[1]
