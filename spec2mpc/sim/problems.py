import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from spec2mpc.robots.kart import Robot, RuntimeConfig, sample_karts
from spec2mpc.sim.scenes import Scene, generate_scenes
from spec2mpc.tokenize.path_tokens import path_tokens
from spec2mpc.tokenize.robot_tokens import robot_columns

TASKS = ("lap", "goto")


@dataclass
class Problems:
    robot: Robot  # [P], one RuntimeConfig
    scene: Scene  # [P]
    path_tok: Tensor  # [P, n_path_tokens, 6]
    problem_id: List[str]
    task: List[str]

    @property
    def n(self) -> int:
        return len(self.problem_id)

    def index(self, idx: Union[Tensor, slice]) -> "Problems":
        # idx: slice | long tensor [M] -> Problems with P = M
        ids = list(range(self.n))[idx] if isinstance(idx, slice) else [int(i) for i in idx]
        return Problems(robot=self.robot.index(idx), scene=self.scene.index(idx), path_tok=self.path_tok[idx],
                        problem_id=[self.problem_id[i] for i in ids], task=[self.task[i] for i in ids])

    def expand(self, rep: Tensor) -> Tuple[Robot, Scene]:
        # rep: long [E] problem index per episode -> robot, scene [E]
        p = self.index(rep)
        return p.robot, p.scene


def stable_seed(*parts: object) -> int:
    # 63-bit seed from repr(parts); the same in every process (never Python hash)
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(), "little") >> 1


def sample_problems(n_per_task: Mapping[str, int], cfg: DictConfig, seed: int, runtime: RuntimeConfig,
                    prefix: str) -> Problems:
    # cfg: root config with robots, sim, tokenize -> Problems [sum n_per_task], lap problems first, then goto
    unknown = set(n_per_task) - set(TASKS)
    assert not unknown, f"unknown task in n_per_task: {unknown}"
    counts = [(t, int(n_per_task.get(t, 0))) for t in TASKS if int(n_per_task.get(t, 0)) > 0]
    gen = torch.Generator().manual_seed(seed)
    robot = sample_karts(sum(n for _, n in counts), cfg.robots, runtime, gen)
    scenes = [generate_scenes(n, cfg.sim, gen, t) for t, n in counts]
    scene = Scene.cat(scenes) if len(scenes) > 1 else scenes[0]
    tok = path_tokens(scene.path, scene, robot.params, cfg.tokenize)
    return Problems(robot=robot, scene=scene, path_tok=tok,
                    problem_id=[f"{prefix}_{t}_{i:03d}" for t, n in counts for i in range(n)],
                    task=[t for t, n in counts for _ in range(n)])


def _stable_bytes(x: Tensor) -> bytes:
    # any numeric tensor -> little-endian float64 bytes after rounding to 1e-5, so CPU/CUDA/arm float noise hashes equal
    return (torch.round(x.detach().cpu().double() * 1e5) / 1e5).contiguous().numpy().astype("<f8").tobytes()


def problem_digest(problems: Problems) -> List[str]:
    # -> [P] blake2b hex of each problem's path tokens and raw robot fields (columns in sorted name order),
    # values rounded to 1e-5 before hashing
    cols = robot_columns(problems.robot)
    names = sorted(cols)
    out = []
    for i in range(problems.n):
        h = hashlib.blake2b(digest_size=16)
        h.update(_stable_bytes(problems.path_tok[i]))
        for n in names:
            h.update(n.encode())
            h.update(_stable_bytes(cols[n][i:i + 1]))
        out.append(h.hexdigest())
    return out


def load_oracle_problems(oracle_dir: Path) -> Tuple[Problems, DictConfig]:
    # oracle_dir: {out_dir}/{split} of an oracle run -> its problems rebuilt from the run's own config.yaml, and that
    # config; if problems.pt ({"problem_id", "digest"}) exists, every rebuilt problem must match its stored digest
    oracle_dir = Path(oracle_dir)
    cfg = OmegaConf.load(oracle_dir / "config.yaml")
    split = str(cfg.split)
    rt = RuntimeConfig(control_rate_hz=float(cfg.runtime.control_rate_hz), solver_iters=int(cfg.runtime.solver_iters),
                       perception_rate_hz=float(cfg.runtime.perception_rate_hz))
    problems = sample_problems(cfg.problems.n_per_task, cfg, int(cfg.problems[f"seed_{split}"]), rt, split)
    stored_path = oracle_dir / "problems.pt"
    if stored_path.exists():
        stored = torch.load(stored_path, weights_only=True)
        want = dict(zip(stored["problem_id"], stored["digest"]))
        bad = [p for p, d in zip(problems.problem_id, problem_digest(problems)) if want.get(p) != d]
        if bad:
            raise ValueError(f"rebuilt problems differ from {stored_path}: {bad}")
    return problems, cfg
