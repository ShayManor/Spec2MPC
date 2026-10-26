"""Simple text-only baselines: default, mean_theta, nn_retrieval (all in normalized theta space)."""
from omegaconf import DictConfig
from torch import Tensor
import torch

from spec2mpc.mpc.theta import default_theta, normalize_theta


def default_theta_n(B: int, mpc_cfg: DictConfig) -> Tensor:
    # -> [B, 14] normalized default theta
    return normalize_theta(default_theta(B, mpc_cfg).values, mpc_cfg)


class MeanTheta:
    def fit(self, theta_n: Tensor, task: Tensor, n_tasks: int = 2) -> "MeanTheta":
        # theta_n [R, 14], task [R] long: mean theta* per task over train oracle records
        missing = [t for t in range(n_tasks) if not (task == t).any()]
        if missing:
            raise ValueError(f"MeanTheta: no train records for task ids {missing}")
        self.mean = torch.stack([theta_n[task == t].mean(0) for t in range(n_tasks)])  # [T, 14]
        return self

    def predict(self, task: Tensor) -> Tensor:
        # task [B] long -> [B, 14]
        bad = sorted({int(t) for t in task if not 0 <= int(t) < len(self.mean)})
        if bad:
            raise ValueError(f"MeanTheta: unknown task ids {bad}")
        return self.mean[task]


class NNRetrieval:
    def fit(self, text_emb: Tensor, task: Tensor, theta_n: Tensor) -> "NNRetrieval":
        # text_emb [S, d] (L2-normalized), task [S] long, theta_n [S, 14]: one row per train statement
        self.emb, self.task, self.theta = text_emb, task, theta_n
        return self

    def predict(self, text_emb: Tensor, task: Tensor) -> Tensor:
        # text_emb [B, d], task [B] long -> [B, 14]: theta of the cosine-nearest train statement of the same task
        sim = torch.nn.functional.normalize(text_emb, dim=-1) @ torch.nn.functional.normalize(self.emb, dim=-1).T  # [B, S]
        sim = sim.masked_fill(task[:, None] != self.task[None, :], float("-inf"))
        return self.theta[sim.argmax(1)]
