import math

import torch
from torch import Tensor


class BatchedCMAES:
    """(mu/mu_w, lambda)-CMA-ES maximizing P independent problems in [-1, 1]^D (Hansen 2016 tutorial defaults)."""

    def __init__(self, mean0: Tensor, sigma0: float, lam: int, gen: torch.Generator):
        # mean0: [P, D]
        P, D = mean0.shape
        dt = torch.float64
        self.gen, self.lam, self.D = gen, lam, D
        self.mu = lam // 2
        w = math.log((lam + 1) / 2) - torch.log(torch.arange(1, self.mu + 1, dtype=dt))
        self.weights = w / w.sum()  # [mu]
        self.mu_eff = float(1.0 / (self.weights ** 2).sum())
        self.c_sigma = (self.mu_eff + 2) / (D + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0.0, math.sqrt((self.mu_eff - 1) / (D + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / D) / (D + 4 + 2 * self.mu_eff / D)
        self.c_1 = 2 / ((D + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((D + 2) ** 2 + self.mu_eff))
        self.chi_n = math.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))
        self.mean = mean0.to(dt).clone()  # [P, D]
        self.sigma = torch.full((P,), float(sigma0), dtype=dt)  # [P]
        self.C = torch.eye(D, dtype=dt).repeat(P, 1, 1)  # [P, D, D]
        self.B = self.C.clone()  # [P, D, D] eigenvectors of C
        self.d = torch.ones(P, D, dtype=dt)  # [P, D] sqrt eigenvalues of C
        self.p_sigma = torch.zeros(P, D, dtype=dt)
        self.p_c = torch.zeros(P, D, dtype=dt)
        self.best_x = self.mean.clone()  # [P, D]
        self.best_f = torch.full((P,), -math.inf, dtype=dt)  # [P]
        self.generation = 0
        self.n_evals = 0

    def _sample_z(self) -> Tensor:
        # -> [P, lam, D] standard normal
        return torch.randn(self.mean.shape[0], self.lam, self.D, generator=self.gen, dtype=torch.float64)

    def ask(self) -> Tensor:
        # -> x [P, lam, D], clipped to [-1, 1]
        y = torch.einsum("pij,pj,pkj->pki", self.B, self.d, self._sample_z())  # B diag(d) z
        return (self.mean[:, None] + self.sigma[:, None, None] * y).clamp(-1.0, 1.0)

    def tell(self, x: Tensor, f: Tensor) -> None:
        # x: [P, lam, D] (as returned by ask), f: [P, lam]; non-finite f ranks last
        x = x.to(torch.float64)
        f = torch.where(torch.isfinite(f), f.to(torch.float64), -math.inf)
        i_best = f.argmax(1)  # [P]
        f_best = f.gather(1, i_best[:, None])[:, 0]
        better = f_best > self.best_f
        self.best_f = torch.where(better, f_best, self.best_f)
        self.best_x = torch.where(better[:, None], x[torch.arange(x.shape[0]), i_best], self.best_x)

        order = torch.argsort(f, dim=1, descending=True, stable=True)[:, :self.mu]  # [P, mu]
        y = (x - self.mean[:, None]) / self.sigma[:, None, None]  # [P, lam, D] from the clipped points
        y_sel = y.gather(1, order[..., None].expand(-1, -1, self.D))  # [P, mu, D]
        y_w = (self.weights[None, :, None] * y_sel).sum(1)  # [P, D]
        self.mean = self.mean + self.sigma[:, None] * y_w

        c_inv_sqrt_y = torch.einsum("pij,pj,pkj,pk->pi", self.B, 1 / self.d, self.B, y_w)  # C^-1/2 y_w
        cs = self.c_sigma
        self.p_sigma = (1 - cs) * self.p_sigma + math.sqrt(cs * (2 - cs) * self.mu_eff) * c_inv_sqrt_y
        ps_norm = self.p_sigma.norm(dim=1)  # [P]
        self.generation += 1
        self.n_evals += self.lam
        h_sigma = (ps_norm / math.sqrt(1 - (1 - cs) ** (2 * self.generation))
                   < (1.4 + 2 / (self.D + 1)) * self.chi_n).to(torch.float64)  # [P]
        cc = self.c_c
        self.p_c = (1 - cc) * self.p_c + h_sigma[:, None] * math.sqrt(cc * (2 - cc) * self.mu_eff) * y_w
        rank1 = self.p_c[:, :, None] * self.p_c[:, None, :] + ((1 - h_sigma) * cc * (2 - cc))[:, None, None] * self.C
        rank_mu = torch.einsum("m,pmi,pmj->pij", self.weights, y_sel, y_sel)
        self.C = (1 - self.c_1 - self.c_mu) * self.C + self.c_1 * rank1 + self.c_mu * rank_mu
        self.C = (self.C + self.C.transpose(1, 2)) / 2
        self.sigma = self.sigma * torch.exp((cs / self.d_sigma) * (ps_norm / self.chi_n - 1))

        evals, self.B = torch.linalg.eigh(self.C)
        self.d = evals.clamp_min(1e-20).sqrt()

    def state_dict(self) -> dict:
        keys = ("mean", "sigma", "C", "B", "d", "p_sigma", "p_c", "best_x", "best_f")
        d = {k: getattr(self, k).clone() for k in keys}
        d.update(generation=self.generation, n_evals=self.n_evals, lam=self.lam, gen_state=self.gen.get_state())
        return d

    def load_state_dict(self, d: dict) -> None:
        assert d["lam"] == self.lam and d["mean"].shape[1] == self.D
        for k in ("mean", "sigma", "C", "B", "d", "p_sigma", "p_c", "best_x", "best_f"):
            setattr(self, k, d[k].clone())
        self.generation, self.n_evals = d["generation"], d["n_evals"]
        self.gen.set_state(d["gen_state"])
