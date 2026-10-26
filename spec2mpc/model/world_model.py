"""World model W (S3): robot, path, theta, xi -> episode features phi. No text, no rubric."""
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional as F

from spec2mpc.features.robust import robust_inv, robust_stats, robust_z
from spec2mpc.model.data import RolloutTensors

N_ROBOT, D_ROBOT = 6, 24
N_PATH, D_PATH = 64, 6
D_THETA, D_XI, D_PHI = 14, 6, 20
N_TYPES = 5  # cls, robot, path, theta, xi


class WorldModel(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.robot_in = nn.Linear(D_ROBOT, d_model)
        self.path_in = nn.Linear(D_PATH, d_model)
        self.theta_in = nn.Linear(D_THETA, d_model)
        self.xi_in = nn.Linear(D_XI, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.path_pos = nn.Parameter(0.02 * torch.randn(1, N_PATH, d_model))
        self.type_emb = nn.Parameter(0.02 * torch.randn(N_TYPES, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.head = nn.Linear(d_model, D_PHI)
        self.register_buffer("phi_center", torch.zeros(D_PHI))
        self.register_buffer("phi_scale", torch.ones(D_PHI))
        types = [0] + [1] * N_ROBOT + [2] * N_PATH + [3, 4]
        self.register_buffer("types", torch.tensor(types), persistent=False)

    def set_phi_stats(self, phi: Tensor) -> None:
        # phi [E, 20] raw, train blocks only -> per-feature median and IQR / 1.349 (floored)
        center, scale = robust_stats(phi)
        self.phi_center.copy_(center)
        self.phi_scale.copy_(scale)

    def phi_to_z(self, phi: Tensor) -> Tensor:
        # phi [..., 20] raw -> z [..., 20] = asinh((phi - center) / scale), robust to heavy tails
        return robust_z(phi, self.phi_center, self.phi_scale)

    def z_to_phi(self, z: Tensor) -> Tensor:
        # z [..., 20] -> phi [..., 20] raw, exact inverse of phi_to_z
        return robust_inv(z, self.phi_center, self.phi_scale)

    def forward_z(self, robot_tok: Tensor, path_tok: Tensor, theta_n: Tensor, xi_n: Tensor) -> Tensor:
        # robot_tok [B,6,24], path_tok [B,64,6], theta_n [B,14], xi_n [B,6] -> z [B,20]
        B = theta_n.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), self.robot_in(robot_tok), self.path_in(path_tok) + self.path_pos,
                       self.theta_in(theta_n)[:, None], self.xi_in(xi_n)[:, None]], dim=1)  # [B, 73, d]
        x = self.encoder(x + self.type_emb[self.types])  # [B, 73, d]
        return self.head(x[:, 0])  # [B, 20]

    def forward(self, robot_tok: Tensor, path_tok: Tensor, theta_n: Tensor, xi_n: Tensor) -> Tensor:
        # robot_tok [B,6,24], path_tok [B,64,6], theta_n [B,14], xi_n [B,6] -> phi_hat [B,20] raw units
        return self.z_to_phi(self.forward_z(robot_tok, path_tok, theta_n, xi_n))

    def expected(self, robot_tok: Tensor, path_tok: Tensor, theta_n: Tensor, xi_n: Tensor) -> Tensor:
        # robot_tok [B,6,24], path_tok [B,64,6], theta_n [B,14], xi_n [B,X,6] -> mean over X of phi_hat [B,20]
        B, X = xi_n.shape[:2]
        rep = lambda t: t.repeat_interleave(X, dim=0)  # noqa: E731
        y = self(rep(robot_tok), rep(path_tok), rep(theta_n), xi_n.reshape(B * X, -1))  # [B*X, 20]
        return y.view(B, X, -1).mean(1)


def build_world_model(cfg: DictConfig) -> WorldModel:
    m = cfg.model
    return WorldModel(int(m.d_model), int(m.n_layers), int(m.n_heads), float(m.dropout))


def wm_loss(model: WorldModel, batch: RolloutTensors, huber_delta: float = 1.0) -> Tensor:
    # Huber on robust z = asinh((phi - center) / scale); batch fields [B, ...] -> scalar
    z = model.forward_z(batch.robot_tok, batch.path_tok, batch.theta_n, batch.xi_n)  # [B, 20]
    target = model.phi_to_z(batch.phi)  # [B, 20]
    return F.huber_loss(z, target, delta=huber_delta)
