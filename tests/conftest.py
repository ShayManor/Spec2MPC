import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.robots.kart import RuntimeConfig


@pytest.fixture
def kart_cfg():
    return OmegaConf.load("configs/robots/kart.yaml")


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(0)


@pytest.fixture
def runtime():
    return RuntimeConfig(control_rate_hz=20.0, solver_iters=10, perception_rate_hz=10.0)
