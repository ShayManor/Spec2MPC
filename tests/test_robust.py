from pathlib import Path

import pytest
import torch

from spec2mpc.features.robust import IQR_TO_STD, SCALE_FLOOR, robust_inv, robust_stats, robust_z

ROOT = Path(__file__).parent.parent


def test_stats_median_iqr_and_floor():
    g = torch.Generator().manual_seed(0)
    phi = torch.randn(20000, 3, generator=g) * torch.tensor([2.0, 1.0, 0.0]) + torch.tensor([1.0, -3.0, 7.0])
    phi[0, 0] = -1310.0  # heavy tail barely moves median and IQR
    center, scale = robust_stats(phi)
    assert center.shape == scale.shape == (3,)
    assert center[0].item() == pytest.approx(1.0, abs=0.05) and scale[0].item() == pytest.approx(2.0, rel=0.05)
    assert center[2].item() == 7.0 and scale[2].item() == pytest.approx(SCALE_FLOOR)
    q = phi[:, 1].quantile(torch.tensor([0.25, 0.75]))
    assert scale[1].item() == pytest.approx(((q[1] - q[0]) / IQR_TO_STD).item())


def test_z_and_inverse_roundtrip():
    phi = torch.randn(64, 5) * 100
    center, scale = robust_stats(phi)
    z = robust_z(phi, center, scale)
    assert torch.equal(z, torch.asinh((phi - center) / scale))
    assert torch.allclose(robust_inv(z, center, scale), phi, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("which", ["world_model", "critic"])
def test_existing_checkpoints_still_load(which):
    path = ROOT / "data" / "models" / which / "seed0" / "model.pt"
    if not path.exists():
        pytest.skip(f"{path} not present")
    from spec2mpc.model.io import load_checkpoint
    if which == "world_model":
        from spec2mpc.model.world_model import build_world_model as build
    else:
        from spec2mpc.critic.critic import build_critic as build
    m, _, _ = load_checkpoint(path, build)  # strict load: buffer names and state_dict keys unchanged
    phi = torch.randn(4, 20) * 3
    assert torch.equal(m.phi_to_z(phi), robust_z(phi, m.phi_center, m.phi_scale))
