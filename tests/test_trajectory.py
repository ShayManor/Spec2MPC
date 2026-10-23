import torch

from spec2mpc.features.metrics import METRIC_NAMES
from spec2mpc.sim.trajectory import DONE_COMPLETED, Trajectory, downsample


def _traj(B=3, T=20, dt=0.05):
    return Trajectory(
        t_s=torch.arange(T).float().repeat(B, 1) * dt, x=torch.zeros(B, T, 8), u=torch.zeros(B, T, 2),
        s=torch.zeros(B, T), a_long=torch.zeros(B, T), a_lat=torch.zeros(B, T),
        clearance_true=torch.ones(B, T), solver_ok=torch.ones(B, T, dtype=torch.bool),
        valid=torch.ones(B, T, dtype=torch.bool), p_drive_W=torch.zeros(B, T),
        done_reason=torch.full((B,), DONE_COMPLETED), completion_time_s=torch.ones(B),
        laps_done=torch.zeros(B, dtype=torch.long), dt_s=dt,
    )


def test_downsample():
    d = downsample(_traj(), 10.0)
    assert d.t_s.shape == (3, 10) and d.x.shape == (3, 10, 8) and abs(d.dt_s - 0.1) < 1e-9
    assert d.done_reason.shape == (3,)
    assert downsample(_traj(), 1000.0).t_s.shape == (3, 20)


def test_index():
    assert _traj().index(torch.tensor([0, 2])).x.shape == (2, 20, 8)
    assert _traj().index(slice(1, 3)).done_reason.shape == (2,)


def test_metric_names():
    assert len(METRIC_NAMES) == 21 and len(set(METRIC_NAMES)) == 21
