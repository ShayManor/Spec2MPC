import math
from dataclasses import dataclass, replace
from typing import Tuple

import torch
from torch import Tensor

from spec2mpc.batch import Batched
from spec2mpc.robots.kart import PerceptionProfile
from spec2mpc.sim.occupancy import GridSpec, cell_centers, rasterize, signed_distance
from spec2mpc.sim.scenes import Scene

PSI_IDX = 2  # heading index in the kart state


@dataclass
class PerceptionNoise(Batched):
    bias_xy: Tensor  # [B, 2] per-episode localization bias
    seed: Tensor  # [B] int64


def sample_perception_noise(profile: PerceptionProfile, gen: torch.Generator) -> PerceptionNoise:
    # profile [B] -> bias_xy [B, 2] (uniform direction, magnitude loc_bias_m), seed [B]
    b = profile.loc_bias_m.shape[0]
    ang = torch.rand(b, generator=gen, device=gen.device) * 2 * math.pi  # [B]
    bias = torch.stack([ang.cos(), ang.sin()], dim=-1) * profile.loc_bias_m.to(gen.device)[:, None]  # [B, 2]
    seed = torch.randint(0, 2**62, (b,), generator=gen, device=gen.device, dtype=torch.int64)
    return PerceptionNoise(bias_xy=bias.to(profile.loc_bias_m.device), seed=seed.to(profile.loc_bias_m.device))


def window_half_m(spec: GridSpec) -> float:
    # twice the grid half-diagonal: deliberate margin so the path window covers curvy paths
    return spec.size_m * math.sqrt(2)


@dataclass
class PerceptionState:
    gen: torch.Generator
    occ_ring: Tensor  # [B, L+1, F, n, n] bool, perceived occupancy per update
    origin_ring: Tensor  # [B, L+1, 2]
    count: int  # perception updates so far

    @classmethod
    def init(cls, B: int, spec: GridSpec, max_latency_updates: int, seed: int, device) -> "PerceptionState":
        n, nf, ring = spec.n, len(spec.frame_offsets_s), max_latency_updates + 1
        return cls(
            gen=torch.Generator(device=device).manual_seed(seed),
            occ_ring=torch.zeros(B, ring, nf, n, n, dtype=torch.bool, device=device),
            origin_ring=torch.zeros(B, ring, 2, device=device),
            count=0,
        )


def perceive(
    scene: Scene,
    x_true: Tensor,
    t_s: float,
    profile: PerceptionProfile,
    noise: PerceptionNoise,
    state: PerceptionState,
    spec: GridSpec,
    s_center: Tensor,
    perception_rate_hz: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    # x_true [B, NX], s_center [B] -> x_hat [B, NX], sdf [B, F, n, n], origin [B, 2]
    b, dev, gen = x_true.shape[0], x_true.device, state.gen
    nf, n = len(spec.frame_offsets_s), spec.n
    half = window_half_m(spec)
    ring = state.occ_ring.shape[1]
    delay = torch.round(profile.latency_s * perception_rate_hz).long()  # [B]
    if (delay > ring - 1).any():
        raise ValueError(f"latency of {int(delay.max())} updates exceeds max_latency_updates={ring - 1}")
    t = torch.full((b,), float(t_s), device=dev)

    x_hat = x_true.clone()
    x_hat[:, :2] = x_true[:, :2] + noise.bias_xy + torch.randn(b, 2, generator=gen, device=dev) * profile.loc_noise_std_m[:, None]  # [B, 2]
    origin = x_hat[:, :2]

    occ = rasterize(scene, x_true[:, :2], t, spec, (s_center, half))  # [B, F, n, n] around the true pose
    pred = profile.has_prediction
    if nf > 1 and pred.any():
        idx = pred.nonzero(as_tuple=True)[0]
        sub = scene.index(idx)
        err = torch.randn(sub.ped_xy0.shape, generator=gen, device=dev) * profile.pred_error_m[idx][:, None, None]  # [Bp, P, 2]
        sub = replace(sub, ped_xy0=sub.ped_xy0 + err)
        spec_future = replace(spec, frame_offsets_s=spec.frame_offsets_s[1:])
        occ[idx, 1:] = rasterize(sub, x_true[idx, :2], t[idx], spec_future, (s_center[idx], half))
    occ[~pred, 1:] = occ[~pred, :1]

    # visibility and flips live in the grid frame (centered at x_hat)
    rel = cell_centers(spec, torch.zeros(b, 2, device=dev, dtype=x_true.dtype))  # [B, n, n, 2]
    dist = rel.norm(dim=-1)  # [B, n, n]
    dpsi = torch.remainder(torch.atan2(rel[..., 1], rel[..., 0]) - x_hat[:, PSI_IDX, None, None] + math.pi, 2 * math.pi) - math.pi
    visible = (dist <= profile.range_m[:, None, None]) & (dpsi.abs() <= profile.fov_rad[:, None, None] / 2 + 1e-6)  # [B, n, n]

    po, pf = profile.p_false_occ[:, None, None], profile.p_false_free[:, None, None]
    u = torch.rand(b, n, n, generator=gen, device=dev)
    for f in range(nf):
        if f > 0:
            fresh = torch.rand(b, n, n, generator=gen, device=dev)  # [B, n, n] one frame at a time
            u = torch.where(pred[:, None, None], fresh, u)  # no prediction: frames 1.. repeat the flipped frame 0
        occ[:, f] = torch.where(occ[:, f], u >= pf, u < po) & visible  # flips, then visibility: no hallucination outside the view

    slot = state.count % ring
    state.occ_ring[:, slot] = occ
    state.origin_ring[:, slot] = origin
    state.count += 1
    delay = torch.minimum(delay, torch.full_like(delay, state.count - 1))
    pick = (state.count - 1 - delay) % ring  # [B]
    ar = torch.arange(b, device=dev)
    sdf = signed_distance(state.occ_ring[ar, pick], spec.res_m, chunk_bytes=spec.edt_chunk_bytes)  # [B, F, n, n]
    return x_hat, sdf, state.origin_ring[ar, pick]
