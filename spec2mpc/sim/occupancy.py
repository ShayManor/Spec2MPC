import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.sim.paths import path_ds_m
from spec2mpc.sim.scenes import Scene, pedestrians_at

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs (Mac): the eager path below is used
    triton = None


if triton is not None:

    @triton.jit
    def _corridor_min_kernel(ce, ca, ae, inv, aa, e2, cc, out, N, W, BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
        # min over W of the eager d2 expression in _corridor_dist, op by op (launched without FMA contraction)
        b = tl.program_id(0).to(tl.int64)
        rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mn = rn < N
        ccv = tl.load(cc + b * N + rn, mask=mn, other=0.0)  # [BLOCK_N]
        acc = tl.full([BLOCK_N], float("inf"), tl.float32)
        for w0 in range(0, W, BLOCK_W):
            rw = w0 + tl.arange(0, BLOCK_W)
            mw = rw < W
            m2 = mn[:, None] & mw[None, :]
            aev = tl.load(ae + b * W + rw, mask=mw, other=0.0)[None, :]
            invv = tl.load(inv + b * W + rw, mask=mw, other=0.0)[None, :]
            aav = tl.load(aa + b * W + rw, mask=mw, other=0.0)[None, :]
            e2v = tl.load(e2 + b * W + rw, mask=mw, other=0.0)[None, :]
            off = b * N * W + rn[:, None].to(tl.int64) * W + rw[None, :]
            cev = tl.load(ce + off, mask=m2, other=0.0)  # [BLOCK_N, BLOCK_W]
            cav = tl.load(ca + off, mask=m2, other=0.0)
            t = (cev - aev) * invv
            t = tl.where(t != t, t, tl.minimum(tl.maximum(t, 0.0), 1.0))  # clamp(0, 1), NaN passes through
            d2 = ccv[:, None] + aav - 2.0 * cav - 2.0 * t * (cev - aev) + t * t * e2v
            d2 = tl.where(m2, d2, float("inf"))
            nan = tl.max((d2 != d2).to(tl.int32), axis=1) > 0
            row = tl.where(nan, float("nan"), tl.min(d2, axis=1))
            acc = tl.where(acc != acc, acc, tl.where(row != row, row, tl.minimum(acc, row)))  # amin propagates NaN
        tl.store(out + b * N + rn, acc, mask=mn)

    @triton.jit
    def _min_plus_kernel(f, out, n, s_m, s_k, s_j, o_m, o_i, o_j, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
        # out[m, i, j] = min_k ((i - k)^2 + f[m, k, j]); f and out addressed by strides
        m = tl.program_id(0).to(tl.int64)
        ri = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
        rj = tl.arange(0, BLOCK_J)
        mj = rj < n
        fi = ri.to(tl.float32)[:, None]
        acc = tl.full([BLOCK_I, BLOCK_J], float("inf"), tl.float32)
        for k in range(0, n):
            g = tl.load(f + m * s_m + k * s_k + rj * s_j, mask=mj, other=float("inf"))[None, :]  # [1, BLOCK_J]
            d = fi - k
            acc = tl.minimum(acc, d * d + g)
        tl.store(out + m * o_m + ri[:, None] * o_i + rj[None, :] * o_j, acc, mask=(ri < n)[:, None] & mj[None, :])


def _use_triton(x: Tensor) -> bool:
    return triton is not None and x.is_cuda and x.dtype == torch.float32


@dataclass
class GridSpec:
    size_m: float = 20.0
    res_m: float = 0.1
    frame_offsets_s: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
    edt_chunk_bytes: float = 2.0e9

    @property
    def n(self) -> int:
        return round(self.size_m / self.res_m)

    @classmethod
    def from_cfg(cls, cfg_grid: DictConfig) -> "GridSpec":
        return cls(
            size_m=float(cfg_grid.size_m),
            res_m=float(cfg_grid.res_m),
            frame_offsets_s=tuple(float(x) for x in cfg_grid.frame_offsets_s),
            edt_chunk_bytes=float(cfg_grid.edt_chunk_bytes),
        )


def cell_centers(spec: GridSpec, origin_xy: Tensor) -> Tensor:
    # origin_xy [B, 2] -> [B, n, n, 2]; [b, i, j] = (x, y) with row i <-> y, column j <-> x
    n = spec.n
    c = -spec.size_m / 2 + (torch.arange(n, device=origin_xy.device, dtype=origin_xy.dtype) + 0.5) * spec.res_m  # [n]
    off = torch.stack([c[None, :].expand(n, n), c[:, None].expand(n, n)], dim=-1)  # [n, n, 2]
    return origin_xy[:, None, None] + off


def _corridor_dist(scene: Scene, cells: Tensor, s_center: Tensor, half_window_m: float) -> Tensor:
    # cells [Bc, N, 2] relative to nothing (world), s_center [Bc] -> distance to the centerline window [Bc, N]
    path = scene.path
    ds = path_ds_m(path)
    w = int(math.ceil(2 * half_window_m / ds)) + 1
    n_open = torch.round(path.length_m / ds).long() + 1  # [Bc]
    n_closed = torch.round(path.length_m / ds).long()
    i0 = torch.floor((s_center - half_window_m) / ds).long()[:, None] + torch.arange(w, device=cells.device)  # [Bc, W]

    def pick(i: Tensor) -> Tensor:
        idx = torch.where(path.closed[:, None], torch.remainder(i, n_closed[:, None]), i.clamp(min=0).minimum(n_open[:, None] - 1))
        return torch.gather(path.xy, 1, idx[..., None].expand(-1, -1, 2))  # [Bc, W, 2]

    a, b = pick(i0), pick(i0 + 1)
    ref = cells.mean(dim=1, keepdim=True)  # local coordinates keep float32 cancellation small
    c, a, b = cells - ref, a - ref, b - ref
    e = b - a  # [Bc, W, 2]
    e2 = (e * e).sum(-1)  # [Bc, W]
    inv = torch.where(e2 > 1e-12, 1.0 / e2.clamp(min=1e-12), torch.zeros_like(e2))
    ce = torch.bmm(c, e.transpose(1, 2))  # [Bc, N, W]
    ae = (a * e).sum(-1)[:, None]
    ca = torch.bmm(c, a.transpose(1, 2))
    cc = (c * c).sum(-1)[..., None]
    aa = (a * a).sum(-1)[:, None]
    if _use_triton(ce):
        bc, nc, wc = ce.shape
        d2min = torch.empty(bc, nc, dtype=ce.dtype, device=ce.device)
        grid = (bc, triton.cdiv(nc, 64))
        _corridor_min_kernel[grid](ce, ca.contiguous(), ae.contiguous(), inv.contiguous(), aa.contiguous(), e2.contiguous(),
                                   cc.contiguous(), d2min, nc, wc, BLOCK_N=64, BLOCK_W=32, enable_fp_fusion=False)
        return d2min.clamp(min=0.0).sqrt()
    t = ((ce - ae) * inv[:, None]).clamp(0.0, 1.0)
    d2 = cc + aa - 2 * ca - 2 * t * (ce - ae) + t * t * e2[:, None]
    return d2.amin(dim=-1).clamp(min=0.0).sqrt()


def _circles(cells: Tensor, c: Tensor, r: Tensor) -> Tensor:
    # cells [Bc, N, 2], c [Bc, O, 2], r [Bc, O] -> [Bc, N] signed distance to the union (inf if none)
    if r.shape[1] == 0:
        return torch.full(cells.shape[:2], float("inf"), dtype=cells.dtype, device=cells.device)
    d = torch.cdist(cells, c) - r[:, None]  # [Bc, N, O]
    return torch.where((r > 0)[:, None], d, torch.full_like(d, float("inf"))).amin(dim=-1)


def clearance_grid(scene: Scene, origin_xy: Tensor, t_s: Tensor, spec: GridSpec, path_window: Tuple[Tensor, float]) -> Tensor:
    # origin_xy [B, 2], t_s [B], path_window = (s_center [B], half_window_m) -> clearance [B, F, n, n] (negative inside)
    s_center, half = path_window
    b, n, nf = origin_xy.shape[0], spec.n, len(spec.frame_offsets_s)
    peds = [pedestrians_at(scene, t_s + off) for off in spec.frame_offsets_s]  # F x [B, P, 2]
    w = int(math.ceil(2 * half / path_ds_m(scene.path))) + 1
    per_env = n * n * (w + scene.obstacles_r.shape[1] + scene.ped_r.shape[1] + 4) * 4 * 6
    chunk = max(1, int(spec.edt_chunk_bytes // per_env))
    out = torch.empty(b, nf, n, n, dtype=origin_xy.dtype, device=origin_xy.device)
    for lo in range(0, b, chunk):
        sl = slice(lo, min(lo + chunk, b))
        sc = scene.index(sl)
        cells = cell_centers(spec, origin_xy[sl]).reshape(-1, n * n, 2)  # [Bc, N, 2]
        clr = sc.path.half_width_m[:, None] - _corridor_dist(sc, cells, s_center[sl], half)
        clr = torch.minimum(clr, _circles(cells, sc.obstacles_xy, sc.obstacles_r))
        for f in range(nf):
            out[sl, f] = torch.minimum(clr, _circles(cells, peds[f][sl], sc.ped_r)).reshape(-1, n, n)
    return out


def rasterize(scene: Scene, origin_xy: Tensor, t_s: Tensor, spec: GridSpec, path_window: Tuple[Tensor, float]) -> Tensor:
    # -> occ [B, F, n, n] bool; frame f is the scene at t + frame_offsets_s[f]
    return clearance_grid(scene, origin_xy, t_s, spec, path_window) <= 0


def _sq_edt_1d(f: Tensor, axis_len: int) -> Tensor:
    # f [M, n, n] squared costs, minimised along dim 1: out[m, i, j] = min_k ((i - k)^2 + f[m, k, j])
    if _use_triton(f):  # integer-valued costs, so the result is exact and equal to the eager path
        m, nj = f.shape[0], f.shape[2]
        out = torch.empty(m, nj, axis_len, dtype=f.dtype, device=f.device).transpose(1, 2)  # [M, n, n], i fastest
        grid = (m, triton.cdiv(axis_len, 8))
        _min_plus_kernel[grid](f, out, axis_len, *f.stride(), *out.stride(), BLOCK_I=8, BLOCK_J=triton.next_power_of_2(nj))
        return out
    k = torch.arange(axis_len, device=f.device, dtype=f.dtype)
    sq = (k[:, None] - k[None]) ** 2  # [n(i), n(k)]
    return (sq[None, :, :, None] + f[:, None]).amin(dim=2)  # [M, n(i), n(k), n(j)] -> [M, n, n]


def signed_distance(occ: Tensor, res_m: float, *, chunk_bytes: float) -> Tensor:
    # occ [B, F, n, n] bool -> sdf [B, F, n, n] meters, positive in free space (exact Euclidean)
    b, nf, n, _ = occ.shape
    flat = occ.reshape(b * nf, n, n)
    inf = float(4 * n * n)
    chunk = max(1, int(chunk_bytes // (n * n * n * 4 * 2)))
    out = torch.empty(b * nf, n, n, dtype=torch.float32, device=occ.device)
    for lo in range(0, b * nf, chunk):
        o = flat[lo : lo + chunk]
        # row 0 of the stack: distance to nearest occupied cell; row 1: distance to nearest free cell
        f0 = torch.stack([torch.where(o, 0.0, inf), torch.where(o, inf, 0.0)], dim=1).reshape(-1, n, n)
        g = _sq_edt_1d(f0, n)  # along y
        d2 = _sq_edt_1d(g.transpose(1, 2), n).transpose(1, 2)  # along x
        d = d2.reshape(-1, 2, n, n).sqrt() * res_m
        out[lo : lo + chunk] = d[:, 0] - d[:, 1]
    sdf = out.reshape(b, nf, n, n)
    any_occ = occ.flatten(2).any(-1)[..., None, None]
    all_occ = occ.flatten(2).all(-1)[..., None, None]
    far = n * res_m
    sdf = torch.where(~any_occ, torch.full_like(sdf, far), sdf)
    return torch.where(all_occ, torch.full_like(sdf, -far), sdf)


def sdf_lookup(sdf: Tensor, origin_xy: Tensor, spec: GridSpec, xy: Tensor, frame_idx: Tensor) -> Tensor:
    # sdf [B, F, n, n], origin_xy [B, 2], xy [B, K, 2], frame_idx [K] long -> [B, K]; bilinear, differentiable in xy
    half = spec.n * spec.res_m / 2
    g = (xy - origin_xy[:, None]) / half  # [B, K, 2] normalised (x, y)
    # non-finite queries read the centre and return NaN: grid_sample's backward is unsafe on NaN grids
    fin = torch.isfinite(g).all(dim=-1)  # [B, K]
    g = torch.where(fin[..., None], g, torch.zeros_like(g))
    out = torch.zeros(xy.shape[:2], dtype=sdf.dtype, device=sdf.device)
    for f in torch.unique(frame_idx).tolist():
        k = (frame_idx == f).nonzero(as_tuple=True)[0]
        v = F.grid_sample(sdf[:, f : f + 1], g[:, k][:, None].to(sdf.dtype), mode="bilinear", padding_mode="border", align_corners=False)  # [B, 1, 1, Kf]
        out[:, k] = v[:, 0, 0]
    return torch.where(fin, out, torch.full_like(out, float("nan")))
