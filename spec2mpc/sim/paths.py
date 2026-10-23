from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import Tensor

from spec2mpc.batch import Batched


@dataclass
class RefPath(Batched):
    xy: Tensor  # [B, M, 2]
    s: Tensor  # [B, M] arc length, s[:, 0] = 0
    heading: Tensor  # [B, M]
    curvature: Tensor  # [B, M]
    length_m: Tensor  # [B] (integer multiple of ds_m)
    closed: Tensor  # [B] bool
    half_width_m: Tensor  # [B]

    @classmethod
    def cat(cls, items: Sequence["RefPath"]) -> "RefPath":
        # items: list of [B_i, M_i, ...] -> [sum B_i, max M_i, ...], padding repeats the last point
        m = max(i.xy.shape[1] for i in items)

        def pad(x: Tensor) -> Tensor:
            if x.shape[1] == m:
                return x
            return torch.cat([x, x[:, -1:].expand(-1, m - x.shape[1], *x.shape[2:])], dim=1)

        return cls(
            xy=torch.cat([pad(i.xy) for i in items]),
            s=torch.cat([pad(i.s) for i in items]),
            heading=torch.cat([pad(i.heading) for i in items]),
            curvature=torch.cat([pad(i.curvature) for i in items]),
            length_m=torch.cat([i.length_m for i in items]),
            closed=torch.cat([i.closed for i in items]),
            half_width_m=torch.cat([i.half_width_m for i in items]),
        )


def path_ds_m(path: RefPath) -> float:
    return float(path.s[0, 1] - path.s[0, 0])


def wrap_angle(a: Tensor) -> Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def path_lookup(path: RefPath, s_query: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    # s_query [B, K] -> xy [B, K, 2], heading [B, K], curvature [B, K]; differentiable in s_query
    ds = path_ds_m(path)
    length = path.length_m[:, None]  # [B, 1]
    closed = path.closed[:, None]
    s_wrapped = torch.remainder(s_query, length)
    s_clamped = torch.minimum(s_query.clamp(min=0.0), length)
    sq = torch.where(closed, s_wrapped, s_clamped)  # [B, K]
    n = torch.round(length / ds).long()  # [B, 1]
    u = sq / ds
    i0 = torch.minimum(u.detach().floor().long(), n - 1)
    i1 = torch.where(closed, (i0 + 1) % n, i0 + 1)
    w = u - i0.to(u.dtype)  # [B, K]

    def g(x: Tensor, i: Tensor) -> Tensor:
        return torch.gather(x, 1, i.unsqueeze(-1).expand(-1, -1, x.shape[-1]) if x.dim() == 3 else i)

    xy = g(path.xy, i0) * (1 - w)[..., None] + g(path.xy, i1) * w[..., None]
    h0, h1 = g(path.heading, i0), g(path.heading, i1)
    heading = h0 + w * wrap_angle(h1 - h0)
    curv = g(path.curvature, i0) * (1 - w) + g(path.curvature, i1) * w
    return xy, heading, curv


def _fold_s(path: RefPath, s: Tensor) -> Tensor:
    # s [B, K] -> wrapped (closed) or clamped (open) into the path domain
    length = path.length_m[:, None]
    return torch.where(path.closed[:, None], torch.remainder(s, length), torch.minimum(s.clamp(min=0.0), length))


def project_to_path(path: RefPath, xy: Tensor, s_guess: Tensor, window_m: float) -> Tensor:
    # xy [B, 2], s_guess [B] -> s [B]: nearest point within +-window_m of s_guess (wrap-aware)
    ds = path_ds_m(path)
    n_side = int(torch.ceil(torch.tensor(window_m / ds)).item())
    offsets = torch.arange(-n_side, n_side + 1, device=xy.device, dtype=xy.dtype) * ds  # [K]
    cand = _fold_s(path, s_guess[:, None] + offsets[None])  # [B, K]
    p, h, _ = path_lookup(path, cand)
    d2 = ((p - xy[:, None]) ** 2).sum(-1)
    best = torch.gather(cand, 1, d2.argmin(dim=1, keepdim=True))  # [B, 1]
    p, h, _ = path_lookup(path, best)
    tang = torch.stack([torch.cos(h), torch.sin(h)], dim=-1)
    along = ((xy[:, None] - p) * tang).sum(-1).clamp(-ds, ds)  # first-order refinement inside one cell
    return _fold_s(path, best + along)[:, 0]


def project_global(path: RefPath, xy: Tensor) -> Tensor:
    # xy [B, K, 2] -> s [B, K]: nearest centerline point over the whole path (padding repeats the last point)
    ds = path_ds_m(path)
    idx = torch.cdist(xy, path.xy).argmin(dim=-1)  # [B, K]
    best = torch.gather(path.s, 1, idx)
    p, h, _ = path_lookup(path, best)
    tang = torch.stack([torch.cos(h), torch.sin(h)], dim=-1)
    along = ((xy - p) * tang).sum(-1).clamp(-ds, ds)
    return _fold_s(path, best + along)
