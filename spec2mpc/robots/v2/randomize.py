from __future__ import annotations

import zlib

import mujoco
import numpy as np


def sample_ranges(ranges: dict, seed: int | None, key: str = "") -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    # ranges: {params: {k: [lo, hi]}, perception: {k: [lo, hi]}, runtime: {k: [choices]}}
    # seed None -> nominal: midpoints and the first runtime choice; key decorrelates robots sharing a seed
    rng = None if seed is None else np.random.default_rng([seed, zlib.crc32(key.encode())])

    def uni(d: dict) -> dict[str, float]:
        if rng is None:
            return {k: 0.5 * (float(lo) + float(hi)) for k, (lo, hi) in d.items()}
        return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in d.items()}

    params, perception = uni(ranges["params"]), uni(ranges["perception"])
    runtime = {k: float(c[0] if rng is None else c[int(rng.integers(len(c)))]) for k, c in ranges["runtime"].items()}
    return params, perception, runtime


def apply_params(model: mujoco.MjModel, params: dict[str, float]) -> mujoco.MjModel:
    # in place on a fresh copy: one scalar per physical quantity
    s = params.get("mass_scale", 1.0)
    model.body_mass[:] *= s
    model.body_inertia[:] *= s
    model.geom_friction[:, 0] *= params.get("friction_scale", 1.0)
    model.dof_damping[:] *= params.get("damping_scale", 1.0)
    return model
